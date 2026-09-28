"""
Multi-Channel Candidate Generation Engine for Business Entity Resolution.
Constructs high-recall inverted index channels per country:
1. Exact normalized name (no legal suffix)
2. Compact alphanumeric name (captures domain names and attached words)
3. Sorted tokens name (captures word order permutations)
4. Rare discriminative name tokens
5. Name token pairs (2-token combinations for robust recall under token insertion/deletion)
6. Address numeric anchor + locality/city token (handles transliteration / regional scripts / DBA names)
7. Rare address tokens (distinct street/building/colony names)

Ensures open-set country partitioning, bounded candidate depth per S1,
and tracks channel provenance for downstream features.
"""

import collections
import re
from typing import Dict, List, Set, Tuple, Optional, Any
from amazon_entity_resolution.src.normalization.normalizer import (
    normalize_name, normalize_address, BusinessNameViews, BusinessAddressViews
)

# Common generic stop words in business names that should not be used as single-token keys
GENERIC_STOPWORDS = {
    'the', 'and', 'for', 'of', 'in', 'at', 'on', 'with', 'to', 'from',
    'shop', 'store', 'center', 'centre', 'enterprises', 'enterprise',
    'solutions', 'services', 'industries', 'group', 'holdings', 'agency',
    'associates', 'trading', 'consulting', 'management', 'international',
    'india', 'bharat', 'american', 'national', 'global', 'delhi', 'mumbai',
    'bangalore', 'chennai', 'kolkata', 'hyderabad', 'texas', 'california',
    'hotel', 'restaurant', 'cafe', 'dhaba', 'mart', 'supermarket', 'bazaar',
    'point', 'corner', 'hub', 'house', 'plaza', 'bhavan', 'nilayam',
    'dept', 'division', 'office', 'branch', 'works', 'factory', 'mill'
}

GENERIC_ADDR_WORDS = {
    'road', 'rd', 'street', 'st', 'lane', 'ln', 'drive', 'dr', 'avenue', 'ave',
    'boulevard', 'blvd', 'court', 'ct', 'floor', 'fl', 'suite', 'ste', 'unit',
    'apartment', 'apt', 'block', 'plot', 'sector', 'phase', 'stage', 'main',
    'cross', 'near', 'opposite', 'behind', 'beside', 'adjacent', 'opp', 'b/h',
    'north', 'south', 'east', 'west', 'city', 'town', 'village', 'district',
    'state', 'country', 'us', 'usa', 'india', 'france', 'rue', 'de', 'la', 'le'
}

class InvertedIndexChannel:
    """Memory-efficient Inverted Index mapping blocking keys to target IDs."""
    def __init__(self, name: str, max_postings: int = 40):
        self.name = name
        self.max_postings = max_postings
        self.index: Dict[str, List[str]] = collections.defaultdict(list)

    def add(self, key: str, target_id: str):
        if not key or len(key) < 2:
            return
        postings = self.index[key]
        if len(postings) < self.max_postings:
            postings.append(target_id)

    def query(self, key: str) -> List[str]:
        if not key or key not in self.index:
            return []
        postings = self.index[key]
        if len(postings) >= self.max_postings:
            # Overly generic key, omit to avoid false positive flooding
            return []
        return postings

    def clear(self):
        self.index.clear()

class CountryCandidateIndex:
    """Manages all retrieval channels for a specific country."""
    def __init__(self, country: str):
        self.country = country
        self.ch_exact = InvertedIndexChannel("exact_name", max_postings=40)
        self.ch_compact = InvertedIndexChannel("compact_name", max_postings=40)
        self.ch_sorted = InvertedIndexChannel("sorted_name", max_postings=40)
        self.ch_rare = InvertedIndexChannel("rare_token", max_postings=35)
        self.ch_token_pair = InvertedIndexChannel("name_token_pair", max_postings=30)
        self.ch_addr_anchor = InvertedIndexChannel("addr_num_loc", max_postings=30)
        self.ch_rare_addr = InvertedIndexChannel("rare_addr", max_postings=30)
        
        self.total_records = 0

    def index_target_record(self, target_id: str, name_views: BusinessNameViews, addr_views: BusinessAddressViews):
        self.total_records += 1
        
        # Channel 1: Exact no_suffix
        if name_views.no_suffix and len(name_views.no_suffix) >= 3:
            self.ch_exact.add(name_views.no_suffix, target_id)
            
        # Channel 2: Compact alphanumeric name
        if name_views.compact and len(name_views.compact) >= 4:
            self.ch_compact.add(name_views.compact, target_id)
            
        # Channel 3: Sorted tokens name
        if name_views.no_suffix_sorted and len(name_views.no_suffix_sorted) >= 3:
            self.ch_sorted.add(name_views.no_suffix_sorted, target_id)
            
        # Channel 4: Discriminative name tokens
        valid_name_tokens = [
            t for t in name_views.tokens 
            if len(t) >= 3 and t not in GENERIC_STOPWORDS and not t.isdigit()
        ]
        for t in valid_name_tokens:
            self.ch_rare.add(t, target_id)
            
        # Channel 5: Name token pairs (robust to single-token insertions/deletions)
        if len(valid_name_tokens) >= 2:
            # Index adjacent or first pairs
            for i in range(min(3, len(valid_name_tokens) - 1)):
                pair_key = f"{valid_name_tokens[i]}_{valid_name_tokens[i+1]}"
                self.ch_token_pair.add(pair_key, target_id)
                
        # Channel 6: Address Number + Locality/City Anchor (crucial for transliterated/DBA names!)
        if addr_views.numeric_tokens and addr_views.tokens:
            primary_num = addr_views.numeric_tokens[0]
            # Find candidate locality/street tokens (non-generic, non-numeric)
            addr_loc_tokens = [
                t for t in addr_views.tokens 
                if len(t) >= 4 and t not in GENERIC_ADDR_WORDS and not t.isdigit()
            ]
            for lt in addr_loc_tokens[:2]:
                anchor_key = f"{primary_num}_{lt}"
                self.ch_addr_anchor.add(anchor_key, target_id)
                
        # Channel 7: Rare address tokens (e.g. building name, specific colony)
        for t in addr_views.tokens:
            if len(t) >= 6 and t not in GENERIC_ADDR_WORDS and not t.isdigit():
                self.ch_rare_addr.add(t, target_id)

    def retrieve_candidates(
        self,
        name_views: BusinessNameViews,
        addr_views: BusinessAddressViews,
        max_total_candidates: int = 35
    ) -> Tuple[List[str], Dict[str, Set[str]]]:
        """
        Retrieves union of candidates across all channels.
        Returns:
            candidates: list of unique candidate target_ids (ordered by priority)
            provenance: dict mapping candidate_id to set of channel names that retrieved it
        """
        provenance = collections.defaultdict(set)
        
        # 1. Exact matches
        if name_views.no_suffix and len(name_views.no_suffix) >= 3:
            for tid in self.ch_exact.query(name_views.no_suffix):
                provenance[tid].add("exact_name")
                
        # 2. Compact name matches
        if name_views.compact and len(name_views.compact) >= 4:
            for tid in self.ch_compact.query(name_views.compact):
                provenance[tid].add("compact_name")
                
        # 3. Sorted tokens matches
        if name_views.no_suffix_sorted and len(name_views.no_suffix_sorted) >= 3:
            for tid in self.ch_sorted.query(name_views.no_suffix_sorted):
                provenance[tid].add("sorted_name")
                
        # 4. Rare token matches
        valid_name_tokens = [
            t for t in name_views.tokens 
            if len(t) >= 3 and t not in GENERIC_STOPWORDS and not t.isdigit()
        ]
        for t in valid_name_tokens:
            for tid in self.ch_rare.query(t):
                provenance[tid].add("rare_token")
                
        # 5. Token pair matches
        if len(valid_name_tokens) >= 2:
            for i in range(min(3, len(valid_name_tokens) - 1)):
                pair_key = f"{valid_name_tokens[i]}_{valid_name_tokens[i+1]}"
                for tid in self.ch_token_pair.query(pair_key):
                    provenance[tid].add("name_token_pair")
                    
        # 6. Address anchor matches
        if addr_views.numeric_tokens and addr_views.tokens:
            primary_num = addr_views.numeric_tokens[0]
            addr_loc_tokens = [
                t for t in addr_views.tokens 
                if len(t) >= 4 and t not in GENERIC_ADDR_WORDS and not t.isdigit()
            ]
            for lt in addr_loc_tokens[:2]:
                anchor_key = f"{primary_num}_{lt}"
                for tid in self.ch_addr_anchor.query(anchor_key):
                    provenance[tid].add("addr_num_loc")
                    
        # 7. Rare address tokens
        for t in addr_views.tokens:
            if len(t) >= 6 and t not in GENERIC_ADDR_WORDS and not t.isdigit():
                for tid in self.ch_rare_addr.query(t):
                    provenance[tid].add("rare_addr")

        # Rank candidates by channel agreement count & high-confidence channels
        sorted_candidates = sorted(
            provenance.keys(),
            key=lambda tid: (
                len(provenance[tid]),
                "exact_name" in provenance[tid],
                "compact_name" in provenance[tid],
                "addr_num_loc" in provenance[tid]
            ),
            reverse=True
        )
        
        # Bounded candidate set
        top_candidates = sorted_candidates[:max_total_candidates]
        filtered_provenance = {tid: provenance[tid] for tid in top_candidates}
        
        return top_candidates, filtered_provenance
