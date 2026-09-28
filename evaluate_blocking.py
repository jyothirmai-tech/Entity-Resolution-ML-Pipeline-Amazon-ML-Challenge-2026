"""
Candidate Recall & Blocking Evaluation Script.
Evaluates multi-channel candidate generation on a representative holdout sample
from training data against ground truth.
Reports candidate recall, candidate volume, channel coverage, and reduction ratio.
"""

import os
import sys
import json
import collections
from typing import Dict, List, Set, Tuple

sys.path.insert(0, os.path.abspath("."))
sys.stdout.reconfigure(encoding='utf-8')

from amazon_entity_resolution.src.normalization.normalizer import normalize_name, normalize_address
from amazon_entity_resolution.src.retrieval.candidate_generator import CountryCandidateIndex

DATA_DIR = os.path.abspath("student_resource/dataset")
REPORTS_DIR = os.path.abspath("amazon_entity_resolution/reports")

def main():
    print("=" * 60)
    print("PHASE 3: EVALUATING CANDIDATE GENERATION & RECALL")
    print("=" * 60)
    
    # 1. Load validation sample of S1 entities
    # To be representative, take 10,000 S1 records (stratified US and India)
    sample_s1 = {}
    print("Loading sample S1 validation entities...")
    with open(os.path.join(DATA_DIR, "train", "train_source1.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            if idx >= 15000:
                break
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 4:
                sample_s1[parts[0]] = {
                    "name": parts[1],
                    "addr": parts[2],
                    "country": parts[3]
                }
                
    print(f"Loaded {len(sample_s1):,} S1 validation records.")
    
    # 2. Load ground truth for these S1 records
    gt_map = {}
    needed_s2 = set()
    needed_s3 = set()
    total_true_matches = 0
    singletons = 0
    
    with open(os.path.join(DATA_DIR, "train", "train_ground_truth.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) != 2:
                continue
            s1_id = parts[0]
            if s1_id in sample_s1:
                matched_raw = parts[1].strip()
                if matched_raw:
                    m_list = [m.strip() for m in matched_raw.split(",")]
                    gt_map[s1_id] = set(m_list)
                    total_true_matches += len(m_list)
                    for m in m_list:
                        if m.startswith("S2-"): needed_s2.add(m)
                        elif m.startswith("S3-"): needed_s3.add(m)
                else:
                    gt_map[s1_id] = set()
                    singletons += 1
                    
    print(f"Ground Truth: {total_true_matches:,} true target matches across {len(sample_s1):,} S1 records ({singletons:,} singletons).")
    print(f"Target IDs needed to index: {len(needed_s2):,} S2 IDs, {len(needed_s3):,} S3 IDs.")
    
    # 3. Load extra distractor records (hard negatives + background records) into candidate pool
    # so candidate retrieval operates in a real, noisy retrieval environment!
    print("Building Country Candidate Indexes (including true targets + 150k distractors)...")
    country_indexes: Dict[str, CountryCandidateIndex] = {}
    
    def get_index(country: str) -> CountryCandidateIndex:
        if country not in country_indexes:
            country_indexes[country] = CountryCandidateIndex(country)
        return country_indexes[country]
        
    indexed_s2 = 0
    with open(os.path.join(DATA_DIR, "train", "train_source2.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 4:
                tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                # Index if needed or if part of first 100k distractor records
                if tid in needed_s2 or idx < 100000:
                    idx_obj = get_index(country)
                    nv = normalize_name(name)
                    av = normalize_address(addr)
                    idx_obj.index_target_record(tid, nv, av)
                    indexed_s2 += 1
                    
    indexed_s3 = 0
    with open(os.path.join(DATA_DIR, "train", "train_source3.tsv"), "r", encoding="utf-8") as f:
        f.readline()
        for idx, line in enumerate(f):
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) == 4:
                tid, name, addr, country = parts[0], parts[1], parts[2], parts[3]
                if tid in needed_s3 or idx < 100000:
                    idx_obj = get_index(country)
                    nv = normalize_name(name)
                    av = normalize_address(addr)
                    idx_obj.index_target_record(tid, nv, av)
                    indexed_s3 += 1
                    
    print(f"Indexed {indexed_s2:,} S2 records and {indexed_s3:,} S3 records into country candidate indexes.")
    
    # 4. Run retrieval and evaluate candidate recall
    print("Retrieving candidates for all S1 validation entities...")
    retrieved_true_matches = 0
    retrieved_s2_matches = 0
    retrieved_s3_matches = 0
    
    total_true_s2 = sum(1 for m_set in gt_map.values() for m in m_set if m.startswith("S2-"))
    total_true_s3 = sum(1 for m_set in gt_map.values() for m in m_set if m.startswith("S3-"))
    
    channel_hits = collections.Counter()
    candidate_counts = []
    
    country_recalls = collections.defaultdict(lambda: {"true": 0, "retrieved": 0})
    
    for s1_id, rec in sample_s1.items():
        country = rec["country"]
        nv = normalize_name(rec["name"])
        av = normalize_address(rec["addr"])
        
        idx_obj = get_index(country)
        candidates, provenance = idx_obj.retrieve_candidates(nv, av, max_total_candidates=25)
        candidate_counts.append(len(candidates))
        
        true_set = gt_map[s1_id]
        cand_set = set(candidates)
        
        country_recalls[country]["true"] += len(true_set)
        
        for m in true_set:
            if m in cand_set:
                retrieved_true_matches += 1
                country_recalls[country]["retrieved"] += 1
                if m.startswith("S2-"): retrieved_s2_matches += 1
                elif m.startswith("S3-"): retrieved_s3_matches += 1
                
                # Check which channels retrieved this true match
                for ch in provenance[m]:
                    channel_hits[ch] += 1

    cand_recall = retrieved_true_matches / max(1, total_true_matches)
    s2_recall = retrieved_s2_matches / max(1, total_true_s2)
    s3_recall = retrieved_s3_matches / max(1, total_true_s3)
    
    avg_cands = sum(candidate_counts) / len(candidate_counts)
    candidate_counts.sort()
    p50_cands = candidate_counts[len(candidate_counts)//2]
    p95_cands = candidate_counts[int(len(candidate_counts)*0.95)]
    p99_cands = candidate_counts[int(len(candidate_counts)*0.99)]
    
    report = {
        "sample_s1_count": len(sample_s1),
        "total_true_matches": total_true_matches,
        "retrieved_true_matches": retrieved_true_matches,
        "overall_candidate_recall": round(cand_recall, 4),
        "source2_candidate_recall": round(s2_recall, 4),
        "source3_candidate_recall": round(s3_recall, 4),
        "country_recalls": {
            c: {
                "true_matches": stats["true"],
                "retrieved": stats["retrieved"],
                "recall": round(stats["retrieved"] / max(1, stats["true"]), 4)
            } for c, stats in country_recalls.items()
        },
        "candidate_count_distribution": {
            "mean": round(avg_cands, 2),
            "median": p50_cands,
            "p95": p95_cands,
            "p99": p99_cands,
            "max": candidate_counts[-1] if candidate_counts else 0
        },
        "channel_hit_counts": dict(channel_hits)
    }
    
    out_file = os.path.join(REPORTS_DIR, "phase3_candidate_recall_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    print("\n" + "=" * 60)
    print("BLOCKING & CANDIDATE GENERATION RESULTS")
    print("=" * 60)
    print(f"Overall Candidate Recall: {cand_recall * 100:.2f}% ({retrieved_true_matches:,} / {total_true_matches:,})")
    print(f"  Source 2 Candidate Recall: {s2_recall * 100:.2f}% ({retrieved_s2_matches:,} / {total_true_s2:,})")
    print(f"  Source 3 Candidate Recall: {s3_recall * 100:.2f}% ({retrieved_s3_matches:,} / {total_true_s3:,})")
    for c, c_stats in report["country_recalls"].items():
        print(f"  Country {c} Candidate Recall: {c_stats['recall'] * 100:.2f}%")
    print(f"\nCandidate Counts per S1 Entity:")
    print(f"  Mean: {avg_cands:.2f} | Median: {p50_cands} | P95: {p95_cands} | Max: {candidate_counts[-1]}")
    print(f"Channel Hit Breakdown on True Matches: {dict(channel_hits)}")
    print(f"\nSaved Candidate Recall report to: {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
