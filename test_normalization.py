"""
Unit tests for multi-view normalization edge cases:
- Accents & Unicode diacritics (France, India)
- Legal suffixes across US, India, France
- Domain name noise in S3
- Word order permutations
- Address abbreviations and landmark prefixes
- Missing or empty strings
- Numeric token extraction
"""

import unittest
from amazon_entity_resolution.src.normalization.normalizer import (
    normalize_name, normalize_address, remove_accents, strip_domain_artifacts
)

class TestNormalization(unittest.TestCase):

    def test_unicode_accents_french(self):
        name = "SCI Ptit Àmicale"
        views = normalize_name(name)
        self.assertEqual(views.clean_ascii, "sci ptit amicale")
        self.assertIn("amicale", views.tokens)
        self.assertEqual(views.compact, "sciptitamicale")

    def test_domain_name_noise(self):
        name = "celestialmemorialtrust.com"
        views = normalize_name(name)
        self.assertEqual(views.clean_ascii, "celestialmemorialtrust")
        self.assertEqual(views.compact, "celestialmemorialtrust")

    def test_legal_suffix_stripping(self):
        # US
        v_us = normalize_name("Acme Logistics Corporation LLC")
        self.assertEqual(v_us.no_suffix, "acme logistics")
        
        # India
        v_in = normalize_name("Nyasa Healthcare Private Limited")
        self.assertEqual(v_in.no_suffix, "nyasa healthcare")
        
        # France
        v_fr = normalize_name("Thermal & Fils SASU")
        self.assertEqual(v_fr.no_suffix, "thermal")

    def test_word_order_sorted_tokens(self):
        v1 = normalize_name("Vision Partners Corp")
        v2 = normalize_name("Partners Vision Inc")
        self.assertEqual(v1.sorted_tokens, "corp partners vision")
        self.assertEqual(v1.no_suffix_sorted, v2.no_suffix_sorted)
        self.assertEqual(v1.no_suffix_sorted, "partners vision")

    def test_address_abbreviations(self):
        addr = "787 Ronny Ct., Ste. 101, Crown Point, IN"
        views = normalize_address(addr)
        self.assertIn("court", views.tokens)
        self.assertIn("suite", views.tokens)
        self.assertIn("787", views.numeric_tokens)
        self.assertIn("101", views.numeric_tokens)

    def test_empty_and_none(self):
        v_none = normalize_name(None)
        self.assertEqual(v_none.raw, "")
        self.assertEqual(len(v_none.tokens), 0)
        
        a_none = normalize_address("")
        self.assertEqual(a_none.raw, "")
        self.assertEqual(len(a_none.tokens), 0)

if __name__ == '__main__':
    unittest.main()
