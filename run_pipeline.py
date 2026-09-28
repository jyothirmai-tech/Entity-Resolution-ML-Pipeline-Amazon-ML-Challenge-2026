"""
Master Entry Point for Amazon ML Challenge 2026 Business Entity Resolution.
Executes the full inference pipeline:
1. Normalizes test records
2. Builds 7-channel inverted indexes partitioned by country
3. Retrieves multi-channel candidate pairs
4. Scores candidates with trained LightGBM model
5. Applies entity-level margin-constrained decision engine
6. Writes output/matching_results.tsv and output/candidate_pairs.tsv
"""

import os
import sys

# Ensure local imports resolve
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, BASE_DIR)

from amazon_entity_resolution.src.inference.run_test_inference import main as run_inference

if __name__ == "__main__":
    print("Executing Amazon ML Challenge 2026 Entity Resolution Pipeline...")
    run_inference()
