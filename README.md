# Business Entity Resolution Pipeline

**Amazon ML Challenge 2026**  
**Team:** EntityResolvers  

## Overview
This package implements an end-to-end entity resolution pipeline across 3 noisy business record sources, optimizing the official macro $F_{0.5}$ metric on open-set countries (US, India, France).

## Architecture
1. **Multi-View Normalization:** Unicode NFKC, casefolding, accent decomposition, legal suffix standardization, domain artifact stripping, and address expansion.
2. **7-Channel Inverted Index Candidate Blocking:** Country-partitioned inverted index capturing exact names, compact/domain names, sorted tokens, rare discriminative tokens, token pairs, and address numeric-locality anchors.
3. **41 Pairwise Dense Features:** RapidFuzz SIMD string similarities, token & character n-grams, numeric address anchors, cross-field interactions, and channel provenance.
4. **LightGBM Matcher:** 250 GBDT trees trained on candidate-generated hard negatives with GroupKFold.
5. **Entity-Level Margin Decision Engine:** Selects multi-match candidates ($p \ge 0.35$ and $p \ge \max(P) - 0.25$) and identifies singletons ($92.98\%$ singleton accuracy, $0.9749$ precision).

## Reproduction Instructions

### 1. Environment Setup
```bash
python -m pip install -r requirements.txt
```

### 2. Run End-to-End Inference (Generates Submission Outputs)
From the repository root:
```bash
python -m code.business_entity_resolution.run_pipeline
```
Or directly:
```bash
python amazon_entity_resolution/src/inference/run_test_inference.py
```

Outputs will be written to:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

### 3. Validate Outputs
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
Exit code `0` confirms full compliance with the official challenge validator.
