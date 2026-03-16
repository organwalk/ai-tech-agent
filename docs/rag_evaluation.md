# RAG Evaluation Plan

## 1. Goals
- Measure retrieval precision and ranking quality for knowledge search.
- Detect invalid labels early (before metric calculation) to avoid misleading all-zero or false-high results.
- Provide reproducible online and offline evaluation workflows with pass/fail gate output.

## 2. Primary Relevance Criterion
- Primary criterion is fixed to `file_id`.
- A retrieval hit is considered relevant when `hit.metadata.file_id` is in `sample.relevant_file_ids`.
- `relevant_chunk_ids` and `relevant_keywords` are preserved as auxiliary diagnostics only (not used for gate pass/fail).

## 3. Data Format
Use `/api/v1/agent/rag/evaluate` request schema:

```json
{
  "top_k": 8,
  "candidate_k": 24,
  "samples": [
    {
      "query": "围绕某个知识点提问",
      "file_ids": ["2026335210100961282"],
      "relevant_file_ids": ["2026335210100961282"],
      "relevant_chunk_ids": ["2026335210100961282_chunk_3"],
      "relevant_keywords": ["示例关键词"],
      "needs_review": true
    }
  ]
}
```

### Required for valid sample
- `query` must be non-empty.
- `relevant_file_ids` must be non-empty and must exist in current vector DB.

### Optional
- `file_ids` retrieval filter.
- `relevant_chunk_ids`, `relevant_keywords` for auxiliary diagnostics.
- `needs_review` for dataset workflow management.

## 4. Validation & Invalid Samples
Evaluation performs pre-checks before retrieval:
- unknown `file_ids`
- unknown `relevant_file_ids`
- unknown `relevant_chunk_ids`
- chunk-file ownership mismatch (`relevant_chunk_ids` not belonging to `relevant_file_ids`)
- no overlap between `file_ids` and `relevant_file_ids`

Invalid samples:
- marked as `valid=false`
- include `reason` and `reason_codes`
- excluded from metric denominator

Response includes:
- `validation_summary.total_samples`
- `validation_summary.valid_samples`
- `validation_summary.invalid_samples`
- `validation_summary.reason_counts`

## 5. Metrics and Gate
Core metrics (primary/file_id):
- `HitRate@K`
- `MRR@K`
- `nDCG@K`
- `Recall@K`

Auxiliary diagnostics:
- chunk-based hit metrics
- keyword-based hit metrics

Gate (default enabled):
- `HitRate@K >= 0.50`
- `MRR@K >= 0.30`
- `nDCG@K >= 0.40`

Response includes `gate`:
- `enabled`
- `pass`
- `thresholds`
- `fail_reasons`

## 6. Execution
### API
- `POST /api/v1/agent/rag/evaluate`

### CLI evaluation
```bash
python scripts/run_rag_eval.py --input docs/rag_eval_example.json --output reports/rag_eval_report.json
```

### Auto-generate seed dataset
```bash
python scripts/build_rag_eval_seed.py --max-files 20 --samples-per-file 2 --output docs/rag_eval_seed.json
```

Generated samples are `needs_review=true`; manually refine query wording/labels before formal benchmarking.

## 7. Continuous Practice
- Always run validation + baseline before changing chunking/embedding/ranking configs.
- Compare old/new metrics on the same sample set.
- Track `validation_summary.reason_counts` to catch dataset quality drift.
- If `valid_samples > 0` and all core metrics are zero, inspect filter constraints and label-file alignment first.

## 8. Test Book 8x10 Workflow
Use this workflow for the `docs/test_book` 8-article专项测试:

```bash
# 1) Build all-file seeds (10 per file)
python scripts/build_rag_eval_seed.py --max-files 200 --samples-per-file 10 --seed 42 --output docs/rag_eval_seed_all_10.json

# 2) Build 8x10 mixed dataset (70% open + 30% file-filtered, mixed difficulty)
python scripts/build_rag_eval_test_book.py --seed-input docs/rag_eval_seed_all_10.json --output docs/rag_eval_test_book_8x10_mixed.json --review-output reports/rag_eval_test_book_8x10_spotcheck.json

# 3) Run evaluation
python scripts/run_rag_eval.py --input docs/rag_eval_test_book_8x10_mixed.json --output reports/rag_eval_test_book_8x10_report.json

# 4) Split by open/filtered subsets and produce dual-gate decision
python scripts/split_rag_eval_report.py --input reports/rag_eval_test_book_8x10_report.json --output reports/rag_eval_test_book_8x10_split.json
```
