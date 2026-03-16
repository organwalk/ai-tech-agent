import argparse
import json
import random
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from services.rag_service import RAGService


def parse_args():
    parser = argparse.ArgumentParser(description="Build seed RAG evaluation dataset from existing vector DB.")
    parser.add_argument("--max-files", type=int, default=20, help="Maximum number of file_id groups to sample.")
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=2,
        help="Number of samples to generate per file_id.",
    )
    parser.add_argument(
        "--output",
        default="docs/rag_eval_seed.json",
        help="Output JSON path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def _guess_query(document: str) -> str:
    text = re.sub(r"\s+", " ", (document or "").strip())
    if not text:
        return "请根据这份资料概括核心观点（需要人工补充具体问题）。"

    snippet = text[:36]
    return f"围绕“{snippet}”这段内容，资料的核心观点是什么？"


def build_samples(records: List[Dict], max_files: int, samples_per_file: int, seed: int) -> List[Dict]:
    grouped = defaultdict(list)
    for record in records:
        file_id = str(record.get("file_id", "")).strip()
        if not file_id:
            continue
        grouped[file_id].append(record)

    file_ids = sorted(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(file_ids)
    selected_file_ids = file_ids[: max(0, max_files)]

    samples = []
    for file_id in selected_file_ids:
        candidates = grouped[file_id]
        if not candidates:
            continue

        rng.shuffle(candidates)
        chosen = candidates[: max(1, samples_per_file)]
        for item in chosen:
            chunk_id = str(item.get("id", "")).strip()
            document = item.get("document", "") or ""
            sample = {
                "query": _guess_query(document),
                "file_ids": [file_id],
                "relevant_file_ids": [file_id],
                "relevant_chunk_ids": [chunk_id] if chunk_id else [],
                "relevant_keywords": [],
                "needs_review": True,
            }
            samples.append(sample)
    return samples


def main():
    args = parse_args()
    rag_service = RAGService()
    records = rag_service.get_document_records()
    samples = build_samples(
        records=records,
        max_files=args.max_files,
        samples_per_file=args.samples_per_file,
        seed=args.seed,
    )

    payload = {
        "top_k": Config.RAG_EVAL_TOP_K,
        "candidate_k": Config.RAG_EVAL_CANDIDATE_K,
        "samples": samples,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_records": len(records),
            "max_files": args.max_files,
            "samples_per_file": args.samples_per_file,
            "seed": args.seed,
            "notes": "Auto-generated seeds. Please manually refine query text and labels before production benchmarking.",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated samples: {len(samples)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
