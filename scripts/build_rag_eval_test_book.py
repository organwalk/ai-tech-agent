import argparse
import copy
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import Config
from services.llm_service import LLMService
from services.rag_service import RAGService

DEFAULT_TARGET_FILE_IDS = [
    "2026978738434932738",
    "2026980149155528706",
    "2026715484383019010",
    "2026335210100961282",
    "2026711954058313730",
    "2026698875018874882",
    "2033501956488540161",
    "2027255017583378434",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build 8x10 mixed-difficulty RAG eval set for docs/test_book.")
    parser.add_argument("--seed-input", default="docs/rag_eval_seed_all_10.json", help="Seed eval JSON path.")
    parser.add_argument(
        "--output",
        default="docs/rag_eval_test_book_8x10_mixed.json",
        help="Output eval JSON path.",
    )
    parser.add_argument(
        "--review-output",
        default="reports/rag_eval_test_book_8x10_spotcheck.json",
        help="Output spot-check checklist JSON path.",
    )
    parser.add_argument(
        "--target-file-ids",
        default=",".join(DEFAULT_TARGET_FILE_IDS),
        help="Comma-separated file_id list.",
    )
    parser.add_argument("--per-file", type=int, default=10, help="Samples per target file.")
    parser.add_argument("--open-per-file", type=int, default=7, help="Open retrieval samples per file.")
    parser.add_argument("--easy-per-file", type=int, default=5, help="Easy samples per file.")
    parser.add_argument("--review-count", type=int, default=16, help="Spot-check items sampled from hard queries.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--disable-llm-rewrite",
        action="store_true",
        help="Disable LLM rewrite and always use template fallback.",
    )
    return parser.parse_args()


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _parse_file_ids(raw: str) -> List[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def _read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _build_chunk_text_map() -> Dict[str, str]:
    rag_service = RAGService()
    records = rag_service.get_document_records()
    return {str(item.get("id", "")).strip(): str(item.get("document", "") or "") for item in records}


def _fallback_hard_query(source_query: str, snippet: str) -> str:
    clean_snippet = _normalize_text(snippet)[:30]
    if clean_snippet:
        return f"不直接复述原文的话，围绕“{clean_snippet}”这部分内容，作者的核心论证是什么？"
    clean_query = _normalize_text(source_query)
    return f"请换一种提问方式解释这个问题背后的核心论证：{clean_query}"


def _rewrite_hard_query(
    llm_service: Optional[LLMService],
    source_query: str,
    snippet: str,
) -> Dict:
    fallback_query = _fallback_hard_query(source_query, snippet)
    if llm_service is None:
        return {
            "query": fallback_query,
            "used_fallback": True,
            "runtime_failed": False,
        }

    prompt = (
        "请将下面问题改写为一个更难的中文提问，要求语义保持一致，但不要直接复述原句。\n"
        "要求:\n"
        "1) 只输出一个问题句，不要解释。\n"
        "2) 15-40字。\n"
        "3) 尽量使用抽象表达或换序表达。\n"
        f"原问题: {source_query}\n"
        f"材料片段: {snippet[:120]}"
    )

    try:
        response = llm_service.generate_response(
            messages=[
                {"role": "system", "content": "你是中文教学场景中的问题改写助手。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
        )
        clean = llm_service.clean_markdown_format(str(response or "")).strip()
        clean = clean.splitlines()[0].strip() if clean else ""
        if clean:
            return {
                "query": clean,
                "used_fallback": False,
                "runtime_failed": False,
            }
    except Exception:
        return {
            "query": fallback_query,
            "used_fallback": True,
            "runtime_failed": True,
        }

    return {
        "query": fallback_query,
        "used_fallback": True,
        "runtime_failed": False,
    }


def _group_seed_samples(seed_payload: Dict, target_file_ids: List[str]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {fid: [] for fid in target_file_ids}
    for sample in seed_payload.get("samples", []):
        relevant_file_ids = [str(item).strip() for item in sample.get("relevant_file_ids", []) if str(item).strip()]
        if not relevant_file_ids:
            continue
        primary_file_id = relevant_file_ids[0]
        if primary_file_id in grouped:
            grouped[primary_file_id].append(sample)
    return grouped


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    target_file_ids = _parse_file_ids(args.target_file_ids)
    if not target_file_ids:
        raise ValueError("target-file-ids cannot be empty")

    if args.per_file <= 0:
        raise ValueError("per-file must be > 0")
    if args.open_per_file < 0 or args.open_per_file > args.per_file:
        raise ValueError("open-per-file must be in [0, per-file]")
    if args.easy_per_file < 0 or args.easy_per_file > args.per_file:
        raise ValueError("easy-per-file must be in [0, per-file]")

    seed_path = Path(args.seed_input)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed input not found: {seed_path}")

    seed_payload = _read_json(seed_path)
    grouped = _group_seed_samples(seed_payload, target_file_ids)

    for fid in target_file_ids:
        sample_count = len(grouped.get(fid, []))
        if sample_count < args.per_file:
            raise ValueError(f"file_id={fid} has only {sample_count} seed samples, need >= {args.per_file}")

    chunk_text_map = _build_chunk_text_map()
    llm_service = None if args.disable_llm_rewrite else LLMService()
    llm_runtime_failed = False

    final_samples: List[Dict] = []
    hard_review_pool: List[Dict] = []

    hard_rewrite_attempted = 0
    hard_rewrite_fallback = 0

    for fid in target_file_ids:
        candidates = grouped[fid][:]
        rng.shuffle(candidates)
        chosen = candidates[: args.per_file]

        for idx, sample in enumerate(chosen):
            sample_copy = copy.deepcopy(sample)
            chunk_ids = [str(item).strip() for item in sample_copy.get("relevant_chunk_ids", []) if str(item).strip()]
            chunk_id = chunk_ids[0] if chunk_ids else ""
            snippet = _normalize_text(chunk_text_map.get(chunk_id, ""))[:120]

            source_query = str(sample_copy.get("query", "")).strip()
            is_hard = idx >= args.easy_per_file

            if is_hard:
                hard_rewrite_attempted += 1
                if llm_runtime_failed:
                    rewritten = _fallback_hard_query(source_query, snippet)
                    hard_rewrite_fallback += 1
                else:
                    rewrite_result = _rewrite_hard_query(
                        llm_service=llm_service,
                        source_query=source_query,
                        snippet=snippet,
                    )
                    rewritten = rewrite_result["query"]
                    if rewrite_result["used_fallback"]:
                        hard_rewrite_fallback += 1
                    if rewrite_result["runtime_failed"]:
                        llm_runtime_failed = True
                sample_copy["query"] = rewritten
            else:
                sample_copy["query"] = source_query

            if idx < args.open_per_file:
                sample_copy["file_ids"] = []
            else:
                sample_copy["file_ids"] = [fid]

            sample_copy["needs_review"] = is_hard
            final_samples.append(sample_copy)

            if is_hard:
                hard_review_pool.append(
                    {
                        "file_id": fid,
                        "chunk_id": chunk_id,
                        "query": sample_copy["query"],
                        "source_query": source_query,
                        "source_excerpt": snippet,
                    }
                )

    for index, sample in enumerate(final_samples, start=1):
        sample["sample_index"] = index

    review_count = min(max(args.review_count, 0), len(hard_review_pool))
    review_items = rng.sample(hard_review_pool, review_count) if review_count > 0 else []
    for index, item in enumerate(review_items, start=1):
        item["review_id"] = index
        item["review_status"] = "pending"
        item["is_semantically_consistent"] = None
        item["review_comment"] = ""

    output_payload = {
        "top_k": seed_payload.get("top_k", Config.RAG_EVAL_TOP_K),
        "candidate_k": seed_payload.get("candidate_k", Config.RAG_EVAL_CANDIDATE_K),
        "samples": final_samples,
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": "test_book_8x10_mixed",
            "target_file_ids": target_file_ids,
            "per_file": args.per_file,
            "open_per_file": args.open_per_file,
            "filtered_per_file": args.per_file - args.open_per_file,
            "easy_per_file": args.easy_per_file,
            "hard_per_file": args.per_file - args.easy_per_file,
            "hard_rewrite_attempted": hard_rewrite_attempted,
            "hard_rewrite_fallback": hard_rewrite_fallback,
            "llm_runtime_failed": llm_runtime_failed,
            "notes": "70% open retrieval + 30% file-filtered retrieval, mixed difficulty.",
        },
    }

    review_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "test_book_8x10_mixed",
        "review_goal": "hard_query_semantic_alignment",
        "required_min_consistency_rate": 0.90,
        "sample_size": len(review_items),
        "items": review_items,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    review_path = Path(args.review_output)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = len(final_samples)
    open_count = sum(1 for item in final_samples if not item.get("file_ids"))
    filtered_count = total - open_count

    print(f"Generated samples: {total}")
    print(f"Open samples: {open_count}")
    print(f"Filtered samples: {filtered_count}")
    print(f"Hard rewrite attempted: {hard_rewrite_attempted}, fallback used: {hard_rewrite_fallback}")
    print(f"Saved dataset to: {output_path}")
    print(f"Saved spot-check file to: {review_path}")


if __name__ == "__main__":
    main()
