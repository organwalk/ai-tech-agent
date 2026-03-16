import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


def parse_args():
    parser = argparse.ArgumentParser(description="Split RAG evaluation report by retrieval filter strategy.")
    parser.add_argument("--input", required=True, help="Input detailed report JSON path from run_rag_eval.py")
    parser.add_argument(
        "--output",
        default="reports/rag_eval_test_book_8x10_split.json",
        help="Output split report JSON path.",
    )
    return parser.parse_args()


def _safe_avg(values: List[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _aggregate_subset(items: List[Dict], top_k: int) -> Dict:
    valid_items = [item for item in items if item.get("valid")]
    valid_count = len(valid_items)

    summary = {
        "top_k": top_k,
        "total_samples": len(items),
        "valid_samples": valid_count,
        "invalid_samples": len(items) - valid_count,
        "hit_rate_at_k": 0.0,
        "mrr_at_k": 0.0,
        "avg_ndcg_at_k": 0.0,
        "avg_recall_at_k": 0.0,
        "avg_first_hit_rank": 0.0,
    }

    if valid_count == 0:
        return summary

    summary["hit_rate_at_k"] = _safe_avg([float(item.get("hit_at_k", 0.0)) for item in valid_items])
    summary["mrr_at_k"] = _safe_avg([float(item.get("mrr_at_k", 0.0)) for item in valid_items])
    summary["avg_ndcg_at_k"] = _safe_avg([float(item.get("ndcg_at_k", 0.0)) for item in valid_items])
    summary["avg_recall_at_k"] = _safe_avg([float(item.get("recall_at_k", 0.0)) for item in valid_items])

    first_hit_ranks = [int(item["first_hit_rank"]) for item in valid_items if item.get("first_hit_rank")]
    summary["avg_first_hit_rank"] = _safe_avg([float(rank) for rank in first_hit_ranks])
    return summary


def _build_gate(summary: Dict, thresholds: Dict) -> Dict:
    valid_samples = int(summary.get("valid_samples", 0))
    fail_reasons: List[str] = []

    if valid_samples <= 0:
        fail_reasons.append("No valid samples for gate evaluation.")
    else:
        if float(summary.get("hit_rate_at_k", 0.0)) < float(thresholds["hit_rate_at_k_min"]):
            fail_reasons.append(
                f"hit_rate_at_k={summary.get('hit_rate_at_k')} < {thresholds['hit_rate_at_k_min']}"
            )
        if float(summary.get("mrr_at_k", 0.0)) < float(thresholds["mrr_at_k_min"]):
            fail_reasons.append(f"mrr_at_k={summary.get('mrr_at_k')} < {thresholds['mrr_at_k_min']}")
        if float(summary.get("avg_ndcg_at_k", 0.0)) < float(thresholds["ndcg_at_k_min"]):
            fail_reasons.append(
                f"avg_ndcg_at_k={summary.get('avg_ndcg_at_k')} < {thresholds['ndcg_at_k_min']}"
            )

    return {
        "enabled": True,
        "pass": len(fail_reasons) == 0,
        "thresholds": thresholds,
        "fail_reasons": fail_reasons,
    }


def _find_risks(name: str, summary: Dict) -> List[str]:
    risks = []
    if float(summary.get("avg_ndcg_at_k", 0.0)) > 1.0:
        risks.append(f"{name}: avg_ndcg_at_k > 1.0 (current metric definition may be non-standard)")
    if float(summary.get("avg_recall_at_k", 0.0)) > 1.0:
        risks.append(f"{name}: avg_recall_at_k > 1.0 (current metric definition may be non-standard)")
    return risks


def main():
    args = parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input report not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    sample_reports = payload.get("samples", [])
    overall_summary = payload.get("summary", {})

    top_k = int(overall_summary.get("top_k", 8))

    gate_payload = payload.get("gate", {})
    thresholds = gate_payload.get("thresholds", {}) or {
        "hit_rate_at_k_min": 0.5,
        "mrr_at_k_min": 0.3,
        "ndcg_at_k_min": 0.4,
    }

    open_subset = [item for item in sample_reports if not item.get("filter_file_ids")]
    filtered_subset = [item for item in sample_reports if item.get("filter_file_ids")]

    open_summary = _aggregate_subset(open_subset, top_k=top_k)
    filtered_summary = _aggregate_subset(filtered_subset, top_k=top_k)

    open_gate = _build_gate(open_summary, thresholds)
    filtered_gate = _build_gate(filtered_summary, thresholds)

    overall_pass = bool(gate_payload.get("pass", False))
    open_pass = bool(open_gate.get("pass", False))
    final_pass = overall_pass and open_pass

    verdict = "专项通过"
    if overall_pass and not open_pass:
        verdict = "过滤依赖型通过，不算专项通过"
    elif not overall_pass:
        verdict = "专项未通过"

    risks: List[str] = []
    risks.extend(_find_risks("overall", overall_summary))
    risks.extend(_find_risks("open_subset", open_summary))
    risks.extend(_find_risks("filtered_subset", filtered_summary))

    output_payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_report": str(input_path),
        "overall": {
            "summary": overall_summary,
            "gate": gate_payload,
        },
        "open_subset": {
            "summary": open_summary,
            "gate": open_gate,
        },
        "filtered_subset": {
            "summary": filtered_summary,
            "gate": filtered_gate,
        },
        "final_decision": {
            "overall_gate_pass": overall_pass,
            "open_subset_gate_pass": open_pass,
            "final_pass": final_pass,
            "verdict": verdict,
        },
        "risks": risks,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== split summary ===")
    print(json.dumps({
        "overall_pass": overall_pass,
        "open_subset_pass": open_pass,
        "final_pass": final_pass,
        "verdict": verdict,
        "open_samples": len(open_subset),
        "filtered_samples": len(filtered_subset),
        "risks": risks,
    }, ensure_ascii=False, indent=2))
    print(f"Saved split report to: {output_path}")


if __name__ == "__main__":
    main()
