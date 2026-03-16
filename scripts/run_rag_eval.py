import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import RAGEvalRequest
from services.rag_eval_service import RAGEvaluationService


def parse_args():
    parser = argparse.ArgumentParser(description="Run RAG retrieval evaluation.")
    parser.add_argument("--input", required=True, help="Path to JSON evaluation request payload.")
    parser.add_argument("--output", default="", help="Optional output path for full result JSON.")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    payload = json.loads(input_path.read_text(encoding="utf-8-sig"))
    request = RAGEvalRequest(**payload)
    eval_service = RAGEvaluationService()
    result = eval_service.evaluate(request)

    print("=== summary ===")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print("=== validation_summary ===")
    print(json.dumps(result.get("validation_summary", {}), ensure_ascii=False, indent=2))
    print("=== gate ===")
    print(json.dumps(result.get("gate", {}), ensure_ascii=False, indent=2))
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved detailed report to: {output_path}")


if __name__ == "__main__":
    main()
