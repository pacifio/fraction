"""CLI entry point for running Fraction benchmarks."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Run Fraction benchmarks")
    parser.add_argument(
        "--dataset",
        type=str,
        default="benchmarks/dataset/locomo10.json",
        help="Path to LoCoMo dataset JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports",
        help="Output directory for results",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=15,
        help="Number of memories to retrieve per query",
    )
    parser.add_argument(
        "--openai_model",
        type=str,
        default="gpt-4o",
        help="OpenAI model for answer generation",
    )
    parser.add_argument(
        "--compression_rate",
        type=float,
        default=0.6,
        help="Token compression rate (0.6 = keep 60%%)",
    )
    parser.add_argument(
        "--max_conversations",
        type=int,
        default=None,
        help="Limit number of conversations to process",
    )
    parser.add_argument(
        "--skip_llm_judge",
        action="store_true",
        help="Skip LLM judge scoring (faster, uses only BLEU/F1)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        print("Please place the LoCoMo dataset (locomo10.json) in benchmarks/dataset/")
        print("You can download it from the LoCoMo HuggingFace repository.")
        sys.exit(1)

    from benchmarks.locomo import run_fraction_benchmark

    report = run_fraction_benchmark(
        dataset_path=args.dataset,
        output_dir=args.output,
        top_k=args.top_k,
        openai_model=args.openai_model,
        compression_rate=args.compression_rate,
        max_conversations=args.max_conversations,
        skip_llm_judge=args.skip_llm_judge,
    )

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)
    overall = report.get("overall", {})
    print(f"  BLEU-1:     {overall.get('bleu1', 0):.4f}")
    print(f"  F1:         {overall.get('f1', 0):.4f}")
    print(f"  LLM Judge:  {overall.get('llm_judge', 0):.2f}")
    print(f"  Questions:  {overall.get('total_questions', 0)}")
    latency = report.get("latency", {})
    print(f"  Add p50:    {latency.get('add_p50_ms', 0):.1f}ms")
    print(f"  Search p50: {latency.get('search_p50_ms', 0):.1f}ms")


if __name__ == "__main__":
    main()
