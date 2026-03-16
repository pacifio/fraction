"""CLI entry point for running Fraction benchmarks."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_report(report, judge_mode):
    """Print a summary of benchmark results."""
    overall = report.get("overall", {})
    config = report.get("config", {})
    print(f"  Compressor: {config.get('compressor_type', 'llmlingua2')}")
    print(f"  BLEU-1:     {overall.get('bleu1', 0):.4f}")
    print(f"  F1:         {overall.get('f1', 0):.4f}")
    if judge_mode in ("likert", "both"):
        print(f"  Judge(1-5): {overall.get('llm_judge', 0):.2f}")
    if judge_mode in ("binary", "both"):
        print(f"  Judge(0/1): {overall.get('binary_judge', 0):.4f}")
    print(f"  Questions:  {overall.get('total_questions', 0)}")
    latency = report.get("latency", {})
    print(f"  Add p50:    {latency.get('add_p50_ms', 0):.1f}ms")
    print(f"  Search p50: {latency.get('search_p50_ms', 0):.1f}ms")


def run_single(args, compressor_type, output_dir):
    """Run a single benchmark with the given compressor type."""
    from benchmarks.locomo import run_fraction_benchmark

    return run_fraction_benchmark(
        dataset_path=args.dataset,
        output_dir=output_dir,
        top_k=args.top_k,
        openai_model=args.openai_model,
        compression_rate=args.compression_rate,
        max_conversations=args.max_conversations,
        skip_llm_judge=args.skip_llm_judge,
        compressor_type=compressor_type,
        judge_mode=args.judge_mode,
        extractor_model=args.extractor_model,
    )


def print_comparison(reports, judge_mode):
    """Print a side-by-side comparison table of multiple benchmark runs."""
    print("\n" + "=" * 70)
    print("COMPARATIVE RESULTS")
    print("=" * 70)

    headers = ["Metric"] + [r["config"]["compressor_type"] for r in reports]
    rows = []

    rows.append(["BLEU-1"] + [f"{r['overall']['bleu1']:.4f}" for r in reports])
    rows.append(["F1"] + [f"{r['overall']['f1']:.4f}" for r in reports])
    if judge_mode in ("likert", "both"):
        rows.append(["Judge (1-5)"] + [f"{r['overall']['llm_judge']:.2f}" for r in reports])
    if judge_mode in ("binary", "both"):
        rows.append(["Judge (0/1)"] + [f"{r['overall']['binary_judge']:.4f}" for r in reports])
    rows.append(["Add p50 (ms)"] + [f"{r['latency']['add_p50_ms']:.1f}" for r in reports])
    rows.append(["Search p50 (ms)"] + [f"{r['latency']['search_p50_ms']:.1f}" for r in reports])
    rows.append(["Questions"] + [str(r["overall"]["total_questions"]) for r in reports])

    # Calculate column widths
    col_widths = [max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)]

    # Print table
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print(" | ".join(v.ljust(w) for v, w in zip(row, col_widths)))


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
    parser.add_argument(
        "--compressor_type",
        type=str,
        default="llmlingua2",
        choices=["llmlingua2", "self_info", "ensemble", "llm", "compare"],
        help="Compressor type, or 'compare' to run both llmlingua2 and llm side-by-side",
    )
    parser.add_argument(
        "--judge_mode",
        type=str,
        default="both",
        choices=["likert", "binary", "both"],
        help="LLM judge mode: likert (1-5), binary (0/1), or both",
    )
    parser.add_argument(
        "--extractor_model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for LLM fact extraction (used with --compressor_type llm)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.dataset):
        print(f"Error: Dataset not found at {args.dataset}")
        print("Please place the LoCoMo dataset (locomo10.json) in benchmarks/dataset/")
        print("You can download it from the LoCoMo HuggingFace repository.")
        sys.exit(1)

    if args.compressor_type == "compare":
        reports = []
        for comp_type in ["llmlingua2", "llm"]:
            out_dir = os.path.join(args.output, comp_type)
            print(f"\n{'=' * 60}")
            print(f"RUNNING BENCHMARK: {comp_type}")
            print(f"{'=' * 60}")
            report = run_single(args, comp_type, out_dir)
            reports.append(report)

            print(f"\n{'=' * 60}")
            print(f"RESULTS: {comp_type}")
            print(f"{'=' * 60}")
            print_report(report, args.judge_mode)

        print_comparison(reports, args.judge_mode)
    else:
        report = run_single(args, args.compressor_type, args.output)

        print("\n" + "=" * 60)
        print("BENCHMARK COMPLETE")
        print("=" * 60)
        print_report(report, args.judge_mode)


if __name__ == "__main__":
    main()
