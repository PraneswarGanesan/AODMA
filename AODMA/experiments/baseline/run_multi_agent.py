
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from agents.diffusion_runtime import load_diffusion_runtime
from agents.orchestrator import run_aodma


def main():
    parser = argparse.ArgumentParser(
        description="Run the AODMA multi-agent diffusion workflow."
    )

    parser.add_argument(
        "--portfolio",
        default="results/cifar10/bqrps/unified_portfolio.json",
    )

    parser.add_argument(
        "--checkpoint",
        default="checkpoints/ddpm_cifar10/latest.pt",
    )

    parser.add_argument(
        "--config",
        default="experiments/configs/ddpm_cifar10.yaml",
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=350.0,
    )

    parser.add_argument(
        "--max-memory-mb",
        type=float,
        default=40.0,
    )

    parser.add_argument(
        "--max-nfe",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--selection-method",
        choices=[
            "bqrps",
            "best_fid",
            "fastest",
            "lowest_nfe",
        ],
        default="bqrps",
    )

    parser.add_argument(
        "--output-dir",
        default="results/agentic",
    )

    args = parser.parse_args()

    # if args.samples < 1000:
    #     raise ValueError(
    #         "Use at least 1000 samples for the configured KID evaluator."
    #     )

    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")

    portfolio_path = Path(args.portfolio)

    if not portfolio_path.exists():
        raise FileNotFoundError(
            f"Portfolio not found: {portfolio_path}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("AODMA MULTI-AGENT DIFFUSION EXPERIMENT")
    print("=" * 65)

    print("\n[1/3] Loading diffusion runtime...")

    runtime = load_diffusion_runtime(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
    )

    print("Device:", runtime["device"])
    print("Checkpoint:", args.checkpoint)

    print("\n[2/3] Starting LangGraph agent workflow...")

    result = run_aodma(
        portfolio_path=portfolio_path,
        runtime=runtime,
        budget={
            "max_latency_ms": args.max_latency_ms,
            "max_memory_mb": args.max_memory_mb,
            "max_nfe": args.max_nfe,
        },
        selection_method=args.selection_method,
        samples=args.samples,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print("\n[3/3] Saving experiment outputs...")

    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    report_path = output_dir / f"aodma_report_{timestamp}.md"
    result_path = output_dir / f"aodma_result_{timestamp}.json"

    report_path.write_text(
        result.get("final_report", "No report was generated."),
        encoding="utf-8",
    )

    serializable_result = {
        "experiment": result.get("experiment_name"),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": result.get("status"),
        "messages": result.get("messages", []),
        "research_summary": result.get("research_summary"),
        "selection": result.get("selection_result"),
        "execution": result.get("execution_result"),
        "evaluation": result.get("evaluation_result"),
        "final_report_path": str(report_path),
    }

    result_path.write_text(
        json.dumps(
            serializable_result,
            indent=4,
            default=str,
        ),
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("AODMA WORKFLOW COMPLETED")
    print("=" * 65)

    print("\nSelected candidate:")
    print(result.get("selected_candidate"))

    print("\nExecution:")
    print(result.get("execution_result"))

    print("\nEvaluation:")
    print(result.get("evaluation_result"))

    print("\nReport:", report_path)
    print("Results:", result_path)

    print("\nAgent messages:")
    for message in result.get("messages", []):
        print("-", message)


if __name__ == "__main__":
    main()
