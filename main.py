"""
RouteMoA — CLI Entry Point
Run the full pipeline from the command line.

Usage:
    python main.py "What is the capital of France?"
    python main.py --top-k 5 --max-layers 4 "Explain quantum computing"
"""
import argparse
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from colorama import init as colorama_init
from config import fetch_model_costs, TOP_K, MAX_LAYERS, EARLY_STOP_THRESHOLD
from core.pipeline import RouteMoAPipeline


def main():
    colorama_init()

    parser = argparse.ArgumentParser(
        description="RouteMoA — Dynamic Routing Mixture of Agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "What is 2+2?"
  python main.py --top-k 5 "Write a Python function to sort a list"
  python main.py --max-layers 4 --slm "Explain the theory of relativity"
        """,
    )
    parser.add_argument("query", type=str, help="The question or prompt to answer")
    parser.add_argument("--top-k", type=int, default=TOP_K, help=f"Number of models per layer (default: {TOP_K})")
    parser.add_argument("--max-layers", type=int, default=MAX_LAYERS, help=f"Maximum pipeline layers (default: {MAX_LAYERS})")
    parser.add_argument("--threshold", type=float, default=EARLY_STOP_THRESHOLD, help=f"Early stopping threshold (default: {EARLY_STOP_THRESHOLD})")
    parser.add_argument("--slm", action="store_true", help="Use trained SLM scorer (requires trained model)")
    parser.add_argument("--no-pricing", action="store_true", help="Skip fetching live pricing from OpenRouter")

    args = parser.parse_args()

    # Fetch live pricing
    if not args.no_pricing:
        print("📡 Fetching live model pricing from OpenRouter...")
        fetch_model_costs()
        print()

    # Create pipeline
    pipeline = RouteMoAPipeline(
        top_k=args.top_k,
        max_layers=args.max_layers,
        early_stop_threshold=args.threshold,
        use_slm_scorer=args.slm,
    )

    # Run
    result = asyncio.run(pipeline.run(args.query))

    # Display final answer
    print(f"\n{'='*60}")
    print(f"📋 FINAL ANSWER")
    print(f"{'='*60}")
    print(result["final_answer"])
    print(f"\n{'='*60}")
    print(f"📊 STATISTICS")
    print(f"{'='*60}")
    stats = result["stats"]
    print(f"  💰 Total Cost:    ${stats['total_cost']:.6f}")
    print(f"  ⏱️  Total Time:    {stats['total_latency_ms']/1000:.2f}s")
    print(f"  📊 Layers Used:   {stats['num_layers']}")
    print(f"  📝 Tokens In:     {stats['total_tokens_in']:,}")
    print(f"  📝 Tokens Out:    {stats['total_tokens_out']:,}")


if __name__ == "__main__":
    main()
