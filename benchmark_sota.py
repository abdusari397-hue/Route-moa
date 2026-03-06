import asyncio
import time
import os
import sys
import random
from colorama import init, Fore, Style

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.pipeline import RouteMoAPipeline
from core.model_pool import OpenRouterModel
from config import fetch_model_costs
from training.data_collector import collect_mmlu, collect_mbpp, collect_arc_challenge

async def benchmark():
    init()
    print(f"{Fore.CYAN}={'='*70}")
    print(f"🌍 RouteMoA vs. SOTA (Claude Opus 4.6) Benchmark Test")
    print(f"   [Evaluating on Hard Benchmarks: MMLU & MBPP & ARC-C]")
    print(f"{'='*70}{Style.RESET_ALL}\n")

    print("Fetching live prices for all models...")
    fetch_model_costs()
    
    # Load questions from various hard datasets
    print("\nLoading benchmark questions from standard datasets...")
    benchmark_questions = []
    
    # Load 1 MBPP (Coding)
    mbpp = collect_mbpp(max_samples=10)
    if mbpp: benchmark_questions.append(random.choice(mbpp))
        
    # Load 1 MMLU (Complex Reasoning / Knowledge)
    mmlu = collect_mmlu(max_samples=10)
    if mmlu: benchmark_questions.append(random.choice(mmlu))
        
    # Load 1 ARC-Challenge (Science / Logic)
    arc = collect_arc_challenge(max_samples=10)
    if arc: benchmark_questions.append(random.choice(arc))
        
    if not benchmark_questions:
        print(f"{Fore.RED}Failed to load datasets. Make sure you have 'datasets' installed.{Style.RESET_ALL}")
        return

    # RouteMoA Pipeline (with SLM)
    # Using a quiet log_callback to prevent spamming the console 
    # during the benchmark.
    def silent_logger(msg):
        pass # print(msg) if you want full logs
        
    pipeline_slm = RouteMoAPipeline(
        top_k=3, 
        max_layers=3, 
        use_slm_scorer=True,
        log_callback=silent_logger # Mute standard execution logs to keep terminal clean
    )
    
    # Claude Opus 4.6
    # Using the OpenRouter ID provided by the user
    claude = OpenRouterModel("anthropic/claude-opus-4.6")
    # Manually inject model info for cost calculation (using assumed values for a high-end model)
    claude.model_info = {"name": "Claude Opus 4.6", "input_cost_per_mtok": 15.0, "output_cost_per_mtok": 75.0}
    claude.name = "Claude Opus 4.6"
    
    for i, sample in enumerate(benchmark_questions):
        q = sample["question"]
        print(f"\n{Fore.YELLOW}▶ Question {i+1}/{len(benchmark_questions)}:{Style.RESET_ALL}\n{q}\n")
        
        # ────────────── 1. RouteMoA ──────────────
        print(f"{Fore.GREEN}⏳ Running RouteMoA (SLM Routing over {len(pipeline_slm.model_pool)} models)...{Style.RESET_ALL}")
        res_moa = await pipeline_slm.run(q)
        moa_answer = res_moa["final_answer"]
        moa_cost = res_moa["stats"]["total_cost"]
        moa_time = res_moa["stats"]["total_latency_ms"] / 1000
        moa_layers = res_moa["stats"]["num_layers"]
        
        # ────────────── 2. Claude Opus 4.6 ──────────────
        print(f"{Fore.BLUE}⏳ Running Claude Opus 4.6 (Direct Baseline)...{Style.RESET_ALL}")
        t0 = time.perf_counter()
        raw_claude = await claude.generate(q, max_tokens=2048)
        c_time = time.perf_counter() - t0
        c_ans = raw_claude["answer"] if not raw_claude["error"] else raw_claude["error"]
        c_cost = raw_claude["cost"]
        
        # ────────────── Comparison ──────────────
        print(f"\n{Fore.MAGENTA}={'='*70}")
        print(f"  📊 RESULTS COMPARISON")
        print(f"{'='*70}{Style.RESET_ALL}")
        print(f"{'Metric':<18} | {'⚡ RouteMoA':<20} | {'🧠 Claude Opus 4.6':<20}")
        print(f"{'-'*18} | {'-'*20} | {'-'*20}")
        print(f"{'Cost':<18} | ${moa_cost:<19.5f} | ${c_cost:<19.5f}")
        print(f"{'Latency':<18} | {moa_time:<17.2f} s  | {c_time:<17.2f} s")
        print(f"{'Layers Used':<18} | {moa_layers:<20} | N/A (Single Call)")
        
        cost_savings = ((c_cost - moa_cost) / max(c_cost, 0.000001)) * 100
        if cost_savings > 0:
            print(f"{'Cost Savings':<18} | {Fore.GREEN}{cost_savings:.1f}% CHEAPER{Style.RESET_ALL}")
        elif cost_savings < 0:
            print(f"{'Cost Compare':<18} | {Fore.RED}{abs(cost_savings):.1f}% EXPENSIVER{Style.RESET_ALL}")
            
        print(f"{'='*70}")
        
        print(f"\n{Fore.GREEN}============= RouteMoA Answer ============={Style.RESET_ALL}\n")
        print(moa_answer[:1000] + "\n... [TRUNCATED] ...\n" if len(moa_answer) > 1000 else moa_answer)
        
        print("\n" + "·"*80 + "\n")
        print(f"{Fore.BLUE}============= Claude Opus 4.6 ============={Style.RESET_ALL}\n")
        print(c_ans[:1000] + "\n... [TRUNCATED] ...\n" if len(c_ans) > 1000 else c_ans)
        
        if i < len(benchmark_questions) - 1:
            print(f"\n{Fore.CYAN}Proceeding to the next question...{Style.RESET_ALL}\n")
            time.sleep(2)

if __name__ == "__main__":
    try:
        asyncio.run(benchmark())
    except KeyboardInterrupt:
        print("\nBenchmark cancelled.")
