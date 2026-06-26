"""
main.py
Runs all three experiments and prints a unified summary table.

  Exp 1: Baseline comparison  — ImprovedTrust vs TitForTat vs Pavlov
  Exp 2: Ablation study       — which mechanisms actually matter?
  Exp 3: Corrupting opponent  — slow adversarial drift detection
"""
import numpy as np
from train import run_baseline_comparison, run_ablation, run_corrupting_opponent
from plot import generate_all_plots


def _sep(m):
    return m["trust_B"][-1] - m["trust_C"][-1]


def print_baseline_table(results: dict):
    header = f"{'Agent':<16} {'trust_B':>8} {'trust_C':>8} {'sep B-C':>8} " \
             f"{'coop_B':>8} {'coop_C':>8} {'rew_B':>8} {'rew_C':>8}"
    print("\n=== Experiment 1: Baseline Comparison ===")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}"
              f" {m['trust_B'][-1]:>8.3f}"
              f" {m['trust_C'][-1]:>8.3f}"
              f" {_sep(m):>8.3f}"
              f" {m['coop_rate_B'].mean():>8.3f}"
              f" {m['coop_rate_C'].mean():>8.3f}"
              f" {m['reward_B'].mean():>8.2f}"
              f" {m['reward_C'].mean():>8.2f}")


def print_ablation_table(results: dict):
    header = f"{'Variant':<22} {'sep B-C':>8} {'coop_B':>8} {'coop_C':>8} {'rew_C':>8}"
    print("\n=== Experiment 2: Ablation Study ===")
    print(header)
    print("-" * len(header))
    full_sep = _sep(results["Full (all on)"])
    for name, m in results.items():
        sep = _sep(m)
        delta = sep - full_sep
        sign = "+" if delta >= 0 else ""
        print(f"{name:<22}"
              f" {sep:>8.3f}"
              f" {m['coop_rate_B'].mean():>8.3f}"
              f" {m['coop_rate_C'].mean():>8.3f}"
              f" {m['reward_C'].mean():>8.2f}"
              f"   (Δsep {sign}{delta:.3f})")


def print_corrupting_table(results: dict):
    header = f"{'Agent':<16} {'trust_C_final':>14} {'coop_C_mean':>12} {'rew_C_mean':>11}"
    print("\n=== Experiment 3: Gradually Corrupting Opponent ===")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}"
              f" {m['trust_C'][-1]:>14.3f}"
              f" {m['coop_rate_C'].mean():>12.3f}"
              f" {m['reward_C'].mean():>11.2f}")


if __name__ == "__main__":
    print("Running Experiment 1: Baseline comparison (20 seeds)...")
    baseline = run_baseline_comparison()

    print("Running Experiment 2: Ablation study (20 seeds)...")
    ablation = run_ablation()

    print("Running Experiment 3: Gradually corrupting opponent (20 seeds)...")
    corrupting = run_corrupting_opponent()

    print_baseline_table(baseline)
    print_ablation_table(ablation)
    print_corrupting_table(corrupting)

    generate_all_plots(baseline, ablation, corrupting)
