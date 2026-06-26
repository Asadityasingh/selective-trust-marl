"""
main.py
Full pipeline for belief_trust_v4.

  Exp A: BeliefTrust + baselines vs AdversarialAgent
  Exp B: BeliefTrust + baselines vs GraduallyCorruptingAgent
"""
import numpy as np
from train import (
    run_belief_adversarial, run_belief_corrupting,
    run_all_baselines,
)
from opponents import AdversarialAgent, GraduallyCorruptingAgent
from plot import generate_all_plots


def _sep(m):
    return float(m["trust_B"][-1] - m["trust_C"][-1])


def print_table(results: dict, title: str):
    header = (f"{'Agent':<16} {'trust_B':>8} {'trust_C':>8} {'sep B-C':>8}"
              f" {'coop_B':>8} {'coop_C':>8} {'rew_B':>8} {'rew_C':>8}"
              f" {'acc_B':>7} {'acc_C':>7}")
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        acc_b = f"{float(m['acc_B'].mean()):.3f}" if len(m["acc_B"]) > 0 else "  n/a"
        acc_c = f"{float(m['acc_C'].mean()):.3f}" if len(m["acc_C"]) > 0 else "  n/a"
        print(f"{name:<16}"
              f" {float(m['trust_B'][-1]):>8.3f}"
              f" {float(m['trust_C'][-1]):>8.3f}"
              f" {_sep(m):>8.3f}"
              f" {float(m['coop_rate_B'].mean()):>8.3f}"
              f" {float(m['coop_rate_C'].mean()):>8.3f}"
              f" {float(m['reward_B'].mean()):>8.2f}"
              f" {float(m['reward_C'].mean()):>8.2f}"
              f" {acc_b:>7}"
              f" {acc_c:>7}")


if __name__ == "__main__":
    # --- Experiment A: vs AdversarialAgent ---
    print("Running BeliefTrustAgent vs AdversarialAgent (20 seeds)...")
    belief_adv = run_belief_adversarial()

    print("Running baselines vs AdversarialAgent...")
    baselines_adv = run_all_baselines(AdversarialAgent, {})

    adv_results = {"BeliefTrust": belief_adv, **baselines_adv}

    # --- Experiment B: vs GraduallyCorruptingAgent ---
    print("Running BeliefTrustAgent vs GraduallyCorruptingAgent (20 seeds)...")
    belief_cor = run_belief_corrupting()

    print("Running baselines vs GraduallyCorruptingAgent...")
    ck = {"corrupt_over": 100, "max_defect_prob": 0.8}
    baselines_cor = run_all_baselines(GraduallyCorruptingAgent, ck)

    corrupt_results = {"BeliefTrust": belief_cor, **baselines_cor}

    # --- Print tables ---
    print_table(adv_results,
                "Adversarial Opponent (200 eval episodes, 20 seeds)")
    print_table(corrupt_results,
                "Gradually Corrupting Opponent (200 eval episodes, 20 seeds)")

    # --- Plots ---
    generate_all_plots(adv_results, corrupt_results)
