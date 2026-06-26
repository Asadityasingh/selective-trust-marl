"""
main.py
Full pipeline for learned_trust_v3:
  1. Train LearnedTrustAgent (REINFORCE) against adversarial + corrupting opponents
  2. Evaluate all rule-based baselines under same protocol
  3. Print summary comparison table
  4. Generate all plots
"""
import numpy as np
from train_rl import run_learned_adversarial, run_learned_corrupting
from compare import run_all_baselines_adversarial, run_all_baselines_corrupting
from plot import generate_all_plots


def _sep(m):
    return float(m["trust_B"][-1] - m["trust_C"][-1])


def print_table(results: dict, title: str):
    header = (f"{'Agent':<16} {'trust_B':>8} {'trust_C':>8} {'sep B-C':>8}"
              f" {'coop_B':>8} {'coop_C':>8} {'rew_B':>8} {'rew_C':>8}")
    print(f"\n=== {title} ===")
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<16}"
              f" {float(m['trust_B'][-1]):>8.3f}"
              f" {float(m['trust_C'][-1]):>8.3f}"
              f" {_sep(m):>8.3f}"
              f" {float(m['coop_rate_B'].mean()):>8.3f}"
              f" {float(m['coop_rate_C'].mean()):>8.3f}"
              f" {float(m['reward_B'].mean()):>8.2f}"
              f" {float(m['reward_C'].mean()):>8.2f}")


if __name__ == "__main__":
    # --- Experiment A: vs AdversarialAgent ---
    print("Training LearnedTrustAgent vs AdversarialAgent (20 seeds × 300 episodes)...")
    learned_adv, adv_losses = run_learned_adversarial()

    print("Evaluating rule-based baselines vs AdversarialAgent...")
    baselines_adv = run_all_baselines_adversarial()
    baselines_adv["LearnedTrust"] = learned_adv   # add learned agent to comparison

    # Reorder so LearnedTrust is first
    adv_results = {"LearnedTrust": learned_adv, **{k: v for k, v in baselines_adv.items()
                                                    if k != "LearnedTrust"}}

    # --- Experiment B: vs GraduallyCorruptingAgent ---
    print("Training LearnedTrustAgent vs GraduallyCorruptingAgent (20 seeds × 300 episodes)...")
    learned_cor, corrupt_losses = run_learned_corrupting()

    print("Evaluating rule-based baselines vs GraduallyCorruptingAgent...")
    baselines_cor = run_all_baselines_corrupting()
    corrupt_results = {"LearnedTrust": learned_cor, **{k: v for k, v in baselines_cor.items()
                                                        if k != "LearnedTrust"}}

    # --- Print tables ---
    print_table(adv_results,     "Adversarial Opponent — Eval (200 episodes, 20 seeds)")
    print_table(corrupt_results, "Gradually Corrupting Opponent — Eval (200 episodes, 20 seeds)")

    # --- Plots ---
    generate_all_plots(adv_results, corrupt_results, adv_losses, corrupt_losses)
