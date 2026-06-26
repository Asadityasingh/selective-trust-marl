"""
Main entry point: Repeated Prisoner's Dilemma — Non-Adversarial Experiment.

Compares:
  Case 1: Baseline Q-learning vs Baseline Q-learning
  Case 2: Trust-based vs Trust-based

Both agents learn simultaneously. Multiple seeds for stability.

Usage:
    source venv/bin/activate
    python main.py
"""

import os
from train import run_multi_seed
from plot import plot_rewards, plot_cooperation, plot_trust, plot_action_heatmap


def main():
    os.makedirs("plots", exist_ok=True)

    num_episodes = 1000
    num_rounds = 100
    seeds = [42, 123, 456, 789, 1024]

    print("=" * 55)
    print(" Repeated Prisoner's Dilemma — Non-Adversarial Setting")
    print("=" * 55)
    print(f" Episodes: {num_episodes} | Rounds/episode: {num_rounds} | Seeds: {len(seeds)}")

    # --- Case 1: Baseline vs Baseline ---
    print("\n[1/2] Running Baseline (Q-learning) vs Baseline (Q-learning)...")
    baseline_metrics = run_multi_seed("baseline", num_episodes, num_rounds, seeds)
    bl_final = baseline_metrics[-1]
    print(f"  Final → Avg Reward: {bl_final['avg_reward']:.2f}, Coop Rate: {bl_final['coop_rate']:.2f}")

    # --- Case 2: Trust vs Trust ---
    print("\n[2/2] Running Trust-based vs Trust-based...")
    trust_metrics = run_multi_seed("trust", num_episodes, num_rounds, seeds)
    tr_final = trust_metrics[-1]
    print(f"  Final → Avg Reward: {tr_final['avg_reward']:.2f}, Coop Rate: {tr_final['coop_rate']:.2f}")

    # --- Generate Plots ---
    print("\nGenerating plots...")
    plot_rewards(baseline_metrics, trust_metrics)
    plot_cooperation(baseline_metrics, trust_metrics)
    plot_trust(trust_metrics)
    plot_action_heatmap(baseline_metrics, label="Baseline", save_path="plots/actions_baseline.png")
    plot_action_heatmap(trust_metrics, label="Trust-Based", save_path="plots/actions_trust.png")

    print("  Saved to ./plots/")
    print("\n✅ Experiment complete!")


if __name__ == "__main__":
    main()
