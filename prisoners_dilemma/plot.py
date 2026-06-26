"""
Plotting utilities for experiment comparison.

Generates:
1. Reward vs Episodes (Baseline vs Trust)
2. Cooperation Rate vs Episodes
3. Trust vs Time (both agents)
4. Action heatmap (C/D over time for last episode)
"""

import numpy as np
import matplotlib.pyplot as plt


def smooth(data, window=50):
    """Moving average smoothing."""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_rewards(baseline, trust, save_path="plots/rewards.png"):
    """Reward comparison: Baseline vs Trust-based."""
    bl = smooth([m["avg_reward"] for m in baseline])
    tr = smooth([m["avg_reward"] for m in trust])

    plt.figure(figsize=(10, 5))
    plt.plot(bl, label="Baseline (Q-learning)", alpha=0.8)
    plt.plot(tr, label="Trust-based", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward (both agents)")
    plt.title("Reward vs Episodes — Baseline vs Trust-Based")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_cooperation(baseline, trust, save_path="plots/cooperation.png"):
    """Cooperation rate comparison."""
    bl = smooth([m["coop_rate"] for m in baseline])
    tr = smooth([m["coop_rate"] for m in trust])

    plt.figure(figsize=(10, 5))
    plt.plot(bl, label="Baseline (Q-learning)", alpha=0.8)
    plt.plot(tr, label="Trust-based", alpha=0.8)
    plt.xlabel("Episode")
    plt.ylabel("Cooperation Rate")
    plt.title("Cooperation Rate vs Episodes")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_trust(trust_metrics, save_path="plots/trust_over_time.png"):
    """Trust values over rounds for both agents (last episode)."""
    last = trust_metrics[-1]
    trust_a = last.get("trust_a", [])
    trust_b = last.get("trust_b", [])

    if not trust_a:
        return

    plt.figure(figsize=(10, 4))
    plt.plot(trust_a, label="Agent A Trust", alpha=0.8)
    plt.plot(trust_b, label="Agent B Trust", alpha=0.8)
    plt.axhline(y=0.6, color="red", linestyle="--", alpha=0.5, label="Threshold (0.6)")
    plt.xlabel("Round")
    plt.ylabel("Trust Value")
    plt.title("Trust Over Time (Last Episode) — Both Agents")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_action_heatmap(metrics, label="", save_path="plots/actions.png"):
    """Action heatmap for both agents in the last episode."""
    last = metrics[-1]
    actions_a = last.get("actions_a", [])
    actions_b = last.get("actions_b", [])

    if not actions_a:
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 3), sharex=True)

    for ax, actions, name in zip(axes, [actions_a, actions_b], ["Agent A", "Agent B"]):
        colors = ["green" if a == 0 else "red" for a in actions]
        ax.bar(range(len(actions)), [1] * len(actions), color=colors, width=1.0)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=10)

    axes[1].set_xlabel("Round")
    fig.suptitle(f"{label} — Actions (Green=Cooperate, Red=Defect) — Last Episode", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
