"""
plot.py
Comparison visualizations: OLD TrustAgent vs NEW ImprovedTrustAgent.
  1. Trust over time (B and C, side-by-side)
  2. Cooperation rate with B vs C
  3. Reward from B vs C interactions
  4. Action heatmaps (before vs after, per opponent)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def _smooth(arr, window=10):
    return np.convolve(arr, np.ones(window) / window, mode="valid")


def plot_trust_comparison(old: dict, new: dict):
    """Two subplots: trust_B and trust_C, each showing OLD vs NEW."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for ax, key, opponent, color in zip(
        axes,
        ["trust_B", "trust_C"],
        ["B (cooperative)", "C (adversarial)"],
        ["steelblue", "tomato"],
    ):
        ax.plot(old[key], color=color, alpha=0.5, linestyle="--", label="Old agent")
        ax.plot(new[key], color=color, linewidth=2, label="Improved agent")
        ax.axhline(0.7, linestyle=":", color="green", linewidth=0.9, label="coop threshold 0.7")
        ax.axhline(0.5, linestyle=":", color="orange", linewidth=0.9, label="defect threshold 0.5")
        ax.set_title(f"Trust toward {opponent}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Trust value")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Trust over Time — Old vs Improved Agent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "trust_comparison.png"), dpi=150)
    plt.close(fig)


def plot_cooperation_comparison(old: dict, new: dict):
    """Cooperation rate with B and C, OLD vs NEW, smoothed."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    w = 10

    for ax, key, title, color in zip(
        axes,
        ["coop_rate_B", "coop_rate_C"],
        ["vs B (cooperative)", "vs C (adversarial)"],
        ["steelblue", "tomato"],
    ):
        ax.plot(_smooth(old[key], w), color=color, alpha=0.5, linestyle="--", label="Old agent")
        ax.plot(_smooth(new[key], w), color=color, linewidth=2, label="Improved agent")
        ax.set_title(f"Cooperation rate {title}")
        ax.set_xlabel("Interaction (smoothed)")
        ax.set_ylabel("Cooperation rate")
        ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=8)

    fig.suptitle("Cooperation Rate — Old vs Improved Agent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "cooperation_comparison.png"), dpi=150)
    plt.close(fig)


def plot_reward_comparison(old: dict, new: dict):
    """Reward from B and C interactions, OLD vs NEW, smoothed."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    w = 10

    for ax, key, title, color in zip(
        axes,
        ["reward_B", "reward_C"],
        ["vs B (cooperative)", "vs C (adversarial)"],
        ["steelblue", "tomato"],
    ):
        ax.plot(_smooth(old[key], w), color=color, alpha=0.5, linestyle="--", label="Old agent")
        ax.plot(_smooth(new[key], w), color=color, linewidth=2, label="Improved agent")
        ax.set_title(f"Reward {title}")
        ax.set_xlabel("Interaction (smoothed)")
        ax.set_ylabel("Reward per interaction")
        ax.legend(fontsize=8)

    fig.suptitle("Reward per Interaction — Old vs Improved Agent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "reward_comparison.png"), dpi=150)
    plt.close(fig)


def plot_heatmap_comparison(old: dict, new: dict):
    """
    2x2 grid of action heatmaps:
      rows = OLD / NEW
      cols = vs B / vs C
    """
    rounds = old["rounds_per_interaction"]

    def reshape(actions):
        arr = np.array(actions)
        n = len(arr) // rounds
        return arr[: n * rounds].reshape(n, rounds)

    datasets = [
        (old["actions_vs_B"], "Old — vs B (cooperative)"),
        (old["actions_vs_C"], "Old — vs C (adversarial)"),
        (new["actions_vs_B"], "Improved — vs B (cooperative)"),
        (new["actions_vs_C"], "Improved — vs C (adversarial)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (actions, title) in zip(axes.flat, datasets):
        mat = reshape(actions)
        im = ax.imshow(mat.T, aspect="auto", cmap="RdYlGn_r",
                       vmin=0, vmax=1, origin="lower")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Interaction index")
        ax.set_ylabel("Round within interaction")
        fig.colorbar(im, ax=ax, label="0=Coop  1=Defect")

    fig.suptitle("Action Heatmaps — Old vs Improved Agent", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "heatmap_comparison.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(old: dict, new: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_trust_comparison(old, new)
    plot_cooperation_comparison(old, new)
    plot_reward_comparison(old, new)
    plot_heatmap_comparison(old, new)
    print(f"Plots saved to {PLOTS_DIR}/")
