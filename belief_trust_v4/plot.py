"""
plot.py
Visualisations for belief_trust_v4.

Plots:
  1. plot_trust_dynamics      — trust_B and trust_C over episodes (all agents)
  2. plot_coop_and_reward     — 2x2: coop rate + reward vs B and C
  3. plot_prediction_accuracy — BeliefTrust prediction accuracy vs B and C
  4. plot_summary_bar         — trust separation + reward vs C (paper figure)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

COLORS = {
    "BeliefTrust":   "#9C27B0",   # purple — our new agent
    "ImprovedTrust": "#2196F3",   # blue
    "TitForTat":     "#4CAF50",   # green
    "Pavlov":        "#FF9800",   # orange
}


def _smooth(arr, w=15):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _shade(ax, x, mean, std, color, label, ls="-"):
    ax.plot(x, mean, color=color, lw=2, label=label, linestyle=ls)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# 1. Trust dynamics
# ---------------------------------------------------------------------------

def plot_trust_dynamics(results: dict, tag: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
    for ax, key, title in zip(
        axes,
        ["trust_B", "trust_C"],
        ["Trust toward B (cooperative)", "Trust toward C (adversarial)"],
    ):
        for name, m in results.items():
            x = np.arange(len(m[key]))
            _shade(ax, x, np.array(m[key]), np.array(m[f"{key}_std"]),
                   COLORS[name], name)
        ax.axhline(0.7, ls=":", color="gray", lw=0.8, label="coop thresh 0.7")
        ax.axhline(0.5, ls=":", color="silver", lw=0.8, label="defect thresh 0.5")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Trust value")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7)

    fig.suptitle(f"Trust Dynamics — {tag} opponent (20 seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"trust_dynamics_{tag}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Cooperation rate and reward
# ---------------------------------------------------------------------------

def plot_coop_and_reward(results: dict, tag: str):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    specs = [
        ("coop_rate_B", "Coop rate vs B (cooperative)"),
        ("coop_rate_C", "Coop rate vs C (adversarial)"),
        ("reward_B",    "Reward vs B (cooperative)"),
        ("reward_C",    "Reward vs C (adversarial)"),
    ]
    for ax, (key, title) in zip(axes.flat, specs):
        for name, m in results.items():
            s     = _smooth(m[key])
            s_std = _smooth(m[f"{key}_std"])
            _shade(ax, np.arange(len(s)), s, s_std, COLORS[name], name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Interaction (smoothed)")
        ax.set_ylabel("Value")
        if "coop" in key:
            ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)

    fig.suptitle(f"Cooperation & Reward — {tag} opponent (20 seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"coop_reward_{tag}.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Prediction accuracy (BeliefTrust only)
# ---------------------------------------------------------------------------

def plot_prediction_accuracy(belief_adv: dict, belief_cor: dict):
    """
    Shows how quickly BeliefNet learns to predict B vs C behavior.
    This is the key interpretability plot for the paper.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for ax, m, tag in zip(axes,
                           [belief_adv, belief_cor],
                           ["Adversarial opponent", "Gradually corrupting opponent"]):
        if len(m["acc_B"]) == 0:
            ax.text(0.5, 0.5, "No accuracy data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        x_B = np.arange(len(m["acc_B"]))
        x_C = np.arange(len(m["acc_C"]))
        _shade(ax, x_B, np.array(m["acc_B"]), np.array(m["acc_B_std"]),
               COLORS["ImprovedTrust"], "Accuracy vs B (cooperative)")
        _shade(ax, x_C, np.array(m["acc_C"]), np.array(m["acc_C_std"]),
               "#E53935", "Accuracy vs C (adversarial)")
        ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="random baseline")
        ax.set_title(f"BeliefNet Prediction Accuracy\n{tag}")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.3, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("BeliefNet — Opponent Behavior Prediction Accuracy (20 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "prediction_accuracy.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Summary bar chart (paper figure)
# ---------------------------------------------------------------------------

def plot_summary_bar(adv_results: dict, corrupt_results: dict):
    agent_names = list(adv_results.keys())
    x = np.arange(len(agent_names))
    w = 0.35
    colors = [COLORS[n] for n in agent_names]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Trust separation
    ax = axes[0]
    sep_adv = [adv_results[n]["trust_B"][-1] - adv_results[n]["trust_C"][-1]
               for n in agent_names]
    sep_cor = [corrupt_results[n]["trust_B"][-1] - corrupt_results[n]["trust_C"][-1]
               for n in agent_names]
    ax.bar(x - w/2, sep_adv, w, color=colors, alpha=0.9, label="vs Adversarial C")
    ax.bar(x + w/2, sep_cor, w, color=colors, alpha=0.5,
           edgecolor="black", lw=0.8, label="vs Corrupting C")
    ax.axhline(0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(agent_names, rotation=15)
    ax.set_ylabel("trust_B − trust_C")
    ax.set_title("Trust Separation (B − C)")
    ax.legend(fontsize=8)

    # Cooperation rate vs C
    ax = axes[1]
    cr_adv = [float(adv_results[n]["coop_rate_C"].mean()) for n in agent_names]
    cr_cor = [float(corrupt_results[n]["coop_rate_C"].mean()) for n in agent_names]
    ax.bar(x - w/2, cr_adv, w, color=colors, alpha=0.9, label="vs Adversarial C")
    ax.bar(x + w/2, cr_cor, w, color=colors, alpha=0.5,
           edgecolor="black", lw=0.8, label="vs Corrupting C")
    ax.set_xticks(x); ax.set_xticklabels(agent_names, rotation=15)
    ax.set_ylabel("Mean cooperation rate")
    ax.set_title("Cooperation Rate vs C")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)

    # Reward vs C
    ax = axes[2]
    rw_adv = [float(adv_results[n]["reward_C"].mean()) for n in agent_names]
    rw_cor = [float(corrupt_results[n]["reward_C"].mean()) for n in agent_names]
    ax.bar(x - w/2, rw_adv, w, color=colors, alpha=0.9, label="vs Adversarial C")
    ax.bar(x + w/2, rw_cor, w, color=colors, alpha=0.5,
           edgecolor="black", lw=0.8, label="vs Corrupting C")
    ax.set_xticks(x); ax.set_xticklabels(agent_names, rotation=15)
    ax.set_ylabel("Mean reward per interaction")
    ax.set_title("Reward vs C")
    ax.legend(fontsize=8)

    fig.suptitle("Summary — BeliefTrust vs Rule-Based Baselines (20 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "summary_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def generate_all_plots(adv_results: dict, corrupt_results: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_trust_dynamics(adv_results,     tag="adversarial")
    plot_trust_dynamics(corrupt_results, tag="corrupting")
    plot_coop_and_reward(adv_results,     tag="adversarial")
    plot_coop_and_reward(corrupt_results, tag="corrupting")
    # Prediction accuracy uses BeliefTrust metrics only
    plot_prediction_accuracy(adv_results["BeliefTrust"],
                             corrupt_results["BeliefTrust"])
    plot_summary_bar(adv_results, corrupt_results)
    print(f"All plots saved to {PLOTS_DIR}/")
