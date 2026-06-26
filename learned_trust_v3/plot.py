"""
plot.py
Visualisations for learned_trust_v3.
  1. plot_training_curve      — REINFORCE loss over training episodes
  2. plot_trust_separation    — trust_B and trust_C over eval episodes (all agents)
  3. plot_coop_and_reward     — cooperation rate + reward vs C (all agents)
  4. plot_corrupting          — same metrics against gradually corrupting opponent
  5. plot_summary_bar         — final summary bar chart for paper
"""
import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")

COLORS = {
    "LearnedTrust":  "#E91E63",   # pink  — our new agent
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


def plot_training_curve(loss_curves: np.ndarray, tag: str = "adversarial"):
    """loss_curves: (n_seeds, n_train_episodes)"""
    mean = loss_curves.mean(axis=0)
    std  = loss_curves.std(axis=0)

    sm_mean = _smooth(mean)
    sm_std  = _smooth(std)
    x = np.arange(len(sm_mean))   # use smoothed length, not original

    fig, ax = plt.subplots(figsize=(8, 4))
    _shade(ax, x, sm_mean, sm_std, COLORS["LearnedTrust"],
           "REINFORCE loss (mean ± std)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Loss")
    ax.set_title(f"LearnedTrustAgent — Training Curve ({tag} opponent)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"training_curve_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_trust_separation(all_results: dict, tag: str = "adversarial"):
    """
    Two subplots: trust_B and trust_C over eval episodes for all agents.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

    for ax, key, title in zip(axes,
                               ["trust_B", "trust_C"],
                               ["Trust toward B (cooperative)",
                                "Trust toward C (adversarial)"]):
        for name, m in all_results.items():
            x = np.arange(len(m[key]))
            _shade(ax, x, np.array(m[key]), np.array(m[f"{key}_std"]),
                   COLORS[name], name)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Trust value")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle(f"Trust Dynamics — All Agents ({tag} opponent, 20 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"trust_dynamics_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_coop_and_reward(all_results: dict, tag: str = "adversarial"):
    """
    Four subplots: coop_rate_B, coop_rate_C, reward_B, reward_C.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    keys = ["coop_rate_B", "coop_rate_C", "reward_B", "reward_C"]
    titles = ["Coop rate vs B (cooperative)", "Coop rate vs C (adversarial)",
              "Reward vs B (cooperative)",    "Reward vs C (adversarial)"]

    for ax, key, title in zip(axes.flat, keys, titles):
        for name, m in all_results.items():
            s = _smooth(m[key])
            s_std = _smooth(m[f"{key}_std"])
            x = np.arange(len(s))
            _shade(ax, x, s, s_std, COLORS[name], name)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Interaction (smoothed)")
        ax.set_ylabel("Value")
        if "coop" in key:
            ax.set_ylim(-0.05, 1.1)
        ax.legend(fontsize=7)

    fig.suptitle(f"Cooperation & Reward — All Agents ({tag} opponent, 20 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, f"coop_reward_{tag}.png"), dpi=150)
    plt.close(fig)


def plot_summary_bar(adv_results: dict, corrupt_results: dict):
    """
    2x2 bar chart: (opponent type) x (metric: trust sep, reward vs C).
    The key paper figure.
    """
    agent_names = list(adv_results.keys())
    x = np.arange(len(agent_names))
    w = 0.35
    colors = [COLORS[n] for n in agent_names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Trust separation ---
    ax = axes[0]
    sep_adv  = [adv_results[n]["trust_B"][-1] - adv_results[n]["trust_C"][-1]
                for n in agent_names]
    sep_cor  = [corrupt_results[n]["trust_B"][-1] - corrupt_results[n]["trust_C"][-1]
                for n in agent_names]
    ax.bar(x - w/2, sep_adv,  w, color=colors, alpha=0.9, label="vs Adversarial C")
    ax.bar(x + w/2, sep_cor,  w, color=colors, alpha=0.5,
           edgecolor="black", lw=0.8, label="vs Corrupting C")
    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.set_xticks(x); ax.set_xticklabels(agent_names)
    ax.set_ylabel("trust_B − trust_C")
    ax.set_title("Trust Separation (B − C)")
    ax.legend(fontsize=8)

    # --- Reward vs C ---
    ax = axes[1]
    rew_adv  = [adv_results[n]["reward_C"].mean() for n in agent_names]
    rew_cor  = [corrupt_results[n]["reward_C"].mean() for n in agent_names]
    ax.bar(x - w/2, rew_adv,  w, color=colors, alpha=0.9, label="vs Adversarial C")
    ax.bar(x + w/2, rew_cor,  w, color=colors, alpha=0.5,
           edgecolor="black", lw=0.8, label="vs Corrupting C")
    ax.set_xticks(x); ax.set_xticklabels(agent_names)
    ax.set_ylabel("Mean reward per interaction")
    ax.set_title("Reward vs C")
    ax.legend(fontsize=8)

    fig.suptitle("Summary — LearnedTrust vs Rule-Based Baselines (20 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "summary_comparison.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(adv_results: dict, corrupt_results: dict,
                       adv_losses: np.ndarray, corrupt_losses: np.ndarray):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_training_curve(adv_losses,     tag="adversarial")
    plot_training_curve(corrupt_losses, tag="corrupting")
    plot_trust_separation(adv_results,     tag="adversarial")
    plot_trust_separation(corrupt_results, tag="corrupting")
    plot_coop_and_reward(adv_results,     tag="adversarial")
    plot_coop_and_reward(corrupt_results, tag="corrupting")
    plot_summary_bar(adv_results, corrupt_results)
    print(f"All plots saved to {PLOTS_DIR}/")
