"""
plot.py
Three plot groups for selective_trust_v2:
  1. plot_baseline_comparison  — trust separation + coop rate + reward (3 agents)
  2. plot_ablation             — bar chart of key metrics per ablation variant
  3. plot_corrupting_opponent  — trust_C over time + coop_rate_C (slow drift test)
"""
import numpy as np
import matplotlib.pyplot as plt
import os

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")
COLORS = {
    "ImprovedTrust": "#2196F3",
    "TitForTat":     "#4CAF50",
    "Pavlov":        "#FF9800",
}
ABLATION_COLOR = "#5C6BC0"


def _smooth(arr, w=15):
    if len(arr) < w:
        return arr
    return np.convolve(arr, np.ones(w) / w, mode="valid")


def _shade(ax, x, mean, std, color, label, linestyle="-"):
    ax.plot(x, mean, color=color, linewidth=2, label=label, linestyle=linestyle)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)


# ---------------------------------------------------------------------------
# Experiment 1: Baseline comparison
# ---------------------------------------------------------------------------

def plot_baseline_comparison(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- Trust separation over episodes ---
    ax = axes[0]
    for name, m in results.items():
        sep = np.array(m["trust_B"]) - np.array(m["trust_C"])
        sep_std = np.sqrt(np.array(m["trust_B_std"])**2 + np.array(m["trust_C_std"])**2)
        x = np.arange(len(sep))
        _shade(ax, x, sep, sep_std, COLORS[name], name)
    ax.axhline(0, linestyle="--", color="gray", linewidth=0.8)
    ax.set_title("Trust Separation (B − C)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("trust_B − trust_C")
    ax.legend(fontsize=8)

    # --- Cooperation rate vs C ---
    ax = axes[1]
    for name, m in results.items():
        s = _smooth(m["coop_rate_C"])
        s_std = _smooth(m["coop_rate_C_std"])
        x = np.arange(len(s))
        _shade(ax, x, s, s_std, COLORS[name], name)
    ax.set_title("Cooperation Rate vs C (adversarial)")
    ax.set_xlabel("Interaction (smoothed)")
    ax.set_ylabel("Cooperation rate")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)

    # --- Mean reward vs C ---
    ax = axes[2]
    for name, m in results.items():
        s = _smooth(m["reward_C"])
        s_std = _smooth(m["reward_C_std"])
        x = np.arange(len(s))
        _shade(ax, x, s, s_std, COLORS[name], name)
    ax.set_title("Reward vs C (adversarial)")
    ax.set_xlabel("Interaction (smoothed)")
    ax.set_ylabel("Reward per interaction")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 1 — Baseline Comparison (20 seeds)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp1_baseline_comparison.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 2: Ablation
# ---------------------------------------------------------------------------

def plot_ablation(results: dict):
    names = list(results.keys())
    metrics = {
        "Trust sep. (B−C)":   lambda m: m["trust_B"][-1] - m["trust_C"][-1],
        "Coop rate vs B":     lambda m: float(m["coop_rate_B"].mean()),
        "Coop rate vs C":     lambda m: float(m["coop_rate_C"].mean()),
        "Reward vs C":        lambda m: float(m["reward_C"].mean()),
    }

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    x = np.arange(len(names))
    width = 0.6

    for ax, (metric_name, fn) in zip(axes, metrics.items()):
        values = [fn(results[n]) for n in names]
        bars = ax.bar(x, values, width, color=ABLATION_COLOR, alpha=0.85,
                      edgecolor="white")
        # Highlight the full model
        bars[0].set_edgecolor("#E53935")
        bars[0].set_linewidth(2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
        ax.set_title(metric_name, fontsize=9)
        ax.set_ylabel("Value")

    fig.suptitle("Experiment 2 — Ablation Study (20 seeds, red bar = full model)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp2_ablation.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 3: Gradually corrupting opponent
# ---------------------------------------------------------------------------

def plot_corrupting_opponent(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # --- trust_C over time (should drop as corruption increases) ---
    ax = axes[0]
    for name, m in results.items():
        x = np.arange(len(m["trust_C"]))
        _shade(ax, x, np.array(m["trust_C"]), np.array(m["trust_C_std"]),
               COLORS[name], name)
    ax.set_title("Trust toward C (gradually corrupting)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Trust value")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # --- Cooperation rate vs C over time ---
    ax = axes[1]
    for name, m in results.items():
        s = _smooth(m["coop_rate_C"])
        s_std = _smooth(m["coop_rate_C_std"])
        x = np.arange(len(s))
        _shade(ax, x, s, s_std, COLORS[name], name)
    ax.set_title("Cooperation Rate vs C (gradually corrupting)")
    ax.set_xlabel("Interaction (smoothed)")
    ax.set_ylabel("Cooperation rate")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(fontsize=8)

    # --- Reward vs C over time ---
    ax = axes[2]
    for name, m in results.items():
        s = _smooth(m["reward_C"])
        s_std = _smooth(m["reward_C_std"])
        x = np.arange(len(s))
        _shade(ax, x, s, s_std, COLORS[name], name)
    ax.set_title("Reward vs C (gradually corrupting)")
    ax.set_xlabel("Interaction (smoothed)")
    ax.set_ylabel("Reward per interaction")
    ax.legend(fontsize=8)

    fig.suptitle("Experiment 3 — Gradually Corrupting Opponent (20 seeds)",
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "exp3_corrupting_opponent.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary bar chart across all three experiments
# ---------------------------------------------------------------------------

def plot_summary_bar(baseline: dict, corrupting: dict):
    """
    Side-by-side bar chart: mean reward vs C for each agent,
    under both opponent types. Quick visual for the paper.
    """
    agent_names = list(baseline.keys())
    x = np.arange(len(agent_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    b1 = ax.bar(x - width/2,
                [baseline[n]["reward_C"].mean() for n in agent_names],
                width, label="vs Adversarial C",
                color=[COLORS[n] for n in agent_names], alpha=0.9)
    b2 = ax.bar(x + width/2,
                [corrupting[n]["reward_C"].mean() for n in agent_names],
                width, label="vs Gradually Corrupting C",
                color=[COLORS[n] for n in agent_names], alpha=0.5,
                edgecolor="black", linewidth=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(agent_names)
    ax.set_ylabel("Mean reward vs C per interaction")
    ax.set_title("Summary — Reward vs Adversarial C (both opponent types)",
                 fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "summary_reward_vs_C.png"), dpi=150)
    plt.close(fig)


def generate_all_plots(baseline: dict, ablation: dict, corrupting: dict):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_baseline_comparison(baseline)
    plot_ablation(ablation)
    plot_corrupting_opponent(corrupting)
    plot_summary_bar(baseline, corrupting)
    print(f"All plots saved to {PLOTS_DIR}/")
