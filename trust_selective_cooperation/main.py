"""
main.py
Entry point — runs OLD vs NEW trust agent comparison experiment.
"""
from train import run_experiment_both
from plot import generate_all_plots


def _print_comparison(old: dict, new: dict):
    header = f"{'Metric':<30} {'Old':>10} {'New':>10} {'Delta':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    rows = [
        ("Final trust_B",        old["trust_B"][-1],          new["trust_B"][-1]),
        ("Final trust_C",        old["trust_C"][-1],          new["trust_C"][-1]),
        ("Trust separation B-C", old["trust_B"][-1] - old["trust_C"][-1],
                                  new["trust_B"][-1] - new["trust_C"][-1]),
        ("Mean coop rate vs B",  old["coop_rate_B"].mean(),   new["coop_rate_B"].mean()),
        ("Mean coop rate vs C",  old["coop_rate_C"].mean(),   new["coop_rate_C"].mean()),
        ("Mean reward vs B",     old["reward_B"].mean(),      new["reward_B"].mean()),
        ("Mean reward vs C",     old["reward_C"].mean(),      new["reward_C"].mean()),
    ]

    for label, o, n in rows:
        delta = n - o
        sign = "+" if delta >= 0 else ""
        print(f"{label:<30} {o:>10.3f} {n:>10.3f} {sign}{delta:>9.3f}")

    print("=" * len(header) + "\n")


if __name__ == "__main__":
    print("Running OLD vs IMPROVED trust agent comparison (5 seeds, 200 episodes)...")
    old, new = run_experiment_both(n_seeds=5, n_episodes=200, rounds_per_interaction=10)
    _print_comparison(old, new)
    generate_all_plots(old, new)
