"""
train.py
Multi-seed training loop. Collects per-episode metrics for Agent A:
  - trust_B, trust_C over time
  - cooperation rate with B and C
  - reward from B and C interactions
  - action sequences (for heatmap)
"""
import numpy as np
from environment import PDEnvironment
from agents import TrustAgent, ImprovedTrustAgent, CooperativeAgent, AdversarialAgent


def _collect_metrics(agent_a, env, rng, n_episodes) -> dict:
    """Run `n_episodes` and collect all metrics for agent_a (id=0)."""
    metrics = {
        "trust_B": [], "trust_C": [],
        "coop_rate_B": [], "coop_rate_C": [],
        "reward_B": [], "reward_C": [],
        "actions_vs_B": [], "actions_vs_C": [],
    }
    for _ in range(n_episodes):
        result = env.run_episode(rng)
        metrics["trust_B"].append(agent_a.get_trust(1))
        metrics["trust_C"].append(agent_a.get_trust(2))
        for (i, j, acts_i, _acts_j, rew_i, _rew_j) in result["logs"]:
            if i == 0 and j == 1:
                metrics["coop_rate_B"].append(1 - np.mean(acts_i))
                metrics["reward_B"].append(rew_i)
                metrics["actions_vs_B"].extend(acts_i)
            elif i == 0 and j == 2:
                metrics["coop_rate_C"].append(1 - np.mean(acts_i))
                metrics["reward_C"].append(rew_i)
                metrics["actions_vs_C"].extend(acts_i)
    return metrics


def _average_across_seeds(all_metrics: list, n_episodes: int,
                          rounds_per_interaction: int) -> dict:
    """Average per-seed metric lists; use seed-0 action sequences for heatmaps."""
    averaged = {}
    for key in ["trust_B", "trust_C"]:
        averaged[key] = np.mean([m[key] for m in all_metrics], axis=0)
    for key in ["coop_rate_B", "coop_rate_C", "reward_B", "reward_C"]:
        arrays = [np.array(m[key]) for m in all_metrics]
        min_len = min(len(a) for a in arrays)
        averaged[key] = np.mean([a[:min_len] for a in arrays], axis=0)
    averaged["actions_vs_B"] = all_metrics[0]["actions_vs_B"]
    averaged["actions_vs_C"] = all_metrics[0]["actions_vs_C"]
    averaged["n_episodes"] = n_episodes
    averaged["rounds_per_interaction"] = rounds_per_interaction
    return averaged


def run_experiment(n_seeds: int = 5, n_episodes: int = 200,
                   rounds_per_interaction: int = 10) -> dict:
    """Original TrustAgent experiment (kept for backward compatibility)."""
    all_metrics = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        agent_a = TrustAgent(0, n_agents=3)
        agents = [agent_a, CooperativeAgent(1, n_agents=3), AdversarialAgent(2, n_agents=3)]
        env = PDEnvironment(agents, rounds_per_interaction=rounds_per_interaction)
        all_metrics.append(_collect_metrics(agent_a, env, rng, n_episodes))
    return _average_across_seeds(all_metrics, n_episodes, rounds_per_interaction)


def run_experiment_both(n_seeds: int = 5, n_episodes: int = 200,
                        rounds_per_interaction: int = 10) -> tuple[dict, dict]:
    """
    Run OLD (TrustAgent) and NEW (ImprovedTrustAgent) under identical seeds.
    Returns (old_metrics, new_metrics).
    """
    old_all, new_all = [], []
    for s in range(n_seeds):
        for agent_cls, store in [(TrustAgent, old_all), (ImprovedTrustAgent, new_all)]:
            rng = np.random.default_rng(s)   # same seed → fair comparison
            agent_a = agent_cls(0, n_agents=3)
            agents = [agent_a,
                      CooperativeAgent(1, n_agents=3),
                      AdversarialAgent(2, n_agents=3)]
            env = PDEnvironment(agents, rounds_per_interaction=rounds_per_interaction)
            store.append(_collect_metrics(agent_a, env, rng, n_episodes))

    old = _average_across_seeds(old_all, n_episodes, rounds_per_interaction)
    new = _average_across_seeds(new_all, n_episodes, rounds_per_interaction)
    return old, new
