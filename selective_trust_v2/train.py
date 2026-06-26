"""
train.py
Unified training loop for all three experiments:
  1. run_baseline_comparison  — ImprovedTrust vs TitForTat vs Pavlov
  2. run_ablation             — 6 ablation variants of ImprovedTrustAgent
  3. run_corrupting_opponent  — ImprovedTrust vs TitForTat vs Pavlov
                                against GraduallyCorruptingAgent
"""
import numpy as np
from environment import PDEnvironment
from agents import (
    ImprovedTrustAgent, AblationTrustAgent,
    TitForTatAgent, PavlovAgent,
    CooperativeAgent, AdversarialAgent, GraduallyCorruptingAgent,
)

N_SEEDS = 20
N_EPISODES = 200
ROUNDS = 10


# ---------------------------------------------------------------------------
# Core metric collector
# ---------------------------------------------------------------------------

def _collect(agent_a, env, rng, n_episodes, corrupting_agent=None) -> dict:
    """
    Run n_episodes and collect metrics for agent_a (always id=0).
    If corrupting_agent is provided, call notify_episode() each episode.
    """
    m = {
        "trust_B": [], "trust_C": [],
        "coop_rate_B": [], "coop_rate_C": [],
        "reward_B": [], "reward_C": [],
        "actions_vs_B": [], "actions_vs_C": [],
    }
    for _ in range(n_episodes):
        if corrupting_agent is not None:
            corrupting_agent.notify_episode()
        result = env.run_episode(rng)
        m["trust_B"].append(agent_a.get_trust(1))
        m["trust_C"].append(agent_a.get_trust(2))
        for (i, j, acts_i, _aj, rew_i, _rj) in result["logs"]:
            if i == 0 and j == 1:
                m["coop_rate_B"].append(1 - np.mean(acts_i))
                m["reward_B"].append(rew_i)
                m["actions_vs_B"].extend(acts_i)
            elif i == 0 and j == 2:
                m["coop_rate_C"].append(1 - np.mean(acts_i))
                m["reward_C"].append(rew_i)
                m["actions_vs_C"].extend(acts_i)
    return m


def _average(all_metrics: list, n_episodes: int, rounds: int) -> dict:
    avg = {}
    for key in ["trust_B", "trust_C"]:
        avg[key] = np.mean([m[key] for m in all_metrics], axis=0)
        avg[f"{key}_std"] = np.std([m[key] for m in all_metrics], axis=0)
    for key in ["coop_rate_B", "coop_rate_C", "reward_B", "reward_C"]:
        arrays = [np.array(m[key]) for m in all_metrics]
        min_len = min(len(a) for a in arrays)
        stacked = np.array([a[:min_len] for a in arrays])
        avg[key] = stacked.mean(axis=0)
        avg[f"{key}_std"] = stacked.std(axis=0)
    avg["actions_vs_B"] = all_metrics[0]["actions_vs_B"]
    avg["actions_vs_C"] = all_metrics[0]["actions_vs_C"]
    avg["n_episodes"] = n_episodes
    avg["rounds_per_interaction"] = rounds
    return avg


def _run_agent(agent_cls, agent_kwargs, opponent_c_cls, opponent_c_kwargs,
               n_seeds, n_episodes, rounds) -> dict:
    """Generic multi-seed runner for any agent class + opponent C class."""
    all_metrics = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        agent_a = agent_cls(0, n_agents=3, **agent_kwargs)
        opp_c = opponent_c_cls(2, n_agents=3, **opponent_c_kwargs)
        agents = [agent_a, CooperativeAgent(1, n_agents=3), opp_c]
        env = PDEnvironment(agents, rounds_per_interaction=rounds)
        corrupting = opp_c if isinstance(opp_c, GraduallyCorruptingAgent) else None
        all_metrics.append(_collect(agent_a, env, rng, n_episodes, corrupting))
    return _average(all_metrics, n_episodes, rounds)


# ---------------------------------------------------------------------------
# Experiment 1: Baseline comparison
# ---------------------------------------------------------------------------

def run_baseline_comparison(n_seeds=N_SEEDS, n_episodes=N_EPISODES,
                             rounds=ROUNDS) -> dict:
    """
    Compare ImprovedTrustAgent, TitForTat, Pavlov
    against the standard mixed environment (CooperativeAgent + AdversarialAgent).
    Returns dict keyed by agent name.
    """
    agents = {
        "ImprovedTrust": (ImprovedTrustAgent, {}),
        "TitForTat":     (TitForTatAgent,     {}),
        "Pavlov":        (PavlovAgent,         {}),
    }
    results = {}
    for name, (cls, kwargs) in agents.items():
        results[name] = _run_agent(
            cls, kwargs, AdversarialAgent, {},
            n_seeds, n_episodes, rounds
        )
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Ablation study
# ---------------------------------------------------------------------------

# Each variant removes exactly one mechanism from ImprovedTrustAgent
ABLATION_VARIANTS = {
    "Full (all on)":        dict(use_asymmetric=True,  use_betrayal=True,
                                 use_hysteresis=True,  use_smoothing=True,
                                 use_belief=True),
    "No asymmetric":        dict(use_asymmetric=False, use_betrayal=True,
                                 use_hysteresis=True,  use_smoothing=True,
                                 use_belief=True),
    "No betrayal penalty":  dict(use_asymmetric=True,  use_betrayal=False,
                                 use_hysteresis=True,  use_smoothing=True,
                                 use_belief=True),
    "No hysteresis":        dict(use_asymmetric=True,  use_betrayal=True,
                                 use_hysteresis=False, use_smoothing=True,
                                 use_belief=True),
    "No smoothing":         dict(use_asymmetric=True,  use_betrayal=True,
                                 use_hysteresis=True,  use_smoothing=False,
                                 use_belief=True),
    "No belief model":      dict(use_asymmetric=True,  use_betrayal=True,
                                 use_hysteresis=True,  use_smoothing=True,
                                 use_belief=False),
}


def run_ablation(n_seeds=N_SEEDS, n_episodes=N_EPISODES, rounds=ROUNDS) -> dict:
    """
    Run each ablation variant and return metrics keyed by variant name.
    """
    results = {}
    for name, flags in ABLATION_VARIANTS.items():
        results[name] = _run_agent(
            AblationTrustAgent, flags, AdversarialAgent, {},
            n_seeds, n_episodes, rounds
        )
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Gradually corrupting opponent
# ---------------------------------------------------------------------------

def run_corrupting_opponent(n_seeds=N_SEEDS, n_episodes=N_EPISODES,
                             rounds=ROUNDS) -> dict:
    """
    All three agents face a GraduallyCorruptingAgent (C) instead of AdversarialAgent.
    Tests whether agents can detect slow adversarial drift.
    """
    agents = {
        "ImprovedTrust": (ImprovedTrustAgent, {}),
        "TitForTat":     (TitForTatAgent,     {}),
        "Pavlov":        (PavlovAgent,         {}),
    }
    corrupt_kwargs = {"corrupt_over": 100, "max_defect_prob": 0.8}
    results = {}
    for name, (cls, kwargs) in agents.items():
        results[name] = _run_agent(
            cls, kwargs, GraduallyCorruptingAgent, corrupt_kwargs,
            n_seeds, n_episodes, rounds
        )
    return results
