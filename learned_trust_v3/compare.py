"""
compare.py
Runs rule-based baselines (ImprovedTrust, TitForTat, Pavlov) using the same
N_EVAL=200 episodes / 20 seeds protocol as the learned agent evaluation.
This ensures a fair apples-to-apples comparison.
"""
import numpy as np
from environment import PDEnvironment
from opponents import (
    ImprovedTrustAgent, TitForTatAgent, PavlovAgent,
    CooperativeAgent, AdversarialAgent, GraduallyCorruptingAgent,
)
from train_rl import N_EVAL, N_SEEDS, ROUNDS


def _collect(agent_a, env, rng, n_episodes, corrupting_agent=None) -> dict:
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


def _average(all_metrics, n_episodes, rounds):
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


def _run_baseline(agent_cls, agent_kwargs, opp_c_cls, opp_c_kwargs,
                  n_seeds=N_SEEDS, n_episodes=N_EVAL, rounds=ROUNDS):
    all_metrics = []
    for s in range(n_seeds):
        rng = np.random.default_rng(s)
        agent_a = agent_cls(0, n_agents=3, **agent_kwargs)
        opp_c = opp_c_cls(2, n_agents=3, **opp_c_kwargs)
        agents = [agent_a, CooperativeAgent(1, n_agents=3), opp_c]
        env = PDEnvironment(agents, rounds_per_interaction=rounds)
        corrupting = opp_c if isinstance(opp_c, GraduallyCorruptingAgent) else None
        all_metrics.append(_collect(agent_a, env, rng, n_episodes, corrupting))
    return _average(all_metrics, n_episodes, rounds)


def run_all_baselines_adversarial(n_seeds=N_SEEDS) -> dict:
    return {
        "ImprovedTrust": _run_baseline(ImprovedTrustAgent, {}, AdversarialAgent, {},
                                        n_seeds=n_seeds),
        "TitForTat":     _run_baseline(TitForTatAgent,     {}, AdversarialAgent, {},
                                        n_seeds=n_seeds),
        "Pavlov":        _run_baseline(PavlovAgent,         {}, AdversarialAgent, {},
                                        n_seeds=n_seeds),
    }


def run_all_baselines_corrupting(n_seeds=N_SEEDS) -> dict:
    ck = {"corrupt_over": 100, "max_defect_prob": 0.8}
    return {
        "ImprovedTrust": _run_baseline(ImprovedTrustAgent, {}, GraduallyCorruptingAgent,
                                        ck, n_seeds=n_seeds),
        "TitForTat":     _run_baseline(TitForTatAgent,     {}, GraduallyCorruptingAgent,
                                        ck, n_seeds=n_seeds),
        "Pavlov":        _run_baseline(PavlovAgent,         {}, GraduallyCorruptingAgent,
                                        ck, n_seeds=n_seeds),
    }
