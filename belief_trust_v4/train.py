"""
train.py
Training and evaluation for belief_trust_v4.

BeliefTrustAgent trains ONLINE — one gradient step per round, no episode
boundary needed. So "training" and "evaluation" are the same loop: the
agent learns continuously while we collect metrics.

We run two phases per seed:
  1. Warm-up  (N_WARMUP episodes): agent learns, metrics NOT collected
  2. Eval     (N_EVAL   episodes): agent continues learning, metrics collected

This mirrors how we'd deploy the agent in practice: it keeps adapting.

Baselines (ImprovedTrust, TitForTat, Pavlov) run for N_EVAL episodes only.

All metric dicts have the same structure so plot.py works unchanged.
"""
import os
import numpy as np
import torch
from environment import PDEnvironment
from agent import BeliefTrustAgent
from opponents import (
    ImprovedTrustAgent, TitForTatAgent, PavlovAgent,
    CooperativeAgent, AdversarialAgent, GraduallyCorruptingAgent,
)

N_WARMUP  = 200   # warm-up episodes (learning only, no metrics)
N_EVAL    = 200   # eval episodes (learning continues + metrics collected)
N_SEEDS   = 20
ROUNDS    = 10
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")


# ---------------------------------------------------------------------------
# Metric collection
# ---------------------------------------------------------------------------

def _collect(agent_a, env, rng, n_episodes, corrupting_agent=None) -> dict:
    """Collect metrics for agent_a (id=0) over n_episodes."""
    m = {
        "trust_B": [], "trust_C": [],
        "coop_rate_B": [], "coop_rate_C": [],
        "reward_B": [], "reward_C": [],
        "actions_vs_B": [], "actions_vs_C": [],
        "acc_B": [], "acc_C": [],   # prediction accuracy (BeliefTrust only)
    }
    for _ in range(n_episodes):
        if corrupting_agent is not None:
            corrupting_agent.notify_episode()
        result = env.run_episode(rng)
        m["trust_B"].append(agent_a.get_trust(1))
        m["trust_C"].append(agent_a.get_trust(2))
        # Prediction accuracy (only BeliefTrustAgent has this method)
        if hasattr(agent_a, "prediction_accuracy"):
            m["acc_B"].append(agent_a.prediction_accuracy(1))
            m["acc_C"].append(agent_a.prediction_accuracy(2))
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
        avg[key]            = np.mean([m[key] for m in all_metrics], axis=0)
        avg[f"{key}_std"]   = np.std( [m[key] for m in all_metrics], axis=0)
    for key in ["coop_rate_B", "coop_rate_C", "reward_B", "reward_C"]:
        arrays  = [np.array(m[key]) for m in all_metrics]
        min_len = min(len(a) for a in arrays)
        stacked = np.array([a[:min_len] for a in arrays])
        avg[key]          = stacked.mean(axis=0)
        avg[f"{key}_std"] = stacked.std(axis=0)
    # Prediction accuracy (may be empty for non-belief agents)
    for key in ["acc_B", "acc_C"]:
        arrays = [m[key] for m in all_metrics if m[key]]
        if arrays:
            min_len = min(len(a) for a in arrays)
            stacked = np.array([a[:min_len] for a in arrays])
            avg[key]          = stacked.mean(axis=0)
            avg[f"{key}_std"] = stacked.std(axis=0)
        else:
            avg[key] = avg[f"{key}_std"] = np.array([])
    avg["actions_vs_B"]        = all_metrics[0]["actions_vs_B"]
    avg["actions_vs_C"]        = all_metrics[0]["actions_vs_C"]
    avg["n_episodes"]          = n_episodes
    avg["rounds_per_interaction"] = rounds
    return avg


# ---------------------------------------------------------------------------
# BeliefTrustAgent runner
# ---------------------------------------------------------------------------

def _run_belief_seed(seed: int, opp_c_cls, opp_c_kwargs: dict) -> dict:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    agent_a = BeliefTrustAgent(0, n_agents=3)
    opp_c   = opp_c_cls(2, n_agents=3, **opp_c_kwargs)
    agents  = [agent_a, CooperativeAgent(1, n_agents=3), opp_c]
    env     = PDEnvironment(agents, rounds_per_interaction=ROUNDS)
    corrupting = opp_c if isinstance(opp_c, GraduallyCorruptingAgent) else None

    # --- Warm-up: learn without collecting metrics ---
    for _ in range(N_WARMUP):
        if corrupting is not None:
            corrupting.notify_episode()
        env.run_episode(rng)

    # Save checkpoint after warm-up
    os.makedirs(CKPT_DIR, exist_ok=True)
    agent_a.save(os.path.join(CKPT_DIR, f"belief_seed_{seed}.pt"))

    # Reset accuracy counters so eval accuracy is clean
    agent_a.reset_accuracy()

    # --- Eval: continue learning, collect metrics ---
    return _collect(agent_a, env, rng, N_EVAL, corrupting)


def run_belief_adversarial(n_seeds: int = N_SEEDS) -> dict:
    all_metrics = []
    for s in range(n_seeds):
        print(f"  seed {s+1}/{n_seeds}", end="\r")
        all_metrics.append(_run_belief_seed(s, AdversarialAgent, {}))
    print()
    return _average(all_metrics, N_EVAL, ROUNDS)


def run_belief_corrupting(n_seeds: int = N_SEEDS) -> dict:
    ck = {"corrupt_over": 100, "max_defect_prob": 0.8}
    all_metrics = []
    for s in range(n_seeds):
        print(f"  seed {s+1}/{n_seeds}", end="\r")
        all_metrics.append(_run_belief_seed(s, GraduallyCorruptingAgent, ck))
    print()
    return _average(all_metrics, N_EVAL, ROUNDS)


# ---------------------------------------------------------------------------
# Baseline runners (same eval protocol)
# ---------------------------------------------------------------------------

def _run_baseline_seed(seed: int, agent_cls, agent_kwargs: dict,
                        opp_c_cls, opp_c_kwargs: dict) -> dict:
    rng     = np.random.default_rng(seed)
    agent_a = agent_cls(0, n_agents=3, **agent_kwargs)
    opp_c   = opp_c_cls(2, n_agents=3, **opp_c_kwargs)
    agents  = [agent_a, CooperativeAgent(1, n_agents=3), opp_c]
    env     = PDEnvironment(agents, rounds_per_interaction=ROUNDS)
    corrupting = opp_c if isinstance(opp_c, GraduallyCorruptingAgent) else None
    return _collect(agent_a, env, rng, N_EVAL, corrupting)


def run_all_baselines(opp_c_cls, opp_c_kwargs: dict,
                      n_seeds: int = N_SEEDS) -> dict:
    baselines = {
        "ImprovedTrust": (ImprovedTrustAgent, {}),
        "TitForTat":     (TitForTatAgent,     {}),
        "Pavlov":        (PavlovAgent,         {}),
    }
    results = {}
    for name, (cls, kwargs) in baselines.items():
        all_m = [_run_baseline_seed(s, cls, kwargs, opp_c_cls, opp_c_kwargs)
                 for s in range(n_seeds)]
        results[name] = _average(all_m, N_EVAL, ROUNDS)
    return results
