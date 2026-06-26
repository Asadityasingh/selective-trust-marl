"""
train_rl.py
REINFORCE training loop for LearnedTrustAgent.

Training procedure per seed:
  - Run N_TRAIN episodes with training=True (soft policy, log-probs recorded)
  - After each episode: compute REINFORCE loss, backprop, clear buffers
  - After training: switch to eval mode, run N_EVAL episodes to collect metrics

The eval metrics dict has the same structure as v2's _average() output so
it plugs directly into compare.py and plot.py.
"""
import os
import numpy as np
import torch
import torch.optim as optim

from environment import PDEnvironment
from learned_agent import LearnedTrustAgent
from opponents import CooperativeAgent, AdversarialAgent, GraduallyCorruptingAgent

N_TRAIN   = 500    # training episodes per seed (more = better convergence)
N_EVAL    = 200    # evaluation episodes per seed (matches v2)
N_SEEDS   = 20
ROUNDS    = 10
LR        = 1e-3   # lower LR for stability
GAMMA     = 0.95   # shorter horizon — PD rewards are dense
CKPT_DIR  = os.path.join(os.path.dirname(__file__), "checkpoints")


# ---------------------------------------------------------------------------
# Single-seed training + evaluation
# ---------------------------------------------------------------------------

def _collect_eval(agent_a, env, rng, n_episodes, corrupting_agent=None) -> dict:
    """Collect evaluation metrics — identical structure to v2's _collect()."""
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


def train_one_seed(seed: int, opponent_c_cls, opponent_c_kwargs: dict,
                   n_train: int = N_TRAIN, n_eval: int = N_EVAL,
                   rounds: int = ROUNDS) -> tuple[dict, list]:
    """
    Train LearnedTrustAgent for one seed.
    Returns (eval_metrics, training_loss_curve).
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    # Epsilon decays from 0.3 -> 0.02 over training to encourage early exploration
    agent_a = LearnedTrustAgent(0, n_agents=3, training=True, epsilon=0.3)
    optimizer = optim.Adam(agent_a.model.parameters(), lr=LR)

    opp_c = opponent_c_cls(2, n_agents=3, **opponent_c_kwargs)
    train_agents = [agent_a, CooperativeAgent(1, n_agents=3), opp_c]
    train_env = PDEnvironment(train_agents, rounds_per_interaction=rounds)

    loss_curve = []

    # --- Training loop ---
    for ep in range(n_train):
        # Decay epsilon: 0.3 -> 0.02 linearly over training
        agent_a.epsilon = max(0.02, 0.3 - (0.3 - 0.02) * ep / n_train)

        if isinstance(opp_c, GraduallyCorruptingAgent):
            opp_c.notify_episode()

        train_env.run_episode(rng)

        # Only update if we collected steps this episode
        if agent_a._episode_steps:
            loss = agent_a.compute_loss(gamma=GAMMA)
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(agent_a.model.parameters(), max_norm=1.0)
            optimizer.step()
            loss_curve.append(loss.item())

        agent_a.clear_buffers()

    # --- Save checkpoint ---
    os.makedirs(CKPT_DIR, exist_ok=True)
    agent_a.save(os.path.join(CKPT_DIR, f"seed_{seed}.pt"))

    # --- Evaluation: fresh agent with trained weights, hard policy ---
    agent_eval = LearnedTrustAgent(0, n_agents=3, training=False)
    agent_eval.load(os.path.join(CKPT_DIR, f"seed_{seed}.pt"))
    agent_eval.set_eval()

    # Fresh opponents for eval (reset state)
    opp_c_eval = opponent_c_cls(2, n_agents=3, **opponent_c_kwargs)
    eval_agents = [agent_eval, CooperativeAgent(1, n_agents=3), opp_c_eval]
    eval_env = PDEnvironment(eval_agents, rounds_per_interaction=rounds)

    corrupting_eval = opp_c_eval if isinstance(opp_c_eval, GraduallyCorruptingAgent) else None
    eval_metrics = _collect_eval(agent_eval, eval_env, rng, n_eval, corrupting_eval)

    return eval_metrics, loss_curve


# ---------------------------------------------------------------------------
# Multi-seed averaging (same format as v2's _average())
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Public experiment runners
# ---------------------------------------------------------------------------

def run_learned_adversarial(n_seeds: int = N_SEEDS) -> tuple[dict, np.ndarray]:
    """Train + eval against AdversarialAgent. Returns (metrics, loss_curves)."""
    all_metrics, all_losses = [], []
    for s in range(n_seeds):
        print(f"  seed {s+1}/{n_seeds}", end="\r")
        m, lc = train_one_seed(s, AdversarialAgent, {})
        all_metrics.append(m)
        all_losses.append(lc)
    print()
    # Pad loss curves to same length for averaging
    max_len = max(len(lc) for lc in all_losses)
    padded = [lc + [lc[-1]] * (max_len - len(lc)) for lc in all_losses]
    return _average(all_metrics, N_EVAL, ROUNDS), np.array(padded)


def run_learned_corrupting(n_seeds: int = N_SEEDS) -> tuple[dict, np.ndarray]:
    """Train + eval against GraduallyCorruptingAgent."""
    corrupt_kwargs = {"corrupt_over": 100, "max_defect_prob": 0.8}
    all_metrics, all_losses = [], []
    for s in range(n_seeds):
        print(f"  seed {s+1}/{n_seeds}", end="\r")
        m, lc = train_one_seed(s, GraduallyCorruptingAgent, corrupt_kwargs)
        all_metrics.append(m)
        all_losses.append(lc)
    print()
    max_len = max(len(lc) for lc in all_losses)
    padded = [lc + [lc[-1]] * (max_len - len(lc)) for lc in all_losses]
    return _average(all_metrics, N_EVAL, ROUNDS), np.array(padded)
