"""
Training loop for multi-agent simultaneous learning.

Both agents act and learn at the same time (non-adversarial setting).
Supports multiple seeds for statistical stability.
"""

import numpy as np
from environment import RepeatedPrisonersDilemma, COOPERATE
from agents import BaselineAgent, TrustAgent


def run_episode(env, agent_a, agent_b, agent_type="baseline"):
    """Run one episode with both agents learning simultaneously."""
    env.reset()
    agent_a.reset_episode()
    agent_b.reset_episode()

    rewards_a, rewards_b = [], []
    actions_a, actions_b = [], []

    for _ in range(env.num_rounds):
        state_a = env.get_state_for(0)
        state_b = env.get_state_for(1)

        action_a = agent_a.select_action(state_a)
        action_b = agent_b.select_action(state_b)

        reward_a, reward_b, done = env.step(action_a, action_b)

        next_state_a = env.get_state_for(0)
        next_state_b = env.get_state_for(1)

        # Both agents learn
        if agent_type == "baseline":
            agent_a.update(state_a, action_a, reward_a, next_state_a)
            agent_b.update(state_b, action_b, reward_b, next_state_b)
        else:
            agent_a.update(state_a, action_a, reward_a, next_state_a, action_b)
            agent_b.update(state_b, action_b, reward_b, next_state_b, action_a)

        rewards_a.append(reward_a)
        rewards_b.append(reward_b)
        actions_a.append(action_a)
        actions_b.append(action_b)

    return {
        "avg_reward_a": np.mean(rewards_a),
        "avg_reward_b": np.mean(rewards_b),
        "avg_reward": (np.mean(rewards_a) + np.mean(rewards_b)) / 2,
        "coop_rate_a": actions_a.count(COOPERATE) / len(actions_a),
        "coop_rate_b": actions_b.count(COOPERATE) / len(actions_b),
        "coop_rate": (actions_a.count(COOPERATE) + actions_b.count(COOPERATE)) / (2 * len(actions_a)),
        "actions_a": actions_a,
        "actions_b": actions_b,
        "trust_a": agent_a.trust_history.copy() if agent_type == "trust" else [],
        "trust_b": agent_b.trust_history.copy() if agent_type == "trust" else [],
    }


def run_experiment(agent_type="baseline", num_episodes=1000, num_rounds=100, seed=42):
    """Run full experiment for one seed."""
    np.random.seed(seed)
    env = RepeatedPrisonersDilemma(num_rounds=num_rounds)

    if agent_type == "baseline":
        agent_a = BaselineAgent()
        agent_b = BaselineAgent()
    else:
        agent_a = TrustAgent()
        agent_b = TrustAgent()

    all_metrics = []
    for ep in range(num_episodes):
        metrics = run_episode(env, agent_a, agent_b, agent_type)
        metrics["episode"] = ep
        all_metrics.append(metrics)

    return all_metrics


def run_multi_seed(agent_type="baseline", num_episodes=1000, num_rounds=100,
                   seeds=None):
    """Run experiment across multiple seeds, return averaged metrics."""
    if seeds is None:
        seeds = [42, 123, 456, 789, 1024]

    all_runs = []
    for seed in seeds:
        metrics = run_experiment(agent_type, num_episodes, num_rounds, seed)
        all_runs.append(metrics)

    # Average across seeds
    num_ep = len(all_runs[0])
    averaged = []
    for ep in range(num_ep):
        avg = {
            "episode": ep,
            "avg_reward": np.mean([run[ep]["avg_reward"] for run in all_runs]),
            "coop_rate": np.mean([run[ep]["coop_rate"] for run in all_runs]),
            "reward_std": np.std([run[ep]["avg_reward"] for run in all_runs]),
            "coop_std": np.std([run[ep]["coop_rate"] for run in all_runs]),
        }
        # Keep last seed's detailed data for action/trust plots
        if ep == num_ep - 1:
            avg["actions_a"] = all_runs[-1][ep]["actions_a"]
            avg["actions_b"] = all_runs[-1][ep]["actions_b"]
            avg["trust_a"] = all_runs[-1][ep]["trust_a"]
            avg["trust_b"] = all_runs[-1][ep]["trust_b"]
        averaged.append(avg)

    return averaged
