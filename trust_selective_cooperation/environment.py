"""
environment.py
Repeated Prisoner's Dilemma environment for N agents.
Handles pairwise interactions and payoff computation.
"""
import numpy as np

# Payoff matrix: payoff[my_action][opp_action] -> (my_reward, opp_reward)
# Actions: 0 = Cooperate, 1 = Defect
PAYOFF = {
    (0, 0): (3, 3),
    (0, 1): (0, 5),
    (1, 0): (5, 0),
    (1, 1): (1, 1),
}


class PDEnvironment:
    """
    Multi-agent Repeated Prisoner's Dilemma.
    Each episode shuffles agent pairs; each pair plays `rounds_per_interaction` rounds.
    """

    def __init__(self, agents: list, rounds_per_interaction: int = 10):
        self.agents = agents  # list of agent objects
        self.n = len(agents)
        self.rounds = rounds_per_interaction

    def _get_pairs(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        """Return all unique pairs, shuffled."""
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        rng.shuffle(pairs)
        return pairs

    def run_episode(self, rng: np.random.Generator) -> dict:
        """
        Run one episode (all pairs, each for `self.rounds` rounds).
        Returns per-agent cumulative rewards and interaction logs.
        """
        rewards = {i: 0.0 for i in range(self.n)}
        logs = []  # list of (i, j, actions_i, actions_j, rewards_i, rewards_j)

        for i, j in self._get_pairs(rng):
            agent_i, agent_j = self.agents[i], self.agents[j]
            actions_i, actions_j, rew_i, rew_j = [], [], 0.0, 0.0

            for _ in range(self.rounds):
                a_i = agent_i.act(j)
                a_j = agent_j.act(i)
                r_i, r_j = PAYOFF[(a_i, a_j)]

                agent_i.update(j, a_j, r_i)
                agent_j.update(i, a_i, r_j)

                actions_i.append(a_i)
                actions_j.append(a_j)
                rew_i += r_i
                rew_j += r_j

            rewards[i] += rew_i
            rewards[j] += rew_j
            logs.append((i, j, actions_i, actions_j, rew_i, rew_j))

        return {"rewards": rewards, "logs": logs}
