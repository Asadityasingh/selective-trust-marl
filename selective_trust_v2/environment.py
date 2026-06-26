"""
environment.py
Repeated Prisoner's Dilemma environment for N agents.
Unchanged from v1 — kept clean so any agent interface works here.
"""
import numpy as np

# Actions: 0 = Cooperate, 1 = Defect
PAYOFF = {
    (0, 0): (3, 3),
    (0, 1): (0, 5),
    (1, 0): (5, 0),
    (1, 1): (1, 1),
}


class PDEnvironment:
    def __init__(self, agents: list, rounds_per_interaction: int = 10):
        self.agents = agents
        self.n = len(agents)
        self.rounds = rounds_per_interaction

    def _get_pairs(self, rng: np.random.Generator) -> list:
        pairs = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        rng.shuffle(pairs)
        return pairs

    def run_episode(self, rng: np.random.Generator) -> dict:
        rewards = {i: 0.0 for i in range(self.n)}
        logs = []

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
