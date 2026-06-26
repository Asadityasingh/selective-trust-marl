"""
opponents.py
All non-learned agents: opponents + rule-based baselines.
Copied from selective_trust_v2/agents.py — interface unchanged.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Rule-based baselines (agents under test)
# ---------------------------------------------------------------------------

class ImprovedTrustAgent:
    def __init__(self, agent_id: int, n_agents: int,
                 coop_threshold: float = 0.7, defect_threshold: float = 0.5,
                 epsilon: float = 0.05, trust_init: float = 0.5):
        self.id = agent_id
        self.coop_threshold = coop_threshold
        self.defect_threshold = defect_threshold
        self.epsilon = epsilon
        opponents = [j for j in range(n_agents) if j != agent_id]
        self.trust = {j: trust_init for j in opponents}
        self.belief_defect = {j: 0.0 for j in opponents}
        self._last_action = {j: 0 for j in opponents}
        self._rng = np.random.default_rng()

    def act(self, opponent_id: int) -> int:
        if self._rng.random() < self.epsilon:
            action = int(self._rng.integers(0, 2))
            self._last_action[opponent_id] = action
            return action
        t = self.trust[opponent_id]
        b = self.belief_defect[opponent_id]
        if b > 0.6:
            action = 1
        elif t > self.coop_threshold:
            action = 0
        elif t < self.defect_threshold:
            action = 1
        else:
            action = self._last_action[opponent_id]
        self._last_action[opponent_id] = action
        return action

    def update(self, opponent_id: int, opponent_action: int, _reward: float):
        t = self.trust[opponent_id]
        delta = 0.05 if opponent_action == 0 else -0.2
        if opponent_action == 1 and t > 0.7:
            delta -= 0.3
        new_trust = float(np.clip(t + delta, 0.0, 1.0))
        self.trust[opponent_id] = 0.9 * t + 0.1 * new_trust
        defected = 1.0 if opponent_action == 1 else 0.0
        self.belief_defect[opponent_id] = (
            0.9 * self.belief_defect[opponent_id] + 0.1 * defected
        )

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]


class TitForTatAgent:
    def __init__(self, agent_id: int, n_agents: int):
        self.id = agent_id
        self._last_opp_action = {j: 0 for j in range(n_agents) if j != agent_id}

    def act(self, opponent_id: int) -> int:
        return self._last_opp_action[opponent_id]

    def update(self, opponent_id: int, opponent_action: int, _reward: float):
        self._last_opp_action[opponent_id] = opponent_action

    def get_trust(self, opponent_id: int) -> float:
        return 1.0 - float(self._last_opp_action[opponent_id])


class PavlovAgent:
    def __init__(self, agent_id: int, n_agents: int):
        self.id = agent_id
        self._last_action = {j: 0 for j in range(n_agents) if j != agent_id}
        self._last_reward = {j: 3.0 for j in range(n_agents) if j != agent_id}

    def act(self, opponent_id: int) -> int:
        if self._last_reward[opponent_id] >= 3:
            return self._last_action[opponent_id]
        return 1 - self._last_action[opponent_id]

    def update(self, opponent_id: int, _opponent_action: int, reward: float):
        self._last_action[opponent_id] = self.act(opponent_id)
        self._last_reward[opponent_id] = reward

    def get_trust(self, opponent_id: int) -> float:
        return 1.0 - float(self._last_action[opponent_id])


# ---------------------------------------------------------------------------
# Opponent agents
# ---------------------------------------------------------------------------

class CooperativeAgent:
    def __init__(self, agent_id: int, n_agents: int, coop_prob: float = 0.9):
        self.id = agent_id
        self.coop_prob = coop_prob
        self._rng = np.random.default_rng()

    def act(self, _opponent_id: int) -> int:
        return 0 if self._rng.random() < self.coop_prob else 1

    def update(self, *_): pass


class AdversarialAgent:
    def __init__(self, agent_id: int, n_agents: int,
                 buildup: int = 3, defect_every: int = 3):
        self.id = agent_id
        self.buildup = buildup
        self.defect_every = defect_every
        self._step: dict = {}

    def act(self, opponent_id: int) -> int:
        step = self._step.get(opponent_id, 0)
        self._step[opponent_id] = step + 1
        if step < self.buildup:
            return 0
        return 1 if (step - self.buildup) % self.defect_every == 0 else 0

    def update(self, *_): pass


class GraduallyCorruptingAgent:
    def __init__(self, agent_id: int, n_agents: int,
                 corrupt_over: int = 100, max_defect_prob: float = 0.8):
        self.id = agent_id
        self.corrupt_over = corrupt_over
        self.max_defect_prob = max_defect_prob
        self._episode = 0
        self._rng = np.random.default_rng()

    def notify_episode(self):
        self._episode += 1

    @property
    def defect_prob(self) -> float:
        return min(self._episode / self.corrupt_over, 1.0) * self.max_defect_prob

    def act(self, _opponent_id: int) -> int:
        return 1 if self._rng.random() < self.defect_prob else 0

    def update(self, *_): pass
