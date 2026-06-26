"""
Agent definitions for the Repeated Prisoner's Dilemma.

- BaselineAgent: Standard Q-learning with epsilon-greedy
- TrustAgent: Trust-based policy combined with epsilon-greedy exploration
"""

import numpy as np
from environment import COOPERATE, DEFECT


class BaselineAgent:
    """Q-learning agent. State = (my_last_action, opponent_last_action)."""

    def __init__(self, lr=0.1, gamma=0.95, epsilon=0.1):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def _get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice([COOPERATE, DEFECT])
        q_c = self._get_q(state, COOPERATE)
        q_d = self._get_q(state, DEFECT)
        if q_c == q_d:
            return np.random.choice([COOPERATE, DEFECT])
        return COOPERATE if q_c > q_d else DEFECT

    def update(self, state, action, reward, next_state):
        q_cur = self._get_q(state, action)
        q_next = max(self._get_q(next_state, COOPERATE), self._get_q(next_state, DEFECT))
        self.q_table[(state, action)] = q_cur + self.lr * (
            reward + self.gamma * q_next - q_cur
        )

    def reset_episode(self):
        """Q-table persists across episodes; nothing to reset."""
        pass


class TrustAgent:
    """Trust-based agent with epsilon-greedy exploration.

    Trust update:
        opponent cooperates → trust += trust_inc
        opponent defects   → trust -= trust_dec
    Policy:
        trust > threshold → cooperate, else defect
    """

    def __init__(self, trust_init=0.5, trust_inc=0.05, trust_dec=0.05,
                 threshold=0.4, epsilon=0.1):
        self.trust_init = trust_init
        self.trust_inc = trust_inc
        self.trust_dec = trust_dec
        self.threshold = threshold
        self.epsilon = epsilon
        self.trust = trust_init
        self.trust_history = []

    def select_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice([COOPERATE, DEFECT])
        return COOPERATE if self.trust > self.threshold else DEFECT

    def update(self, state, action, reward, next_state, opponent_action):
        """Update trust based on opponent's action."""
        if opponent_action == COOPERATE:
            self.trust += self.trust_inc
        else:
            self.trust -= self.trust_dec
        self.trust = np.clip(self.trust, 0.0, 1.0)
        self.trust_history.append(self.trust)

    def reset_episode(self):
        """Trust persists across episodes (agents build long-term trust)."""
        self.trust_history = []
