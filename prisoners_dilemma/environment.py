"""
Repeated Prisoner's Dilemma Environment.

Actions: 0 = Cooperate (C), 1 = Defect (D)

Payoff Matrix:
    (C, C) → (3, 3)
    (C, D) → (0, 5)
    (D, C) → (5, 0)
    (D, D) → (1, 1)
"""

COOPERATE = 0
DEFECT = 1

PAYOFF = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1),
}


class RepeatedPrisonersDilemma:
    """Two-player Repeated Prisoner's Dilemma."""

    def __init__(self, num_rounds=100):
        self.num_rounds = num_rounds
        self.reset()

    def reset(self):
        self.round = 0
        self.history_a = []
        self.history_b = []

    def get_state_for(self, agent_id):
        """State from agent's perspective: (my_last, opponent_last). None if first round."""
        if not self.history_a:
            return None
        if agent_id == 0:
            return (self.history_a[-1], self.history_b[-1])
        return (self.history_b[-1], self.history_a[-1])

    def step(self, action_a, action_b):
        """Execute one round. Returns (reward_a, reward_b, done)."""
        reward_a, reward_b = PAYOFF[(action_a, action_b)]
        self.history_a.append(action_a)
        self.history_b.append(action_b)
        self.round += 1
        return reward_a, reward_b, self.round >= self.num_rounds
