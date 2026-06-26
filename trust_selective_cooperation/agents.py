"""
agents.py
Agent definitions for the trust-based selective cooperation experiment.
  - TrustAgent      (Agent A): maintains per-opponent trust, decides based on threshold
  - CooperativeAgent (Agent B): cooperates with small noise
  - AdversarialAgent (Agent C): builds trust then defects periodically
"""
import numpy as np


class TrustAgent:
    """
    Maintains a trust value in [0,1] for each opponent.
    Cooperates if trust > threshold, else defects.
    Optional epsilon-greedy exploration.
    """

    def __init__(self, agent_id: int, n_agents: int,
                 threshold: float = 0.6, epsilon: float = 0.05,
                 trust_init: float = 0.5):
        self.id = agent_id
        self.threshold = threshold
        self.epsilon = epsilon
        # trust[j] = trust toward agent j
        self.trust = {j: trust_init for j in range(n_agents) if j != agent_id}
        self._rng = np.random.default_rng()

    def act(self, opponent_id: int) -> int:
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, 2))  # explore
        return 0 if self.trust[opponent_id] > self.threshold else 1

    def update(self, opponent_id: int, opponent_action: int, _reward: float):
        delta = 0.05 if opponent_action == 0 else -0.1
        self.trust[opponent_id] = float(
            np.clip(self.trust[opponent_id] + delta, 0.0, 1.0)
        )

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]


class ImprovedTrustAgent:
    """
    Robust trust-based agent with 5 improvements over TrustAgent:
      1. Asymmetric update  : slow trust gain (+0.02), fast trust loss (-0.2)
      2. Betrayal penalty   : extra -0.3 if opponent defects while trust > 0.7
      3. Hysteresis         : two thresholds (0.7 cooperate, 0.5 defect) to
                              prevent oscillation near the boundary
      4. Trust smoothing    : exponential moving average (alpha=0.9) for inertia
      5. Belief model       : tracks EMA of opponent defection rate;
                              forces defect if belief_defect > 0.6
    """

    def __init__(self, agent_id: int, n_agents: int,
                 coop_threshold: float = 0.7,
                 defect_threshold: float = 0.5,
                 epsilon: float = 0.05,
                 trust_init: float = 0.5):
        self.id = agent_id
        self.coop_threshold = coop_threshold
        self.defect_threshold = defect_threshold
        self.epsilon = epsilon

        opponents = [j for j in range(n_agents) if j != agent_id]
        self.trust = {j: trust_init for j in opponents}          # smoothed trust
        self.belief_defect = {j: 0.0 for j in opponents}        # EMA defection rate
        self._last_action = {j: 0 for j in opponents}           # hysteresis memory
        self._rng = np.random.default_rng()

    def act(self, opponent_id: int) -> int:
        if self._rng.random() < self.epsilon:
            action = int(self._rng.integers(0, 2))
            self._last_action[opponent_id] = action
            return action

        t = self.trust[opponent_id]
        b = self.belief_defect[opponent_id]

        # Belief model overrides: if opponent is reliably defecting, defect
        if b > 0.6:
            action = 1
        elif t > self.coop_threshold:
            action = 0
        elif t < self.defect_threshold:
            action = 1
        else:
            # Hysteresis zone: hold previous action to avoid oscillation
            action = self._last_action[opponent_id]

        self._last_action[opponent_id] = action
        return action

    def update(self, opponent_id: int, opponent_action: int, _reward: float):
        t = self.trust[opponent_id]

        # 1: Asymmetric delta — gain must outpace occasional noise from cooperative
        #    agents (10% defect rate). +0.05 ensures trust_B stays high while
        #    the -0.2 / betrayal penalty still drives trust_C to near zero.
        if opponent_action == 0:   # cooperated
            delta = 0.05
        else:                      # defected
            delta = -0.2
            # 2: Betrayal penalty — extra hit when trust was high
            if t > 0.7:
                delta -= 0.3

        # 3 & 4: Update then smooth
        #   new_trust = clip(t + delta)          — where trust *should* go
        #   trust     = EMA(trust, new_trust)    — inertia, prevents jitter
        new_trust = float(np.clip(t + delta, 0.0, 1.0))
        alpha = 0.1
        self.trust[opponent_id] = (1 - alpha) * t + alpha * new_trust

        # 5: Belief model — EMA of defection indicator
        defected = 1.0 if opponent_action == 1 else 0.0
        self.belief_defect[opponent_id] = (
            0.9 * self.belief_defect[opponent_id] + 0.1 * defected
        )

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]


class CooperativeAgent:
    """Always cooperates with small random noise (10% defect)."""

    def __init__(self, agent_id: int, n_agents: int, coop_prob: float = 0.9):
        self.id = agent_id
        self.coop_prob = coop_prob
        self._rng = np.random.default_rng()

    def act(self, _opponent_id: int) -> int:
        return 0 if self._rng.random() < self.coop_prob else 1

    def update(self, *_):
        pass  # stateless


class AdversarialAgent:
    """
    Cooperates initially to build trust, then defects periodically.
    Pattern: cooperate for `buildup` rounds, then defect every `defect_every` rounds.
    """

    def __init__(self, agent_id: int, n_agents: int,
                 buildup: int = 3, defect_every: int = 3):
        self.id = agent_id
        self.buildup = buildup
        self.defect_every = defect_every
        self._step: dict[int, int] = {}  # per-opponent step counter

    def act(self, opponent_id: int) -> int:
        step = self._step.get(opponent_id, 0)
        self._step[opponent_id] = step + 1
        if step < self.buildup:
            return 0  # cooperate to build trust
        # after buildup: defect every `defect_every` rounds
        return 1 if (step - self.buildup) % self.defect_every == 0 else 0

    def update(self, *_):
        pass  # stateless
