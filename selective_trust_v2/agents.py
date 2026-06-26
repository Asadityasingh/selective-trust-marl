"""
agents.py
All agent definitions for selective_trust_v2.

Agents under test (id=0):
  - ImprovedTrustAgent     : full 5-mechanism trust agent (our main agent)
  - TitForTatAgent         : classical baseline — copy opponent's last action
  - PavlovAgent            : Win-Stay-Lose-Shift baseline

Opponent agents (id=1, 2):
  - CooperativeAgent       : 90% cooperate
  - AdversarialAgent       : buildup then periodic defection
  - GraduallyCorruptingAgent: starts fully cooperative, linearly increases
                              defection probability over episodes

Ablation:
  - AblationTrustAgent     : ImprovedTrustAgent with individual mechanisms
                             toggled off, controlled by boolean flags
"""
import numpy as np


# ---------------------------------------------------------------------------
# Agents under test
# ---------------------------------------------------------------------------

class ImprovedTrustAgent:
    """
    Full trust agent with all 5 mechanisms:
      1. Asymmetric update (+0.05 / -0.20)
      2. Betrayal penalty  (-0.30 extra if trust>0.7 and opponent defects)
      3. Hysteresis        (two thresholds: coop=0.7, defect=0.5)
      4. Trust smoothing   (EMA with alpha=0.1)
      5. Belief model      (EMA defection rate; override if belief>0.6)
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
        if opponent_action == 0:
            delta = 0.05
        else:
            delta = -0.2
            if t > 0.7:
                delta -= 0.3
        new_trust = float(np.clip(t + delta, 0.0, 1.0))
        self.trust[opponent_id] = 0.9 * t + 0.1 * new_trust
        defected = 1.0 if opponent_action == 1 else 0.0
        self.belief_defect[opponent_id] = (
            0.9 * self.belief_defect[opponent_id] + 0.1 * defected
        )

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]


class AblationTrustAgent:
    """
    ImprovedTrustAgent with individual mechanisms togglable off.
    Used for ablation study — disable one mechanism at a time.

    Flags (all True = full ImprovedTrustAgent):
      use_asymmetric   : if False, use symmetric +0.05/-0.05
      use_betrayal     : if False, skip the -0.30 betrayal penalty
      use_hysteresis   : if False, use single threshold (0.6)
      use_smoothing    : if False, apply delta directly (no EMA)
      use_belief       : if False, skip belief model override
    """

    def __init__(self, agent_id: int, n_agents: int,
                 use_asymmetric: bool = True,
                 use_betrayal: bool = True,
                 use_hysteresis: bool = True,
                 use_smoothing: bool = True,
                 use_belief: bool = True,
                 trust_init: float = 0.5):
        self.id = agent_id
        self.use_asymmetric = use_asymmetric
        self.use_betrayal = use_betrayal
        self.use_hysteresis = use_hysteresis
        self.use_smoothing = use_smoothing
        self.use_belief = use_belief
        opponents = [j for j in range(n_agents) if j != agent_id]
        self.trust = {j: trust_init for j in opponents}
        self.belief_defect = {j: 0.0 for j in opponents}
        self._last_action = {j: 0 for j in opponents}
        self._rng = np.random.default_rng()

    def act(self, opponent_id: int) -> int:
        t = self.trust[opponent_id]
        b = self.belief_defect[opponent_id]

        if self.use_belief and b > 0.6:
            action = 1
        elif self.use_hysteresis:
            if t > 0.7:
                action = 0
            elif t < 0.5:
                action = 1
            else:
                action = self._last_action[opponent_id]
        else:
            action = 0 if t > 0.6 else 1

        self._last_action[opponent_id] = action
        return action

    def update(self, opponent_id: int, opponent_action: int, _reward: float):
        t = self.trust[opponent_id]

        if self.use_asymmetric:
            delta = 0.05 if opponent_action == 0 else -0.2
        else:
            delta = 0.05 if opponent_action == 0 else -0.05

        if self.use_betrayal and opponent_action == 1 and t > 0.7:
            delta -= 0.3

        new_trust = float(np.clip(t + delta, 0.0, 1.0))

        if self.use_smoothing:
            self.trust[opponent_id] = 0.9 * t + 0.1 * new_trust
        else:
            self.trust[opponent_id] = new_trust

        defected = 1.0 if opponent_action == 1 else 0.0
        self.belief_defect[opponent_id] = (
            0.9 * self.belief_defect[opponent_id] + 0.1 * defected
        )

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]


class TitForTatAgent:
    """
    Classical Tit-for-Tat: cooperate on first move,
    then copy opponent's last action.
    Maintains per-opponent last-seen action.
    """

    def __init__(self, agent_id: int, n_agents: int):
        self.id = agent_id
        self._last_opp_action = {j: 0 for j in range(n_agents) if j != agent_id}

    def act(self, opponent_id: int) -> int:
        return self._last_opp_action[opponent_id]

    def update(self, opponent_id: int, opponent_action: int, _reward: float):
        self._last_opp_action[opponent_id] = opponent_action

    def get_trust(self, opponent_id: int) -> float:
        # Expose "trust" as 1 - last defection for unified metric logging
        return 1.0 - float(self._last_opp_action[opponent_id])


class PavlovAgent:
    """
    Win-Stay-Lose-Shift (Pavlov):
      - Win  (reward >= 3): repeat last action
      - Lose (reward <  3): switch action
    Maintains per-opponent state.
    """

    def __init__(self, agent_id: int, n_agents: int):
        self.id = agent_id
        self._last_action = {j: 0 for j in range(n_agents) if j != agent_id}
        self._last_reward = {j: 3.0 for j in range(n_agents) if j != agent_id}

    def act(self, opponent_id: int) -> int:
        if self._last_reward[opponent_id] >= 3:
            return self._last_action[opponent_id]          # win-stay
        return 1 - self._last_action[opponent_id]          # lose-shift

    def update(self, opponent_id: int, _opponent_action: int, reward: float):
        self._last_action[opponent_id] = self.act(opponent_id)
        self._last_reward[opponent_id] = reward

    def get_trust(self, opponent_id: int) -> float:
        return 1.0 - float(self._last_action[opponent_id])


# ---------------------------------------------------------------------------
# Opponent agents
# ---------------------------------------------------------------------------

class CooperativeAgent:
    """Cooperates 90% of the time."""

    def __init__(self, agent_id: int, n_agents: int, coop_prob: float = 0.9):
        self.id = agent_id
        self.coop_prob = coop_prob
        self._rng = np.random.default_rng()

    def act(self, _opponent_id: int) -> int:
        return 0 if self._rng.random() < self.coop_prob else 1

    def update(self, *_): pass


class AdversarialAgent:
    """
    Cooperates for `buildup` rounds then defects every `defect_every` rounds.
    Step counter persists across episodes (simulates a persistent adversary).
    """

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
    """
    New opponent type for paper experiment 3.

    Starts fully cooperative (defect_prob=0) and linearly increases
    its defection probability over `corrupt_over` episodes until it
    reaches `max_defect_prob`.

    This tests whether the trust agent can detect *slow* adversarial drift
    — a much harder problem than sudden periodic defection.
    """

    def __init__(self, agent_id: int, n_agents: int,
                 corrupt_over: int = 100,
                 max_defect_prob: float = 0.8):
        self.id = agent_id
        self.corrupt_over = corrupt_over
        self.max_defect_prob = max_defect_prob
        self._episode = 0          # incremented by notify_episode()
        self._rng = np.random.default_rng()

    def notify_episode(self):
        """Call once per episode from the training loop to advance corruption."""
        self._episode += 1

    @property
    def defect_prob(self) -> float:
        progress = min(self._episode / self.corrupt_over, 1.0)
        return progress * self.max_defect_prob

    def act(self, _opponent_id: int) -> int:
        return 1 if self._rng.random() < self.defect_prob else 0

    def update(self, *_): pass
