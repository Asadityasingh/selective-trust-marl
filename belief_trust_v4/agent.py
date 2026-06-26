"""
agent.py
BeliefTrustAgent: learns trust as a belief about opponent behavior.

Core loop (per round):
  1. Build history input from last K interactions with this opponent
  2. Predict p_defect = BeliefNet(history)
  3. Derive trust = 1 - p_defect
  4. Decide action via hysteresis threshold policy
  5. Observe opponent's actual action
  6. Compute BCE loss against actual action
  7. One gradient step (online learning)

Key properties:
  - Training signal is opponent behavior only — reward is never used
  - Same network weights shared across all opponents (general predictor)
  - Same act/update/get_trust interface as all v2/v3 agents
  - Tracks prediction accuracy per opponent for paper metrics
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from model import BeliefNet

MAX_REWARD = 5.0


class BeliefTrustAgent:

    def __init__(self, agent_id: int, n_agents: int,
                 K: int = 5,
                 lr: float = 1e-3,
                 coop_threshold: float = 0.7,
                 defect_threshold: float = 0.5,
                 trust_init: float = 0.5):
        self.id               = agent_id
        self.K                = K
        self.coop_threshold   = coop_threshold
        self.defect_threshold = defect_threshold

        opponents = [j for j in range(n_agents) if j != agent_id]

        # Per-opponent trust (float, derived from belief each round)
        self.trust        = {j: trust_init for j in opponents}
        # Per-opponent history: deque of (my_action, opp_action, norm_reward)
        self.history      = {
            j: deque([(0, 0, 1.0)] * K, maxlen=K) for j in opponents
        }
        self._last_action = {j: 0 for j in opponents}
        self._pending_action: dict = {}

        # Shared belief network + optimizer
        self.model     = BeliefNet(K=K)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()

        # Prediction accuracy tracking (for paper metrics)
        # correct[j] / total[j] = accuracy vs opponent j
        self._correct = {j: 0 for j in opponents}
        self._total   = {j: 0 for j in opponents}

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def act(self, opponent_id: int) -> int:
        t = self.trust[opponent_id]

        # Hysteresis threshold policy (identical to ImprovedTrustAgent)
        if t > self.coop_threshold:
            action = 0
        elif t < self.defect_threshold:
            action = 1
        else:
            action = self._last_action[opponent_id]

        self._pending_action[opponent_id] = action
        self._last_action[opponent_id]    = action
        return action

    def update(self, opponent_id: int, opponent_action: int, reward: float):
        """
        Called after each round with the opponent's actual action.
        Steps:
          1. Build input from current history (BEFORE appending this step)
          2. Forward pass -> p_defect
          3. Update trust = 1 - p_defect
          4. Compute BCE loss against actual opponent action
          5. One gradient step
          6. Append this step to history
          7. Track prediction accuracy
        """
        my_action = self._pending_action.get(opponent_id, 0)

        # --- Step 1: build input from past K steps ---
        x = self._build_input(opponent_id)   # shape (K*3,)

        # --- Steps 2-3: predict and update trust ---
        self.model.train()
        p_defect = self.model(x).squeeze()          # scalar tensor, grad enabled
        self.trust[opponent_id] = float(
            torch.clamp(1.0 - p_defect.detach(), 0.0, 1.0).item()
        )

        # --- Steps 4-5: supervised update on opponent's actual action ---
        target = torch.tensor(float(opponent_action))   # 1=defect, 0=coop
        loss   = self.criterion(p_defect, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # --- Step 6: append to history ---
        self.history[opponent_id].append(
            (my_action, opponent_action, reward / MAX_REWARD)
        )

        # --- Step 7: track accuracy ---
        predicted_defect = int(p_defect.detach().item() > 0.5)
        self._correct[opponent_id] += int(predicted_defect == opponent_action)
        self._total[opponent_id]   += 1

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]

    def prediction_accuracy(self, opponent_id: int) -> float:
        """Fraction of correct defect/coop predictions for this opponent."""
        total = self._total[opponent_id]
        return self._correct[opponent_id] / total if total > 0 else 0.0

    def reset_accuracy(self):
        """Reset accuracy counters (call between train and eval phases)."""
        for j in self._correct:
            self._correct[j] = 0
            self._total[j]   = 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_input(self, opponent_id: int) -> torch.Tensor:
        """
        Flatten last K steps into a (K*3,) tensor:
          [a_self_t-K, ..., a_self_t-1,
           a_opp_t-K,  ..., a_opp_t-1,
           r_t-K,      ..., r_t-1]
        All values in [0, 1].
        """
        hist  = list(self.history[opponent_id])
        a_self = [float(h[0]) for h in hist]
        a_opp  = [float(h[1]) for h in hist]
        rews   = [float(h[2]) for h in hist]
        return torch.tensor(a_self + a_opp + rews, dtype=torch.float32)
