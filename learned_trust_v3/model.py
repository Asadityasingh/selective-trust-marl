"""
model.py
Neural trust update network: f_theta(history) -> delta_trust

Architecture:
  Input  : K*3 + 1  (K steps of [my_action, opp_action, reward] + current trust)
  Hidden : 64 -> 32, ReLU
  Output : 1 scalar, tanh-bounded -> scaled to [-delta_max, +delta_max]

The network learns a GENERAL trust update rule shared across all opponents.
Same weights are used regardless of who the opponent is — the history
encodes the opponent's behavior, so the network learns to infer trustworthiness
from patterns, not from opponent identity.
"""
import torch
import torch.nn as nn


class TrustUpdateNet(nn.Module):
    """
    MLP that maps interaction history + current trust -> delta_trust.

    Input features (all normalized to [0,1] or [-1,1]):
      - last K self actions       (0=C, 1=D)
      - last K opponent actions   (0=C, 1=D)
      - last K rewards            (normalized by max_reward=5)
      - current trust value       (already in [0,1])

    Output:
      - delta_trust in [-delta_max, +delta_max] via tanh + scale
    """

    def __init__(self, K: int = 5, delta_max: float = 0.3):
        super().__init__()
        self.K = K
        self.delta_max = delta_max
        input_dim = K * 3 + 1   # K*(a_self, a_opp, reward) + trust

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Tanh(),            # output in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, input_dim) or (input_dim,)
        returns: delta_trust scaled to [-delta_max, +delta_max]
        """
        return self.net(x) * self.delta_max
