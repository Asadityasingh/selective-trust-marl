"""
model.py
BeliefNet: predicts P(opponent defects | interaction history).

Architecture: MLP with sigmoid output.
  Input  : K * 3  features — last K steps of (my_action, opp_action, reward)
  Hidden : 64 (ReLU) -> 32 (ReLU)
  Output : 1 logit -> sigmoid -> p_defect in (0, 1)

Training signal: Binary Cross-Entropy against the actual opponent action.
  y = 1  if opponent defected
  y = 0  if opponent cooperated

Trust is derived AFTER prediction:
  trust = 1 - p_defect

This decouples trust from reward entirely — the network is a pure
behavior predictor, not a reward maximizer.
"""
import torch
import torch.nn as nn


class BeliefNet(nn.Module):

    def __init__(self, K: int = 5):
        super().__init__()
        self.K = K
        input_dim = K * 3   # K * (my_action, opp_action, norm_reward)

        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),   # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x     : (..., K*3) float tensor, all values in [0, 1]
        return: (..., 1)   p_defect in (0, 1) via sigmoid
        """
        return torch.sigmoid(self.net(x))
