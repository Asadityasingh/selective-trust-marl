"""
learned_agent.py
LearnedTrustAgent: REINFORCE agent whose trust update rule is a learned MLP.

Gradient flow design:
  - During an episode, we store the raw history inputs and rewards at each step.
  - In compute_loss(), we RECOMPUTE the trust trajectory using the model
    (with gradients enabled), reconstruct the action log-probs from the
    recomputed trust values, and apply REINFORCE.
  - This is the standard "recomputation" approach for REINFORCE through
    non-differentiable state (trust is stored as a float during rollout
    for speed, then recomputed with grad for the update).

Interface is identical to all v2 agents: act(), update(), get_trust().
"""
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from model import TrustUpdateNet

MAX_REWARD  = 5.0
SCALE       = 10.0    # sharpness of soft policy sigmoid


class LearnedTrustAgent:

    def __init__(self, agent_id: int, n_agents: int,
                 K: int = 5,
                 coop_threshold: float = 0.7,
                 defect_threshold: float = 0.5,
                 epsilon: float = 0.05,
                 trust_init: float = 0.5,
                 delta_max: float = 0.3,
                 training: bool = True):
        self.id              = agent_id
        self.K               = K
        self.coop_threshold  = coop_threshold
        self.defect_threshold= defect_threshold
        self.epsilon         = epsilon
        self.trust_init      = trust_init
        self.training        = training

        opponents = [j for j in range(n_agents) if j != agent_id]

        # Runtime trust (float) — used for act() decisions
        self.trust        = {j: trust_init for j in opponents}
        # History buffer: (my_action, opp_action, norm_reward)
        self.history      = {j: deque([(0, 0, MAX_REWARD / MAX_REWARD)] * K, maxlen=K)
                             for j in opponents}
        self._last_action = {j: 0 for j in opponents}
        self._pending_action: dict = {}

        self.model = TrustUpdateNet(K=K, delta_max=delta_max)

        # Episode buffers — one entry per round across all interactions
        # Each entry: (input_vec, action_taken, reward, opponent_id)
        self._episode_steps: list = []

        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Agent interface
    # ------------------------------------------------------------------

    def act(self, opponent_id: int) -> int:
        t = self.trust[opponent_id]

        # Epsilon-greedy exploration
        if self.training and self._rng.random() < self.epsilon:
            action = int(self._rng.integers(0, 2))
            self._pending_action[opponent_id] = action
            self._last_action[opponent_id]    = action
            return action

        if self.training:
            # Soft threshold policy — action sampled stochastically
            coop_prob = torch.sigmoid(
                torch.tensor(SCALE * (t - self.coop_threshold), dtype=torch.float32)
            )
            action = 0 if self._rng.random() < coop_prob.item() else 1
        else:
            # Hard threshold policy for evaluation
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
        my_action = self._pending_action.get(opponent_id, 0)

        # Snapshot current input BEFORE updating history (history = past)
        x = self._build_input(opponent_id)   # numpy array, no grad

        # Run model (no grad during rollout — we recompute in compute_loss)
        with torch.no_grad():
            delta = self.model(
                torch.tensor(x, dtype=torch.float32)
            ).squeeze().item()

        # Update runtime trust
        t = self.trust[opponent_id]
        self.trust[opponent_id] = float(np.clip(t + delta, 0.0, 1.0))

        # Append step to history
        self.history[opponent_id].append(
            (my_action, opponent_action, reward / MAX_REWARD)
        )

        # Store step data for compute_loss recomputation
        if self.training:
            self._episode_steps.append({
                "x":           x,                    # input at this step
                "action":      my_action,
                "reward":      reward,
                "trust_before": t,                   # trust value used in act()
                "opponent_id": opponent_id,
            })

    def get_trust(self, opponent_id: int) -> float:
        return self.trust[opponent_id]

    # ------------------------------------------------------------------
    # REINFORCE
    # ------------------------------------------------------------------

    def compute_loss(self, gamma: float = 0.99) -> torch.Tensor:
        """
        Recompute trust trajectory WITH gradients, reconstruct log-probs,
        apply REINFORCE with mean-reward baseline + entropy bonus.
        """
        if not self._episode_steps:
            return torch.tensor(0.0)

        steps = self._episode_steps

        # --- Recompute trust trajectory with grad ---
        # We track per-opponent trust tensors so gradients flow through
        # the sequence of model calls.
        trust_tensors: dict = {}   # opponent_id -> current trust tensor

        log_probs = []
        rewards   = []

        for step in steps:
            opp_id = step["opponent_id"]
            x      = torch.tensor(step["x"], dtype=torch.float32)

            # Initialise trust tensor for this opponent if first time seen
            if opp_id not in trust_tensors:
                trust_tensors[opp_id] = torch.tensor(
                    self.trust_init, dtype=torch.float32
                )

            t_tensor = trust_tensors[opp_id]

            # Forward pass WITH grad
            delta = self.model(x).squeeze()
            new_trust = torch.clamp(t_tensor + delta, 0.0, 1.0)
            trust_tensors[opp_id] = new_trust   # carry forward (with grad)

            # Reconstruct log-prob of the action that was taken
            # using the trust value BEFORE this update (= t_tensor)
            coop_prob = torch.sigmoid(SCALE * (t_tensor - self.coop_threshold))
            # P(cooperate=0) = coop_prob, P(defect=1) = 1 - coop_prob
            if step["action"] == 0:
                lp = torch.log(coop_prob + 1e-8)
            else:
                lp = torch.log(1.0 - coop_prob + 1e-8)

            log_probs.append(lp)
            rewards.append(step["reward"])

        # --- Discounted returns ---
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        log_probs_t = torch.stack(log_probs)

        # --- REINFORCE + entropy bonus ---
        pg_loss = -(log_probs_t * returns_t.detach()).mean()
        entropy = -(log_probs_t * torch.exp(log_probs_t.detach())).mean()
        return pg_loss - 0.01 * entropy

    def clear_buffers(self):
        self._episode_steps.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_input(self, opponent_id: int) -> np.ndarray:
        hist   = list(self.history[opponent_id])
        a_self = [h[0] for h in hist]
        a_opp  = [h[1] for h in hist]
        rews   = [h[2] for h in hist]
        trust  = [self.trust[opponent_id]]
        return np.array(a_self + a_opp + rews + trust, dtype=np.float32)

    def set_eval(self):
        self.training = False
        self.model.eval()

    def set_train(self):
        self.training = True
        self.model.train()

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, weights_only=True))
