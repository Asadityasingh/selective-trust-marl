# Trust-Based Selective Cooperation — TODO

## Experiment
Test whether a trust-based agent (A) can distinguish between a cooperative agent (B)
and an adversarial agent (C) in a Repeated Prisoner's Dilemma setting.

---

## ✅ Done
- [x] Create experiment folder `trust_selective_cooperation/`
- [x] Set up Python virtual environment (`venv/`)
- [x] Create `TODO.md`
- [x] `environment.py` — PD environment, pairwise interactions, payoff matrix
- [x] `agents.py` — TrustAgent (A), CooperativeAgent (B), AdversarialAgent (C)
- [x] `agents.py` — ImprovedTrustAgent with asymmetric update, betrayal penalty, hysteresis, smoothing, belief model
- [x] `train.py` — multi-seed training loop, metrics collection, `run_experiment_both()`
- [x] `plot.py` — OLD vs NEW comparison plots (trust, cooperation, reward, heatmap)
- [x] `main.py` — entry point with summary table
- [x] `requirements.txt`
- [x] Fix trust smoothing bug: `0.9*t + delta` → `EMA(t, clip(t+delta))` with alpha=0.1
- [x] Fix delta_coop: `+0.02` → `+0.05` to overcome CooperativeAgent's 10% noise floor

## 🔲 Remaining
- [ ] Run experiment and verify plots in `plots/`
- [ ] Tune trust threshold / epsilon for best separation
- [ ] Try adaptive adversary variant (C defects more when trust_A_C > 0.7)
- [ ] Add more agent types to test scalability
- [ ] Write experiment report / README
