# learned_trust_v3 — TODO

## Goal
Show that a learned trust module (REINFORCE + MLP) can match or exceed
rule-based trust in selective cooperation, without hand-designed update rules.

---

## ✅ Done
- [x] Folder + venv + deps (torch, numpy, matplotlib)
- [x] `environment.py`     — copied from v2, unchanged
- [x] `model.py`           — TrustUpdateNet (MLP: 16→64→32→1, tanh output)
- [x] `learned_agent.py`   — LearnedTrustAgent (REINFORCE, soft policy, same interface as v2)
- [x] `opponents.py`       — all rule-based agents + opponents from v2
- [x] `train_rl.py`        — REINFORCE loop (300 train + 200 eval per seed, 20 seeds)
- [x] `compare.py`         — rule-based baselines under same eval protocol
- [x] `plot.py`            — training curve, trust dynamics, coop/reward, summary bar
- [x] `main.py`            — full pipeline
- [x] Run and verify

## 🔲 Remaining (toward the paper)
- [ ] Interpret what the network learned:
      - Visualise delta_trust as a function of history patterns
      - Does it rediscover asymmetric update? betrayal penalty?
- [ ] Statistical tests: bootstrap CI or Mann-Whitney U on reward_C differences
- [ ] Hyperparameter sensitivity: K (history window), delta_max, LR
- [ ] Try GRU instead of MLP (richer temporal modeling)
- [ ] Extend to 4-5 agent mixed populations
- [ ] Write paper section: "Learned Trust Module"
