# belief_trust_v4 — TODO

## Core idea
Trust = 1 - P(opponent defects | history).
Trained online with BCE on actual opponent actions.
No reward signal used for trust learning.

---

## ✅ Done
- [x] Folder + venv + deps
- [x] `environment.py`  — copied from v3
- [x] `opponents.py`    — copied from v3
- [x] `model.py`        — BeliefNet (MLP, K*3 input, sigmoid output)
- [x] `agent.py`        — BeliefTrustAgent (online BCE, same interface as all agents)
- [x] `train.py`        — warm-up + eval loop, baseline runners, metric averaging
- [x] `plot.py`         — trust dynamics, coop/reward, prediction accuracy, summary bar
- [x] `main.py`         — full pipeline, two experiments, summary tables
- [x] Run and verify

## 🔲 Remaining (toward the paper)
- [ ] Analyse prediction accuracy curves:
      does acc_C rise faster than acc_B? (adversary is more predictable)
- [ ] Visualise what the network learned:
      plot p_defect as a function of recent opponent action sequences
- [ ] Statistical tests: bootstrap CI on trust separation and reward_C
- [ ] Try GRU instead of MLP (captures longer temporal patterns)
- [ ] Extend to 4-5 agent mixed populations
- [ ] Combine BeliefTrust with ImprovedTrust mechanisms (hybrid agent)
- [ ] Write paper section: "Belief-Based Trust Learning"
