# selective_trust_v2 — TODO

## Goal
Establish that ImprovedTrustAgent is a strong enough baseline to motivate
a *learned* trust module. Three experiments must hold up before we build the NN.

---

## ✅ Done
- [x] Folder + venv setup
- [x] `environment.py` — clean PD core (unchanged from v1)
- [x] `agents.py`
  - [x] ImprovedTrustAgent (carried over + verified)
  - [x] AblationTrustAgent (5 boolean flags, one mechanism off at a time)
  - [x] TitForTatAgent
  - [x] PavlovAgent (Win-Stay-Lose-Shift)
  - [x] CooperativeAgent
  - [x] AdversarialAgent
  - [x] GraduallyCorruptingAgent (new — slow adversarial drift)
- [x] `train.py` — unified runner for all 3 experiments (20 seeds)
- [x] `plot.py` — 4 figures (baseline, ablation bar, corrupting, summary)
- [x] `main.py` — orchestrates all experiments + prints summary tables
- [x] Run and verify all experiments

## 🔲 Remaining (toward the paper)
- [ ] Interpret ablation: which mechanism contributes most to trust separation?
- [ ] Write up Exp 1–3 results as a "motivation" section for the learned trust module
- [ ] Design Neural Trust Module (NTM):
      input  = [last k actions (self+opp), rewards, current trust]
      output = Δtrust
- [ ] Implement NTM and add as 4th agent in Exp 1 & 3
- [ ] Statistical tests (Mann-Whitney U or bootstrap CI) on reward differences
- [ ] Extend to 4–5 agent mixed populations
- [ ] Write README with experiment descriptions and result summary
