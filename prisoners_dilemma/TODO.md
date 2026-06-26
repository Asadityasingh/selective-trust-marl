# TODO - Repeated Prisoner's Dilemma (Non-Adversarial)

## Project Setup
- [x] Create project directory structure
- [x] Set up Python virtual environment (`venv`)
- [x] Install dependencies (numpy, matplotlib)
- [x] Create `requirements.txt`

## Implementation
- [x] Environment: Repeated Prisoner's Dilemma (`environment.py`)
- [x] Baseline Agent: Q-learning agent (`agents.py`)
- [x] Trust-Based Agent: Trust-state agent (`agents.py`)
- [x] Training loop with multi-agent simultaneous learning (`train.py`)
- [x] Visualization / Plotting (`plot.py`)
- [x] Main entry point (`main.py`)

## Experiments (NON-ADVERSARIAL)
- [x] Case 1: Baseline Q-learning vs Baseline Q-learning
- [x] Case 2: Trust-based vs Trust-based
- [x] Multi-seed runs for stability
- [x] Compare results and generate plots

## Metrics Tracked
- [x] Average reward per agent per episode
- [x] Cooperation rate per episode
- [x] Trust values over time (both agents)
- [x] Action heatmap (C/D over time for last episode)

## Plots Generated
- [x] Reward vs Episodes (Baseline vs Trust)
- [x] Cooperation Rate vs Episodes
- [x] Trust vs Time (both agents)
- [x] Action heatmap for last episode

## Future Extensions
- [ ] Integrate with PettingZoo MARL framework
- [ ] Add more agent strategies (Tit-for-Tat, Pavlov, etc.)
- [ ] Multi-agent tournament mode
- [ ] Hyperparameter sweep (trust threshold, learning rate, etc.)

---

## How to Run

```bash
cd /Users/adityasingh/Documents/MARL/prisoners_dilemma
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```
