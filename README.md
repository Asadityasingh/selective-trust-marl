# Learning to Trust: Selective Cooperation in Mixed Multi-Agent Environments

This repository contains a collection of experiments exploring how trust-like mechanisms can help agents cooperate selectively in repeated Prisoner’s Dilemma environments. The work is based on the research report included in this project and focuses on the idea of selective cooperation in mixed multi-agent systems.

## Overview

The central question is simple but important:

How can an agent cooperate with reliable partners while avoiding exploitation by adversarial ones?

This project studies that problem using several experimental setups, including:

- rule-based trust agents
- Q-learning baselines
- improved trust mechanisms with asymmetric penalties
- REINFORCE-based trust learning
- belief-based neural trust models

The main goal is to show that trust can act as a useful inductive bias for selective cooperation in mixed environments.

## Project Highlights

- Trust-based agents can sustain cooperation where standard Q-learning tends to collapse into mutual defection.
- Asymmetric penalties are critical for making trust robust to periodic defection.
- Reward-driven trust learning can fail because it learns to exploit cooperative agents rather than model trustworthiness.
- Supervised belief-based trust models achieve strong separation between cooperative and adversarial opponents.

## Repository Structure

- [belief_trust_v4](belief_trust_v4) — belief-based trust experiments, baselines, training pipeline, and plots
- [learned_trust_v3](learned_trust_v3) — REINFORCE-based learned trust experiments
- [selective_trust_v2](selective_trust_v2) — earlier selective trust experiments
- [trust_selective_cooperation](trust_selective_cooperation) — alternative trust and cooperation experiments
- [prisoners_dilemma](prisoners_dilemma) — base repeated Prisoner’s Dilemma environment and experiments
- [Learning_to_Trust_Research_Report_1.txt](Learning_to_Trust_Research_Report_1.txt) — full research report

## Requirements

Each experiment folder contains its own requirements file. In general, the project uses:

- Python 3.9+
- NumPy
- Matplotlib
- PyTorch

## Quick Start

1. Clone the repository
2. Open any experiment folder
3. Create a virtual environment
4. Install dependencies
5. Run the main script

Example:

```bash
cd belief_trust_v4
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

You can repeat the same pattern for the other experiment folders.

## Reproducing the Experiments

The main scripts are:

- [belief_trust_v4/main.py](belief_trust_v4/main.py)
- [learned_trust_v3/main.py](learned_trust_v3/main.py)

These scripts train and evaluate the agents, print summary tables, and generate plots saved in each experiment’s plots folder.

## Research Context

This repository is based on the research idea of modeling trust as a learned belief about an opponent’s future behavior rather than as a direct reward-driven signal. The experiments highlight the importance of aligning the training objective with the semantic meaning of trust.

## How to Push This to GitHub

If you want to publish this repository on GitHub, use the following commands:

```bash
git init
git add .
git commit -m "Initial commit: add trust experiments and research report"
git branch -M main
git remote add origin https://github.com/Asadityasingh/selective-trust-marl.git
git push -u origin main
```

## Suggested Commit Message

A clean commit message for the first upload could be:

```text
Initial commit: add selective trust experiments and documentation
```

## License

This project is intended for academic and research use. Add a license file if you plan to share it publicly.
