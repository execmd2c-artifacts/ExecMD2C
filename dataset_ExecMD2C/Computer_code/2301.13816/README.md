# Execution-based Code Generation using Deep Reinforcement Learning (2301.13816)

## Project source

- **Project ID:** `2301.13816`
- **Paper:** Execution-based Code Generation using Deep Reinforcement Learning
- **PDF:** https://arxiv.org/pdf/2301.13816v4.pdf
- **GitHub:** https://github.com/reddy-lab-code-research/PPOCoder

## Reproduction task

The system improves code generation through execution feedback and reinforcement learning. In `template.py`, complete `CodeT5HeadWithValueModel.forward`, `PPOTrainer.loss`, and the other TODO-marked functions to run the CodeT5 policy and attach a scalar value estimate to every decoded token and compute the PPO clipped policy loss, clipped value loss, and diagnostic statistics.
