# Estimating the electrical power output of industrial devices with end-to-end time-series classification in the presence of label noise (2105.00349)

## Project source

- **Project ID:** `2105.00349`
- **Paper:** Estimating the electrical power output of industrial devices with end-to-end time-series classification in the presence of label noise
- **PDF:** https://arxiv.org/pdf/2105.00349v2.pdf
- **GitHub:** https://github.com/Castel44/SREA

## Reproduction task

SREA classifies industrial time series robustly despite noisy labels. In `template.py`, complete `AEandClass.forward`, `create_hard_labels`, and the other TODO-marked functions to produce reconstruction and classification outputs from one shared latent embedding and form SREA hard pseudo-labels from history, cluster membership, and noisy observations.
