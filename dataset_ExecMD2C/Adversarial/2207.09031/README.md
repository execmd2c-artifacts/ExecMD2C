# Decorrelative Network Architecture for Robust Electrocardiogram Classification (2207.09031)

## Project source

- **Project ID:** `2207.09031`
- **Paper:** Decorrelative Network Architecture for Robust Electrocardiogram Classification
- **PDF:** https://arxiv.org/pdf/2207.09031v4.pdf
- **GitHub:** https://github.com/wang-axis/dna_ecg

## Reproduction task

The network classifies ECG signals while reducing harmful correlations between feature representations. In `template.py`, complete `get_Y_hat`, `CNN.get_features`, and the other TODO-marked functions to estimate the part of one feature matrix that is linearly explainable by another and extract the intermediate convolutional representation used for DVERGE feature distillation.
