# MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers (2402.02263)

## Project source

- **Project ID:** `2402.02263`
- **Paper:** MixedNUTS: Training-Free Accuracy-Robustness Balance via Nonlinearly Mixed Classifiers
- **PDF:** https://arxiv.org/pdf/2402.02263v5.pdf
- **GitHub:** https://github.com/Bai-YT/MixedNUTS

## Reproduction task

MixedNUTS combines classifiers to balance clean accuracy and adversarial robustness without retraining. In `template.py`, complete `outer_prod`, `NonLinMixedClassifier.forward`, and the other TODO-marked functions to compute an outer product that preserves every axis of both input tensors and run the MixedNUTS classifier with nonlinear base-logit maps and alpha mixing.
