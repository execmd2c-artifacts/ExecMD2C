# Learning to Cut by Watching Movies (2108.04294)

## Project source

- **Project ID:** `2108.04294`
- **Paper:** Learning to Cut by Watching Movies
- **PDF:** https://arxiv.org/pdf/2108.04294v3.pdf
- **GitHub:** https://github.com/PardoAlejo/LearningToCut

## Reproduction task

The model learns where to cut video shots by comparing visual features from movies. In `template.py`, complete `CutsModel._create_sequential_linear_relu_layers`, `CutsModel.forward`, and `CutsLoss.nce_loss` to reproduce the sequential linear/ReLU feature-transform builder and reproduce the normalized contrastive loss used by LearningToCut.
