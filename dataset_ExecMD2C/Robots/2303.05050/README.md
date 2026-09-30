# Lifelong-MonoDepth: Lifelong Learning for Multi-Domain Monocular Metric Depth Estimation (2303.05050)

## Project source

- **Project ID:** `2303.05050`
- **Paper:** Lifelong-MonoDepth: Lifelong Learning for Multi-Domain Monocular Metric Depth Estimation
- **PDF:** https://arxiv.org/pdf/2303.05050v3.pdf
- **GitHub:** https://github.com/freeformrobotics/lifelong-monodepth

## Reproduction task

The model estimates metric depth across changing visual domains through lifelong learning. In `template.py`, complete `MFF.forward`, `Uncertainty_depth.forward`, and `model_ll.forward` to fuse four encoder feature scales into one shared depth representation and run shared encoding/fusion once and dispatch the result to all domain-specific heads.
