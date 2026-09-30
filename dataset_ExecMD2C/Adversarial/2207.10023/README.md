# Tailoring Self-Supervision for Supervised Learning (2207.10023)

## Project source

- **Project ID:** `2207.10023`
- **Paper:** Tailoring Self-Supervision for Supervised Learning
- **PDF:** https://arxiv.org/pdf/2207.10023v1.pdf
- **GitHub:** https://github.com/wjun0830/localizable-rotation

## Reproduction task

The model combines supervised image classification with a tailored local-rotation self-supervised task. In `template.py`, complete `NormedLinear.forward`, `ResNet_s.forward`, and `rand_bbox` to compute the normalized linear classifier scores used by the long-tailed CIFAR model and sample the square local patch used by the LoRot-I pretext task.
