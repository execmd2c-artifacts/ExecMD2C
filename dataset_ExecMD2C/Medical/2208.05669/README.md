# PA-Seg: Learning from Point Annotations for 3D Medical Image Segmentation using Contextual Regularization and Cross Knowledge Distillation (2208.05669)

## Project source

- **Project ID:** `2208.05669`
- **Paper:** PA-Seg: Learning from Point Annotations for 3D Medical Image Segmentation using Contextual Regularization and Cross Knowledge Distillation
- **PDF:** https://arxiv.org/pdf/2208.05669v2.pdf
- **GitHub:** https://github.com/hilab-git/pa-seg

## Reproduction task

PA-Seg segments 3D medical images from point annotations using contextual regularization and distillation. In `template.py`, complete `Attention_block.forward`, `KDLoss.forward`, and the other TODO-marked functions to gate an encoder skip tensor with decoder context and compute temperature-scaled KL distillation from teacher logits to student logits.
