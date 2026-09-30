# clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation (2003.07311)

## Project source

- **Project ID:** `2003.07311`
- **Paper:** clDice -- A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation
- **PDF:** https://arxiv.org/pdf/2003.07311v7.pdf
- **GitHub:** https://github.com/jocpae/clDice

## Reproduction task

clDice preserves tubular connectivity during image segmentation through a topology-aware loss. In `template.py`, complete `SoftSkeletonize.soft_erode`, `soft_dice_cldice.forward`, and the other TODO-marked functions to compute differentiable morphological erosion for 2D or 3D probability maps and compute the alpha-weighted combination of Dice loss and soft clDice loss.
