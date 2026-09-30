# Feature Intertwiner for Object Detection (1903.11851)

## Project source

- **Project ID:** `1903.11851`
- **Paper:** Feature Intertwiner for Object Detection
- **PDF:** http://arxiv.org/pdf/1903.11851v1.pdf
- **GitHub:** https://github.com/hli2020/feature_intertwiner

## Reproduction task

The model improves object detection by transferring information between small-object and reliable features. In `template.py`, complete `OptTrans.forward`, `Classifier.forward`, and the other TODO-marked functions to compute the feature-intertwiner optimal-transport loss between small and reliable features and run the Mask R-CNN classifier head with optional small-feature intertwining.
