# Hear The Flow: Optical Flow-Based Self-Supervised Visual Sound Source Localization (2211.03019)

## Project source

- **Project ID:** `2211.03019`
- **Paper:** Hear The Flow: Optical Flow-Based Self-Supervised Visual Sound Source Localization
- **PDF:** https://arxiv.org/pdf/2211.03019v1.pdf
- **GitHub:** https://github.com/denfed/heartheflow

## Reproduction task

Hear The Flow localizes visual sound sources using audio, video, and optical flow without manual labels. In `template.py`, complete `HearTheFlowVSSLModel.lvs_loss`, `HearTheFlowVSSLModel.forward`, and `Self_Attn.forward` to compute the self-supervised localization-via-similarity loss and compute cross-attention from visual features and flow features.
