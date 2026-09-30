# A Time-Frequency Generative Adversarial based method for Audio Packet Loss Concealment (2307.15611)

## Project source

- **Project ID:** `2307.15611`
- **Paper:** A Time-Frequency Generative Adversarial based method for Audio Packet Loss Concealment
- **PDF:** https://arxiv.org/pdf/2307.15611v1.pdf
- **GitHub:** https://github.com/aircarlo/bin2bin-gan-plc

## Reproduction task

The adversarial model conceals lost audio packets by reconstructing time-frequency content. In `template.py`, complete `Generator.forward`, `PatchDiscriminator.forward`, and `PerPixelDiscriminator.forward` to run the pix2pix-style U-Net generator and run the conditional per-pixel discriminator.
