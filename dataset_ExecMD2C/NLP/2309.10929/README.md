# Specializing Small Language Models towards Complex Style Transfer via Latent Attribute Pre-Training (2309.10929)

## Project source

- **Project ID:** `2309.10929`
- **Paper:** Specializing Small Language Models towards Complex Style Transfer via Latent Attribute Pre-Training
- **PDF:** https://arxiv.org/pdf/2309.10929v1.pdf
- **GitHub:** https://github.com/ruiqixu37/BTTS_ECAI2023

## Reproduction task

The model adapts a small language model for complex style transfer through latent-attribute pretraining. In `template.py`, complete `BarlowTwinsLoss.forward`, `T5ForConditionalGenerationWithExtractor.get_extractor_output`, and `T5ForConditionalGenerationWithExtractor.forward` to reproduce the Barlow Twins latent alignment loss and reproduce the T5 conditional generation forward pass with extractor fusion.
