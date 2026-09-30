# Zero-Shot Text-to-Speech for Text-Based Insertion in Audio Narration (2109.05426)

## Project source

- **Project ID:** `2109.05426`
- **Paper:** Zero-Shot Text-to-Speech for Text-Based Insertion in Audio Narration
- **PDF:** https://arxiv.org/pdf/2109.05426v1.pdf
- **GitHub:** https://github.com/rishikksh20/Zero-Shot-TTS

## Reproduction task

The model synthesizes speech for new text without speaker-specific training examples. In `template.py`, complete `DurationPredictor._forward`, `FFTransformerBlock.forward`, and the other TODO-marked functions to predict token durations in training or inference domain and stack feed-forward Transformer layers with length-derived padding masks.
