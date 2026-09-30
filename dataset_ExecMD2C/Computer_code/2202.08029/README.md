# Code Search based on Context-aware Code Translation (2202.08029)

## Project source

- **Project ID:** `2202.08029`
- **Paper:** Code Search based on Context-aware Code Translation
- **PDF:** https://arxiv.org/pdf/2202.08029v1.pdf
- **GitHub:** https://github.com/wssun/trancs

## Reproduction task

The model retrieves code by aligning natural-language queries with context-aware translated code representations. In `template.py`, complete `SeqEncoder_LSTM.forward`, `TranEmbeder.forward`, and the other TODO-marked functions to encode a padded token batch with an LSTM while preserving original batch order and compute the TranCS triplet ranking loss for code search.
