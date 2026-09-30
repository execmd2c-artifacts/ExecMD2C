# Open-Domain Text Evaluation via Contrastive Distribution Methods (2306.11879)

## Project source

- **Project ID:** `2306.11879`
- **Paper:** Open-Domain Text Evaluation via Contrastive Distribution Methods
- **PDF:** https://arxiv.org/pdf/2306.11879v4.pdf
- **GitHub:** https://github.com/pluslabnlp/cdm

## Reproduction task

The model evaluates open-domain text by contrasting distributions of generated and reference content. In `template.py`, complete `T5ForContrastiveDistributionModeling._prepare_encoder_decoder_kwargs_for_generation`, `T5ForContrastiveDistributionModeling._expand_inputs_for_generation`, and `T5ForContrastiveDistributionModeling.forward` to reproduce CDM multi-model encoder preparation for generation and reproduce the contrastive distribution modeling forward pass.
