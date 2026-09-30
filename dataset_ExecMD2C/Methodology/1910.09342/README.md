# Domain-agnostic Question-Answering with Adversarial Training (1910.09342)

## Project source

- **Project ID:** `1910.09342`
- **Paper:** Domain-agnostic Question-Answering with Adversarial Training
- **PDF:** https://arxiv.org/pdf/1910.09342v2.pdf
- **GitHub:** https://github.com/seanie12/mrqa

## Reproduction task

The model answers questions across domains using adversarially trained representations. In `template.py`, complete `kl_coef`, `DomainQA.get_sep_embedding`, and the other TODO-marked functions to compute the scalar KL annealing coefficient used by adversarial QA and select the SEP-related representation used by concat mode.
