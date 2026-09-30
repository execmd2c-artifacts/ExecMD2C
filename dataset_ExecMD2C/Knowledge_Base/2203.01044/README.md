# SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs (2203.01044)

## Project source

- **Project ID:** `2203.01044`
- **Paper:** SelfKG: Self-Supervised Entity Alignment in Knowledge Graphs
- **PDF:** https://arxiv.org/pdf/2203.01044v1.pdf
- **GitHub:** https://github.com/THUDM/SelfKG

## Reproduction task

SelfKG aligns entities across knowledge graphs using self-supervised representation learning. In `template.py`, complete `MyEmbedder.contrastive_loss`, `BatchMultiHeadGraphAttention.forward`, and the other TODO-marked functions to compute the SelfKG InfoNCE objective for one positive view and a queue of negatives and apply masked multi-head graph attention over a batched neighborhood.
