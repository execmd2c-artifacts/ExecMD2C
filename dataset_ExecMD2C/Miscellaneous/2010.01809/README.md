# Long-tailed Recognition by Routing Diverse Distribution-Aware Experts (2010.01809)

## Project source

- **Project ID:** `2010.01809`
- **Paper:** Long-tailed Recognition by Routing Diverse Distribution-Aware Experts
- **PDF:** https://arxiv.org/pdf/2010.01809v4.pdf
- **GitHub:** https://github.com/frank-xwang/RIDE-LongTailRecognition

## Reproduction task

RIDE improves long-tailed image recognition by routing inputs among distribution-aware experts. In `template.py`, complete `ResNet_s._separate_part`, `ResNet_s.forward`, and `RIDELoss.forward` to reproduce one RIDE expert branch after the shared CIFAR ResNet trunk and reproduce the RIDE training loss over ensemble and per-expert logits.
