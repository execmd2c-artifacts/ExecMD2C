# UPV at CheckThat! 2021: Mitigating Cultural Differences for Identifying Multilingual Check-worthy Claims (2109.09232)

## Project source

- **Project ID:** `2109.09232`
- **Paper:** UPV at CheckThat! 2021: Mitigating Cultural Differences for Identifying Multilingual Check-worthy Claims
- **PDF:** https://arxiv.org/pdf/2109.09232v1.pdf
- **GitHub:** https://github.com/isspek/cross_lingual_checkworthy_detection

## Reproduction task

The model identifies check-worthy claims across languages while accounting for cultural differences. In `template.py`, complete `SentenceTransformer.mean_pooling`, `SentenceTransformer.forward`, and `SentenceTransformerAdversarial.forward` to compute an attention-mask-aware sentence embedding and run the multitask claim and language classifier.
