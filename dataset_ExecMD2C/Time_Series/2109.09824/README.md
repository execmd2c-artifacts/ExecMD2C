# Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends (2109.09824)

## Project source

- **Project ID:** `2109.09824`
- **Paper:** Well Googled is Half Done: Multimodal Forecasting of New Fashion Product Sales with Image-based Google Trends
- **PDF:** https://arxiv.org/pdf/2109.09824v6.pdf
- **GitHub:** https://github.com/humaticslab/gtm-transformer

## Reproduction task

GTM-Transformer forecasts fashion-product sales using images and Google Trends signals. In `template.py`, complete `FusionNetwork.forward`, `GTrendEmbedder._generate_encoder_mask`, and `GTrendEmbedder.forward` to fuse selected image/text embeddings with calendar features and encode multivariate Google Trends with positional transformer attention.
