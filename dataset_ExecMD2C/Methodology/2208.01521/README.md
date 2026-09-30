# DSR -- A dual subspace re-projection network for surface anomaly detection (2208.01521)

## Project source

- **Project ID:** `2208.01521`
- **Paper:** DSR -- A dual subspace re-projection network for surface anomaly detection
- **PDF:** https://arxiv.org/pdf/2208.01521v2.pdf
- **GitHub:** https://github.com/vitjanz/dsr_anomaly_detection

## Reproduction task

DSR detects surface anomalies by projecting features into restricted latent subspaces. In `template.py`, complete `VectorQuantizerEMA.get_quantized`, `UpsamplingModule.forward`, and the other TODO-marked functions to quantize a BCHW feature map with the learned embedding table and refine/upscale anomaly logits using images and a coarse segmentation map.
