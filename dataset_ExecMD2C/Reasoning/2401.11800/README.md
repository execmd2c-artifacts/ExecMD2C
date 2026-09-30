# Revisiting Document-Level Relation Extraction with Context-Guided Link Prediction (2401.11800)

## Project source

- **Project ID:** `2401.11800`
- **Paper:** Revisiting Document-Level Relation Extraction with Context-Guided Link Prediction
- **PDF:** https://arxiv.org/pdf/2401.11800v1.pdf
- **GitHub:** https://github.com/kracr/document-level-relation-extraction

## Reproduction task

The model extracts document-level relations through context-guided graph link prediction. In `template.py`, complete `GraphConvolutionLayer.forward`, `RGCNConv.update`, and the other TODO-marked functions to apply relation-aware graph convolution over document graph nodes and add root node contribution and optional bias after aggregation.
