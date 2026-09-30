# Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in OpenQA (2210.05197)

## Project source

- **Project ID:** `2210.05197`
- **Paper:** Mixed-modality Representation Learning and Pre-training for Joint Table-and-Text Retrieval in OpenQA
- **PDF:** https://arxiv.org/pdf/2210.05197v1.pdf
- **GitHub:** https://github.com/jun-jie-huang/otter

## Reproduction task

OTTeR retrieves relevant tables and text passages jointly for open-domain question answering. In `template.py`, complete `pooling_masked_part`, `RobertaSingleRetrieverThreeCatPool.forward`, and the other TODO-marked functions to pool one masked segment from token hidden states and encode the RoBERTa training batch into query, positive context, and negative context embeddings.
