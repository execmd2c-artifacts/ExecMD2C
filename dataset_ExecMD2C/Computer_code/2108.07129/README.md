# Autoencoders as Tools for Program Synthesis (2108.07129)

## Project source

- **Project ID:** `2108.07129`
- **Paper:** Autoencoders as Tools for Program Synthesis
- **PDF:** https://arxiv.org/pdf/2108.07129v2.pdf
- **GitHub:** https://github.com/sander102907/autoencoder_program_synthesis

## Reproduction task

The system uses tree-structured autoencoders to represent and generate programs for synthesis. In `template.py`, complete `Sampling._filter_top_k`, `TreeLstmDecoderComplete.update_rnn_state`, and the other TODO-marked functions to mask logits outside the top-k candidates and update decoder parent and sibling recurrent states under teacher forcing.
