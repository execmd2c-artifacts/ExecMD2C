# Latent Execution for Neural Program Synthesis (2107.00101)

## Project source

- **Project ID:** `2107.00101`
- **Paper:** Latent Execution for Neural Program Synthesis
- **PDF:** https://arxiv.org/pdf/2107.00101v2.pdf
- **GitHub:** https://github.com/jungyhuk/latent-execution

## Reproduction task

The model synthesizes programs while tracking their execution in a learned latent space. In `template.py`, complete `CodeGenerator.attention`, `CodeGenerator.prog_exec`, and `CodeGenerator.forward` to compute the additive attention summary used between encoder and decoder streams and run the full neural program synthesizer forward pass with optional latent execution.
