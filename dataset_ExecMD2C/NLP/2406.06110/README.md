# Recurrent Context Compression: Efficiently Expanding the Context Window of LLM (2406.06110)

## Project source

- **Project ID:** `2406.06110`
- **Paper:** Recurrent Context Compression: Efficiently Expanding the Context Window of LLM
- **PDF:** https://arxiv.org/pdf/2406.06110v1.pdf
- **GitHub:** https://github.com/WUHU-G/RCC_Transformer

## Reproduction task

RCC extends a language model's effective context window by recurrently compressing earlier tokens. In `template.py`, complete `GPTNeoXModel.get_rcc_encoder_output_conti_embeddings`, `GPTNeoXModel.forward`, and `GPTNeoXForCausalLM.forward` to implement RCC's recurrent context compression over a long token stream and implement the causal-LM wrapper around the RCC decoder.
