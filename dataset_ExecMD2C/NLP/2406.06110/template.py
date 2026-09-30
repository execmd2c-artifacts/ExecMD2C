import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.nn import functional as F
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXLayer

from typing import Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import is_flash_attn_2_available, logging

logger = logging.get_logger(__name__)
if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa


def _make_gpt_neox_layer(config, layer_idx):
    try:
        return GPTNeoXLayer(config, layer_idx=layer_idx)
    except TypeError as exc:
        if "layer_idx" not in str(exc) and "unexpected keyword" not in str(exc):
            raise
        return GPTNeoXLayer(config)


# --- [Original file: model/modeling_mcp2.py] ---
class GPTNeoXModel(GPTNeoXPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embed_in = nn.Embedding(config.vocab_size, config.hidden_size)
        self.emb_dropout = nn.Dropout(config.hidden_dropout)
        self.layers = nn.ModuleList([_make_gpt_neox_layer(config, i) for i in range(config.num_hidden_layers)])
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"

        self.gradient_checkpointing = False
        # nn.SiLU()
        # self.act_concat = nn.ModuleList([nn.SiLU() for _ in range(config.num_hidden_layers)])
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.layers_Equal = False

        rcc_encoder_config = config
        self.rcc_encoder = None

        self.rcc_encoder2gpt_A_matrix = nn.ModuleList(
            [
                nn.Linear(rcc_encoder_config.hidden_size, config.hidden_size, bias=False)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.rcc_encoder2gpt_embed = nn.Linear(rcc_encoder_config.hidden_size, config.hidden_size, bias=False)
        self.rcc_encoder_length = 64 * 512
        self.mem_len = 64
        self.sliding_len = 0

        # self.AEembed = nn.Embedding(1, config.hidden_size)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_in

    def set_input_embeddings(self, value):
        self.embed_in = value

    def get_rcc_encoder_output_conti_embeddings(self, input_ids):  # 64*512):#2048*4):#2048):#
        """
        TODO: Implement RCC's recurrent context compression over a long token stream.

        Input:
            input_ids: LongTensor of shape (batch, total_tokens), where the first
                self.rcc_encoder_length tokens form the context to compress.

        Output:
            A tuple:
            - compressed hidden states of shape
              (batch, self.rcc_encoder_length // self.mem_len, num_hidden_layers + 1, hidden_size)
            - trimmed decoder input ids of shape
              (batch, self.rcc_encoder_length // self.mem_len)

"""
        pass

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        rcc_encoder_last_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        """
        TODO: Implement the RCC-augmented GPTNeoX decoder forward pass.

        Input:
            input_ids: optional LongTensor of shape (batch, sequence).
            inputs_embeds: optional FloatTensor of shape (batch, sequence, hidden_size).
            rcc_encoder_last_hidden_states: optional FloatTensor of shape
                (batch, prefix, num_hidden_layers + 1, hidden_size), containing
                compressed encoder states aligned to the decoder prefix.

        Output:
            BaseModelOutputWithPast or tuple whose last hidden state has shape
            (batch, sequence, hidden_size).

"""
        pass



class GPTNeoXForCausalLM(GPTNeoXPreTrainedModel):
    _tied_weights_keys = ["embed_out.weight"]

    def __init__(self, config):
        super().__init__(config)

        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.embed_out

    def set_output_embeddings(self, new_embeddings):
        self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        rcc_encoder_last_hidden_states: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """
        TODO: Implement the causal-LM wrapper around the RCC decoder.

        Input:
            input_ids: optional LongTensor of shape (batch, sequence).
            rcc_encoder_last_hidden_states: optional FloatTensor of shape
                (batch, prefix, num_hidden_layers + 1, hidden_size).
            labels: optional LongTensor of shape (batch, sequence) for next-token loss.

        Output:
            CausalLMOutputWithPast or tuple. Logits must have shape
            (batch, sequence, vocab_size); loss, when labels are supplied, must be scalar.

"""
        pass

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        input_shape = input_ids.shape
        # cut decoder_input_ids if past is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
        )

        return model_inputs

    def _reorder_cache(self, past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past


if __name__ == "__main__":
    torch.manual_seed(42)
    passed = 0
    failed = 0

    def check(test_name, condition, detail=""):
        global passed, failed
        if bool(condition):
            passed += 1
            print(f"  [{test_name}] PASS")
        else:
            failed += 1
            suffix = f" - {detail}" if detail else ""
            print(f"  [{test_name}] FAIL{suffix}")

    def skip_checks(count, reason):
        global failed
        failed += count
        print(f"  [skipped {count} check(s)] FAIL - {reason}")

    print("=" * 70)
    print("RCC Transformer benchmark: recurrent context compression model code")
    print("=" * 70)

    try:
        from transformers import GPTNeoXConfig
    except Exception as exc:
        GPTNeoXConfig = None
        print(f"  [dependency] transformers GPTNeoXConfig unavailable: {exc}")

    if GPTNeoXConfig is None:
        skip_checks(19, "required transformers dependency is unavailable")
    else:
        config = GPTNeoXConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=4096,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        config._attn_implementation = "eager"

        class TinyEncoder(nn.Module):
            def __init__(self, num_layers, hidden_size):
                super().__init__()
                self.num_layers = num_layers
                self.hidden_size = hidden_size

            def forward(self, input_ids, output_hidden_states=True):
                del output_hidden_states
                batch, seq = input_ids.shape
                base = input_ids.float().unsqueeze(-1).repeat(1, 1, self.hidden_size)
                hidden_states = tuple(base + layer_index * 1000 for layer_index in range(self.num_layers + 1))
                return type("TinyOutput", (), {"hidden_states": hidden_states})()

        print("-" * 70)
        print("[Test 1/4] GPTNeoXModel.get_rcc_encoder_output_conti_embeddings")
        try:
            model = GPTNeoXModel(config)
            model.rcc_encoder = TinyEncoder(config.num_hidden_layers, config.hidden_size)
            model.rcc_encoder_length = 2048
            model.mem_len = 2
            input_ids = torch.arange(2048).unsqueeze(0).long()
            compressed_states, trimmed_ids = model.get_rcc_encoder_output_conti_embeddings(input_ids)
            check("RCC compression output not None", compressed_states is not None)
            if compressed_states is not None:
                check("RCC compressed state shape", compressed_states.shape == (1, 1024, 3, 16), str(tuple(compressed_states.shape)))
                check("RCC trimmed ids shape", trimmed_ids.shape == (1, 1024), str(tuple(trimmed_ids.shape)))
                check("RCC samples every mem_len-th hidden state", torch.equal(compressed_states[0, :4, 0, 0], torch.tensor([1.0, 3.0, 5.0, 7.0])))
                check("RCC keeps the trailing compressed-token ids", torch.equal(trimmed_ids[0, :4], torch.tensor([1024, 1025, 1026, 1027])))
            else:
                skip_checks(4, "RCC compression returned None")
        except Exception as exc:
            skip_checks(5, f"RCC compression raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 2/4] GPTNeoXModel.forward RCC prefix integration")
        try:
            model = GPTNeoXModel(config)
            model.eval()
            batch, seq, prefix = 2, 6, 3
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            rcc_states = torch.randn(batch, prefix, config.num_hidden_layers + 1, config.hidden_size)
            with torch.no_grad():
                output = model(input_ids=input_ids, rcc_encoder_last_hidden_states=rcc_states, output_hidden_states=True)
            check("RCC decoder output not None", output is not None)
            if output is not None:
                check("RCC decoder last hidden shape", output.last_hidden_state.shape == (batch, seq, config.hidden_size), str(tuple(output.last_hidden_state.shape)))
                check("RCC decoder last hidden finite", torch.isfinite(output.last_hidden_state).all().item())
                check("RCC decoder exposes hidden states", output.hidden_states is not None and len(output.hidden_states) == config.num_hidden_layers + 1)
                check("RCC projection modules match layer count", len(model.rcc_encoder2gpt_A_matrix) == config.num_hidden_layers)
            else:
                skip_checks(4, "RCC decoder returned None")
        except Exception as exc:
            skip_checks(5, f"RCC decoder forward raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 3/4] GPTNeoXForCausalLM.forward language modeling wrapper")
        try:
            lm_model = GPTNeoXForCausalLM(config)
            lm_model.eval()
            batch, seq, prefix = 2, 7, 2
            input_ids = torch.randint(0, config.vocab_size, (batch, seq))
            labels = input_ids.clone()
            rcc_states = torch.randn(batch, prefix, config.num_hidden_layers + 1, config.hidden_size)
            with torch.no_grad():
                output = lm_model(input_ids=input_ids, labels=labels, rcc_encoder_last_hidden_states=rcc_states)
            check("LM output not None", output is not None)
            if output is not None:
                check("LM logits shape", output.logits.shape == (batch, seq, config.vocab_size), str(tuple(output.logits.shape)))
                check("LM logits finite", torch.isfinite(output.logits).all().item())
                check("LM loss scalar", output.loss is not None and output.loss.dim() == 0)
                check("LM loss finite", output.loss is not None and torch.isfinite(output.loss).item())
            else:
                skip_checks(4, "LM forward returned None")
        except Exception as exc:
            skip_checks(5, f"LM forward raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 4/4] Generation helpers preserve cache semantics")
        try:
            lm_model = GPTNeoXForCausalLM(config)
            input_ids = torch.tensor([[5, 6, 7, 8]])
            attention_mask = torch.ones_like(input_ids)
            fake_past = tuple(
                (
                    torch.randn(2, config.num_attention_heads, 3, config.hidden_size // config.num_attention_heads),
                    torch.randn(2, config.num_attention_heads, 3, config.hidden_size // config.num_attention_heads),
                )
                for _ in range(config.num_hidden_layers)
            )
            prepared = lm_model.prepare_inputs_for_generation(
                input_ids.repeat(2, 1), past_key_values=fake_past, attention_mask=attention_mask.repeat(2, 1)
            )
            reordered = lm_model._reorder_cache(fake_past, torch.tensor([1, 0]))
            check("prepare_inputs returns model input dict", isinstance(prepared, dict))
            check("prepare_inputs keeps only uncached suffix", prepared["input_ids"].shape == (2, 1), str(tuple(prepared["input_ids"].shape)))
            check("reorder_cache preserves layer count", len(reordered) == config.num_hidden_layers)
            check("reorder_cache reorders batch dimension", torch.equal(reordered[0][0][0], fake_past[0][0][1]))
        except Exception as exc:
            skip_checks(4, f"generation helpers raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
