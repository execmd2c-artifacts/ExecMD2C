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
        # input_shape = input_ids.size()
        # batch_size, seq_length = input_shape
        # seq_length = seq_length//2
        # 每隔512个token，将rcc_encoder的last_ssm_state映射到llama的hidden_states
        rcc_encoder_max_input = 2048
        rcc_encoder_length = self.rcc_encoder_length
        mem_len = self.mem_len
        # sliding_len = self.sliding_len
        assert rcc_encoder_max_input % mem_len == 0
        assert rcc_encoder_length % mem_len == 0, rcc_encoder_length
        rcc_encoder_compress_num = rcc_encoder_max_input // mem_len

        for id_index in range(0, rcc_encoder_length, rcc_encoder_max_input):
            if id_index == 0:
                output = self.rcc_encoder(
                    input_ids[:, id_index : id_index + rcc_encoder_max_input], output_hidden_states=True
                )
            else:
                output = self.rcc_encoder(
                    input_ids[:, id_index : id_index + rcc_encoder_max_input], output_hidden_states=True
                )
            if id_index == 0:
                rcc_encoder_last_hidden_states = torch.cat(
                    [
                        i[:, [j * mem_len + mem_len - 1 for j in range(rcc_encoder_compress_num)], :].unsqueeze(1)
                        for i in output.hidden_states
                    ],
                    dim=1,
                )  # .detach()
            else:
                rcc_encoder_last_hidden_states_temp = torch.cat(
                    [
                        i[:, [j * mem_len + mem_len - 1 for j in range(rcc_encoder_compress_num)], :].unsqueeze(1)
                        for i in output.hidden_states
                    ],
                    dim=1,
                )  # .detach()
                rcc_encoder_last_hidden_states = torch.cat(
                    (rcc_encoder_last_hidden_states, rcc_encoder_last_hidden_states_temp), dim=2
                )  # .detach()#batch,sqlen,layer,hidden
        # rcc_encoder_length = 64*1024
        # input_ids_length = 64*1024+1024
        input_ids = input_ids[:, rcc_encoder_length - rcc_encoder_length // mem_len :]
        rcc_encoder_last_hidden_states = rcc_encoder_last_hidden_states.permute(0, 2, 1, 3)  # b,squence,Layer,h
        assert rcc_encoder_last_hidden_states.shape[1] == rcc_encoder_length // mem_len

        return rcc_encoder_last_hidden_states, input_ids

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
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.
            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * self.config.num_hidden_layers)
        else:
            past_length = past_key_values[0][0].size(-2)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            if self._use_flash_attention_2:
                attention_mask = attention_mask if 0 in attention_mask else None
            else:
                # We create a 3D attention mask from a 2D tensor mask.
                # Sizes are [batch_size, 1, 1, to_seq_length]
                # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
                # this attention mask is more simple than the triangular masking of causal attention
                # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
                attention_mask = attention_mask[:, None, None, :]

                # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
                # masked positions, this operation will create a tensor which is 0.0 for
                # positions we want to attend and the dtype's smallest value for masked positions.
                # Since we are adding it to the raw scores before the softmax, this is
                # effectively the same as removing these entirely.
                attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
                attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        if inputs_embeds is None:
            inputs_embeds = self.embed_in(input_ids)

        hidden_states = self.emb_dropout(inputs_embeds)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_attentions = () if output_attentions else None
        all_hidden_states = () if output_hidden_states else None

        if rcc_encoder_last_hidden_states is not None:
            prefix_len = rcc_encoder_last_hidden_states.shape[1]
            # print("hidden_states[:,:prefix_len,:]",hidden_states[:,:prefix_len,:].shape)
            # print("rcc_encoder_last_hidden_states[:,:,0,:].shape",rcc_encoder_last_hidden_states[:,:,0,:].shape)
            hidden_states[:, :prefix_len, :] = self.rcc_encoder2gpt_embed(rcc_encoder_last_hidden_states[:, :, 0, :])

        for i, (layer, layer_past) in enumerate(zip(self.layers, past_key_values)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                outputs = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    head_mask[i],
                    use_cache,
                    None,
                    output_attentions,
                )
            else:
                outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    head_mask=head_mask[i],
                    layer_past=layer_past,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )
            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)
            if output_attentions:
                all_attentions = all_attentions + (outputs[2 if use_cache else 1],)

            if rcc_encoder_last_hidden_states is not None:
                # print("i",i)
                # print("hidden_states",hidden_states.shape)
                prefix_len = rcc_encoder_last_hidden_states.shape[1]
                # print("prefix_len",prefix_len)
                # print("rcc_encoder_last_hidden_states.data[:,:,2*i+1:2*i+3,:]",rcc_encoder_last_hidden_states.data[:,:,2*i+1:2*i+3,:].shape)
                # rcc_encoder_states_layer_mean = rcc_encoder_last_hidden_states[:,:,2*i+1:2*i+3,:].mean(dim=2) ##batch,sqlen,hidden

                if self.layers_Equal:
                    rcc_encoder_states_layer_mean = rcc_encoder_last_hidden_states[:, :, 2 * i + 1 : 2 * i + 3, :].mean(
                        dim=2
                    )  ##batch,sqlen,hidden
                else:
                    rcc_encoder_states_layer_mean = rcc_encoder_last_hidden_states[:, :, i + 1, :]  ##batch,sqlen,hidden

                rcc_encoder_states_layer_mean = self.rcc_encoder2gpt_A_matrix[i](rcc_encoder_states_layer_mean)
                hidden_states[:, :prefix_len, :] = hidden_states[:, :prefix_len, :] + rcc_encoder_states_layer_mean

        hidden_states = self.final_layer_norm(hidden_states)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, presents, all_hidden_states, all_attentions] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


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
        r"""
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of shape
            `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`. The two additional tensors are
            only required when the model is used as a decoder in a Sequence to Sequence model.
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the left-to-right language modeling loss (next word prediction). Indices should be in
            `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are
            ignored (masked), the loss is only computed for the tokens with labels n `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print("input_ids",input_ids.shape)
        if rcc_encoder_last_hidden_states is None and self.training:
            rcc_encoder_last_hidden_states, input_ids = self.gpt_neox.get_rcc_encoder_output_conti_embeddings(input_ids)

        # print("input_ids2",input_ids.shape)
        # print(labels.shape)
        # print(rcc_encoder_last_hidden_states.shape)

        outputs = self.gpt_neox(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            rcc_encoder_last_hidden_states=rcc_encoder_last_hidden_states,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shift_logits = lm_logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

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
