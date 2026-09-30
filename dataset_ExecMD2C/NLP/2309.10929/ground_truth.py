# ============================================================
# ground_truth.py - BTTS_ECAI2023 Core Model Components
# Source: NLP/BTTS_ECAI2023-main
#
# Contains ONLY model architecture components and direct model losses.
# No data module, trainer wrapper, evaluation script, saved-weight I/O, or remote model loading.
# ============================================================

# --- Third-party imports ---
from transformers.models.t5.modeling_t5 import T5Stack, T5PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutput,
                                           Seq2SeqLMOutput)
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map
import warnings
import copy
import torch
import torch.nn as nn


# --- [Original file: loss.py] ---
class BarlowTwinsLoss(nn.Module):
    def __init__(self, batch_size=64, lambda_coeff=5e-3, z_dim=128):
        super().__init__()

        self.z_dim = z_dim
        self.batch_size = batch_size
        self.lambda_coeff = lambda_coeff

    def off_diagonal_ele(self, x):
        # taken from: https://github.com/facebookresearch/barlowtwins/blob/main/main.py
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z1, z2):
        z1 = z1.flatten(start_dim=1)
        z2 = z2.flatten(start_dim=1)

        # N x D, where N is the batch size and D is output dim of projection head
        z1_norm = (z1 - torch.mean(z1, dim=0)) / torch.std(z1, dim=0)
        z2_norm = (z2 - torch.mean(z2, dim=0)) / torch.std(z2, dim=0)

        cross_corr = torch.matmul(z1_norm.T, z2_norm) / self.batch_size

        on_diag = torch.diagonal(cross_corr).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal_ele(cross_corr).pow_(2).sum()

        return on_diag + self.lambda_coeff * off_diag


# --- [Original file: model.py] ---
__HEAD_MASK_WARNING_MSG = """
The input argument `head_mask` was split into two arguments `head_mask` and `decoder_head_mask`. Currently,
`decoder_head_mask` is set to copy `head_mask`, but this feature is deprecated and will be removed in future versions.
If you do not want to use any `decoder_head_mask` now, please set `decoder_head_mask = torch.ones(num_layers,
num_heads)`.
"""


class T5ForConditionalGenerationWithExtractor(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model
        self.lambda_factor = 1
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        extractor_config = copy.deepcopy(config)
        extractor_config.is_decoder = False
        extractor_config.use_cache = False
        extractor_config.is_encoder_decoder = False
        self.extractor = T5Stack(extractor_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block),
                           range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.extractor.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.extractor.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.extractor = self.extractor.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.extractor.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_extractor(self):
        return self.extractor

    def get_decoder(self):
        return self.decoder

    def get_extractor_output(self,
                             input_ids=None,
                             # use cache is simply to a trick to use the generator mixin
                             use_cache_context_ids=None,
                             use_cache_target_examplars_ids=None,
                             use_cache_origin_examplars_ids=None,
                             attention_mask=None,
                             decoder_input_ids=None,
                             decoder_attention_mask=None,
                             head_mask=None,
                             decoder_head_mask=None,
                             cross_attn_head_mask=None,
                             encoder_outputs=None,
                             extractor_outputs=None,
                             past_key_values=None,
                             inputs_embeds=None,
                             context_embeds=None,
                             decoder_inputs_embeds=None,
                             labels=None,
                             use_cache=None,
                             output_attentions=None,
                             output_hidden_states=None,
                             return_dict=None,):
        extractor_hidden = None
        if use_cache_context_ids is None:
            target_styles = ()
            for target_ids in use_cache_target_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=target_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                target_styles += (extractor_hidden,)

            original_styles = ()
            for origin_ids in use_cache_origin_examplars_ids:
                extractor_hidden = self.extractor(
                    input_ids=origin_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )[0]
                original_styles += (extractor_hidden,)

            input_style = self.extractor(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=context_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )[0]
            extractor_hidden = self.lambda_factor * (torch.mean(torch.vstack(
                target_styles), 0) - (torch.mean(torch.vstack(original_styles), 0))) + input_style

        else:
            if extractor_outputs is None:
                extractor_outputs = self.extractor(
                    input_ids=use_cache_context_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=context_embeds,
                    head_mask=head_mask,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=return_dict,
                )
            elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
                extractor_outputs = BaseModelOutput(
                    last_hidden_state=extractor_outputs[0],
                    hidden_states=extractor_outputs[1] if len(
                        extractor_outputs) > 1 else None,
                    attentions=extractor_outputs[2] if len(extractor_outputs) > 2 else None,)
            extractor_hidden = extractor_outputs[0]
        return extractor_hidden

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        use_cache_extractor_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        context_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(
                    encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(
                    encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0] + use_cache_extractor_outputs

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(
                    self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            # loss_extractor_fct = BarlowTwinsLoss(batch_size=64)
            loss_output_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            # loss_extractor = loss_extractor_fct()
            loss = loss_output_fct(
                lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        use_cache_extractor_outputs=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            # "input_ids": input_ids,
            # "use_cache_context_ids": use_cache_context_ids,
            # "use_cache_target_examplars_ids": use_cache_target_examplars_ids,
            # "use_cache_origin_examplars_ids": use_cache_origin_examplars_ids,
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "use_cache_extractor_outputs": use_cache_extractor_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            warnings.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + \
                (reordered_layer_past_states,)
        return reordered_decoder_past


# ============================================================
# __main__: Automated test suite for 3 ablated functions
# ============================================================

if __name__ == "__main__":
    from transformers import T5Config

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

    def make_config():
        return T5Config(
            vocab_size=32,
            d_model=16,
            d_kv=4,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dropout_rate=0.0,
            feed_forward_proj="relu",
            pad_token_id=0,
            eos_token_id=1,
            decoder_start_token_id=0,
            use_cache=False,
            return_dict=True,
        )

    def make_model():
        model = T5ForConditionalGenerationWithExtractor(make_config())
        return model

    print("=" * 70)
    print("BTTS_ECAI2023 - benchmark suite")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1/4: BarlowTwinsLoss.forward
    # ------------------------------------------------------------
    try:
        loss_fn = BarlowTwinsLoss(batch_size=4, lambda_coeff=0.01, z_dim=6)
        z1 = torch.randn(4, 2, 3, requires_grad=True)
        z2 = z1.detach().clone() + 0.05 * torch.randn(4, 2, 3)
        z2.requires_grad_(True)
        loss = loss_fn(z1, z2)
        check("BarlowTwinsLoss output not None", loss is not None)
        if loss is not None:
            check("BarlowTwinsLoss scalar shape", tuple(loss.shape) == (), str(tuple(loss.shape)))
            check("BarlowTwinsLoss finite", torch.isfinite(loss).item())
            check("BarlowTwinsLoss nonnegative", loss.item() >= 0.0)
            loss.backward()
            check("BarlowTwinsLoss gradients reach both views", z1.grad is not None and z1.grad.abs().sum().item() > 0 and z2.grad is not None and z2.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "BarlowTwinsLoss returned None")
    except Exception as exc:
        skip_checks(5, f"BarlowTwinsLoss raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 2/4: get_extractor_output cached context branch
    # ------------------------------------------------------------
    try:
        model = make_model()
        ids = torch.tensor([[2, 3, 4, 5, 1], [6, 7, 8, 9, 1]], dtype=torch.long)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            out = model.get_extractor_output(use_cache_context_ids=ids, attention_mask=mask, return_dict=True)
            direct = model.extractor(input_ids=ids, attention_mask=mask, return_dict=True)[0]
        check("extractor cache output not None", out is not None)
        if out is not None:
            check("extractor cache output shape", tuple(out.shape) == (2, 5, 16), str(tuple(out.shape)))
            check("extractor cache output finite", torch.isfinite(out).all().item())
            check("extractor cache matches direct extractor", torch.allclose(out, direct, atol=1e-5))
            check("extractor cache preserves dtype", out.dtype == direct.dtype, str(out.dtype))
        else:
            skip_checks(4, "cached extractor branch returned None")
    except Exception as exc:
        skip_checks(5, f"cached extractor branch raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 3/4: get_extractor_output exemplar style branch
    # ------------------------------------------------------------
    try:
        model = make_model()
        ids = torch.tensor([[2, 3, 4, 5, 1], [6, 7, 8, 9, 1]], dtype=torch.long)
        target_a = torch.tensor([[3, 4, 5, 6, 1], [7, 8, 9, 10, 1]], dtype=torch.long)
        target_b = torch.tensor([[4, 5, 6, 7, 1], [8, 9, 10, 11, 1]], dtype=torch.long)
        origin_a = torch.tensor([[9, 8, 7, 6, 1], [5, 4, 3, 2, 1]], dtype=torch.long)
        origin_b = torch.tensor([[10, 9, 8, 7, 1], [6, 5, 4, 3, 1]], dtype=torch.long)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            model.lambda_factor = 0
            out_zero = model.get_extractor_output(
                input_ids=ids,
                use_cache_target_examplars_ids=(target_a, target_b),
                use_cache_origin_examplars_ids=(origin_a, origin_b),
                attention_mask=mask,
                return_dict=True,
            )
            direct_input = model.extractor(input_ids=ids, attention_mask=mask, return_dict=True)[0]
            model.lambda_factor = 1
            out_style = model.get_extractor_output(
                input_ids=ids,
                use_cache_target_examplars_ids=(target_a, target_b),
                use_cache_origin_examplars_ids=(origin_a, origin_b),
                attention_mask=mask,
                return_dict=True,
            )
        check("extractor style output not None", out_style is not None)
        if out_style is not None:
            check("extractor style output shape", tuple(out_style.shape) == (2, 5, 16), str(tuple(out_style.shape)))
            check("extractor style output finite", torch.isfinite(out_style).all().item())
            check("extractor lambda zero keeps input style", torch.allclose(out_zero, direct_input, atol=1e-5))
            check("extractor lambda one changes style state", not torch.allclose(out_style, direct_input, atol=1e-5))
        else:
            skip_checks(4, "style extractor branch returned None")
    except Exception as exc:
        skip_checks(5, f"style extractor branch raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 4/4: T5ForConditionalGenerationWithExtractor.forward
    # ------------------------------------------------------------
    try:
        model = make_model()
        ids = torch.tensor([[2, 3, 4, 5, 1], [6, 7, 8, 9, 1]], dtype=torch.long)
        labels = torch.tensor([[3, 4, 5, 1], [7, 8, 9, 1]], dtype=torch.long)
        mask = torch.ones_like(ids)
        with torch.no_grad():
            extractor_state = model.get_extractor_output(use_cache_context_ids=ids, attention_mask=mask, return_dict=True)
            output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache_extractor_outputs=extractor_state, return_dict=True)
            zero_output = model(input_ids=ids, attention_mask=mask, labels=labels, use_cache_extractor_outputs=torch.zeros_like(extractor_state), return_dict=True)
        check("T5 extractor forward output not None", output is not None)
        if output is not None:
            check("T5 extractor forward logits shape", tuple(output.logits.shape) == (2, 4, 32), str(tuple(output.logits.shape)))
            check("T5 extractor forward logits finite", torch.isfinite(output.logits).all().item())
            check("T5 extractor forward loss finite", output.loss is not None and torch.isfinite(output.loss).item())
            check("T5 extractor state affects logits", not torch.allclose(output.logits, zero_output.logits, atol=1e-5))
        else:
            skip_checks(4, "T5 extractor forward returned None")
    except Exception as exc:
        skip_checks(5, f"T5 extractor forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"RESULT: passed={passed} failed={failed}")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
