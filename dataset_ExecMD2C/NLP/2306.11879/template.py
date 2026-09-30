# ============================================================
# ground_truth.py - CDM Core Model Components
# Source: NLP/CDM-main
#
# Contains ONLY model architecture components from models/modeling_t5cdm.py.
# No trainer, dataset, evaluation script, saved-weight I/O, or remote model loading.
# ============================================================

# --- Third-party imports ---
from typing import Optional, Tuple, Union, List

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import PreTrainedModel
from transformers.models.t5.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import ModelOutput, dataclass


# --- [Original file: models/modeling_t5cdm.py] ---
@dataclass
class Seq2SeqLMSeriesOutput(ModelOutput):
    likelihoods: Optional[List[torch.FloatTensor]] = None
    logits: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[List[Tuple[Tuple[torch.FloatTensor]]]] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


class T5ForContrastiveDistributionModeling(PreTrainedModel):

    def __init__(self, config: List[T5Config]):
        super(T5ForContrastiveDistributionModeling, self).__init__(config[0])
        self.models = nn.ModuleList([
            T5ForConditionalGeneration(config=config_i) for config_i in config
        ])

    def _prepare_encoder_decoder_kwargs_for_generation(self, inputs_tensor, model_kwargs, model_input_name):
        """
        TODO: Reproduce CDM multi-model encoder preparation for generation.

        Input:
            inputs_tensor: (batch, src_len) encoder input ids or embeddings used by each component model.
            model_kwargs: dictionary containing generation kwargs such as attention_mask.
            model_input_name: string name of the model input consumed by the wrapped T5 models.

        Output:
            Dictionary with shared generation kwargs and encoder_outputs as a list of length num_models,
            where each encoder output has last_hidden_state shaped (batch, src_len, d_model).

"""
        pass
    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[ModelOutput] = None,
            **model_kwargs,
    ):
        """
        TODO: Reproduce CDM beam/return expansion for multiple encoder outputs.

        Input:
            input_ids: (batch, seq_len) decoder/input ids to duplicate.
            expand_size: number of expanded copies per original batch item.
            is_encoder_decoder: whether encoder outputs must also be expanded.
            attention_mask: optional (batch, seq_len) mask to expand with input ids.
            encoder_outputs: optional list of ModelOutput objects, one per component model, each containing
                last_hidden_state shaped (batch, src_len, d_model).
            model_kwargs: optional extra tensor kwargs such as token_type_ids.

        Output:
            expanded_input_ids: (batch * expand_size, seq_len).
            expanded_model_kwargs: dictionary whose masks and every component encoder last_hidden_state are expanded
            along the batch dimension in the same order as input ids.

"""
        pass
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        model_weight: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMSeriesOutput]:
        """
        TODO: Reproduce the contrastive distribution modeling forward pass.

        Input:
            input_ids: optional (batch, src_len) source token ids shared by all component T5 models.
            attention_mask: optional (batch, src_len) source mask.
            model_weight: optional list/tensor of scalar weights with length num_models.
            decoder_input_ids: optional (batch, tgt_len) decoder token ids.
            encoder_outputs: optional list of per-model encoder outputs aligned with self.models.
            past_key_values: optional list of per-model decoder caches aligned with self.models.
            labels: optional (batch, tgt_len) target token ids.

        Output:
            Seq2SeqLMSeriesOutput with:
                likelihoods: list of num_models tensors, each (batch, tgt_len, vocab_size).
                logits: weighted summed logits tensor of shape (batch, tgt_len, vocab_size).
                model state lists collected from every component model.

"""
        pass

    def prepare_inputs_for_generation(
        self,
        input_ids,
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
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
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
            return past

        return [self.models[i]._reorder_cache(past[i], beam_idx) for i in range(len(self.models))]


# ============================================================
# __main__: Automated test suite for 3 ablated functions
# ============================================================

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

    def make_config():
        return T5Config(
            vocab_size=37,
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
        configs = [make_config(), make_config()]
        model = T5ForContrastiveDistributionModeling(configs)
        for child in model.models:
            original_prepare = child._prepare_encoder_decoder_kwargs_for_generation

            def prepare_compat(inputs_tensor, model_kwargs, model_input_name, original_prepare=original_prepare, child=child):
                try:
                    prepared = original_prepare(inputs_tensor, model_kwargs, model_input_name)
                except TypeError as exc:
                    if "generation_config" not in str(exc):
                        raise
                    prepared = original_prepare(inputs_tensor, model_kwargs, model_input_name, child.generation_config)
                for key in ["decoder_input_ids", "output_attentions", "output_hidden_states", "use_cache", "attention_mask", "encoder_outputs"]:
                    prepared.setdefault(key, None)
                return prepared

            child._prepare_encoder_decoder_kwargs_for_generation = prepare_compat
        return model

    print("=" * 70)
    print("CDM - benchmark suite")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1/3: _prepare_encoder_decoder_kwargs_for_generation
    # ------------------------------------------------------------
    try:
        model = make_model()
        ids = torch.tensor([[2, 3, 4, 1], [5, 6, 7, 1]], dtype=torch.long)
        mask = torch.ones_like(ids)
        prepared = model._prepare_encoder_decoder_kwargs_for_generation(
            ids,
            {"attention_mask": mask},
            "input_ids",
        )
        check("prepare kwargs output not None", prepared is not None)
        if prepared is not None:
            check("prepare kwargs has two encoder outputs", isinstance(prepared.get("encoder_outputs"), list) and len(prepared["encoder_outputs"]) == 2)
            check("prepare kwargs encoder output shapes", all(tuple(out.last_hidden_state.shape) == (2, 4, 16) for out in prepared["encoder_outputs"]))
            check("prepare kwargs attention mask propagated", tuple(prepared["attention_mask"].shape) == (2, 4))
            check("prepare kwargs finite encoder states", all(torch.isfinite(out.last_hidden_state).all().item() for out in prepared["encoder_outputs"]))
        else:
            skip_checks(4, "prepare kwargs returned None")
    except Exception as exc:
        skip_checks(5, f"prepare kwargs raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 2/3: _expand_inputs_for_generation
    # ------------------------------------------------------------
    try:
        ids = torch.tensor([[2, 3, 4, 1], [5, 6, 7, 1]], dtype=torch.long)
        mask = torch.ones_like(ids)
        enc1 = ModelOutput(last_hidden_state=torch.randn(2, 4, 16))
        enc2 = ModelOutput(last_hidden_state=torch.randn(2, 4, 16))
        expanded_ids, expanded_kwargs = T5ForContrastiveDistributionModeling._expand_inputs_for_generation(
            ids,
            expand_size=3,
            is_encoder_decoder=True,
            attention_mask=mask,
            encoder_outputs=[enc1, enc2],
        )
        check("expand generation output not None", expanded_ids is not None and expanded_kwargs is not None)
        if expanded_ids is not None and expanded_kwargs is not None:
            check("expand generation input shape", tuple(expanded_ids.shape) == (6, 4), str(tuple(expanded_ids.shape)))
            check("expand generation attention mask shape", tuple(expanded_kwargs["attention_mask"].shape) == (6, 4), str(tuple(expanded_kwargs["attention_mask"].shape)))
            check("expand generation preserves two encoder outputs", len(expanded_kwargs["encoder_outputs"]) == 2)
            check("expand generation encoder states expanded", all(tuple(out.last_hidden_state.shape) == (6, 4, 16) for out in expanded_kwargs["encoder_outputs"]))
        else:
            skip_checks(4, "expand generation returned None")
    except Exception as exc:
        skip_checks(5, f"expand generation raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 3/3: T5ForContrastiveDistributionModeling.forward
    # ------------------------------------------------------------
    try:
        model = make_model()
        ids = torch.tensor([[2, 3, 4, 1], [5, 6, 7, 1]], dtype=torch.long)
        labels = torch.tensor([[3, 4, 1], [6, 7, 1]], dtype=torch.long)
        mask = torch.ones_like(ids)
        output_default = model(input_ids=ids, attention_mask=mask, labels=labels, return_dict=True)
        output_weighted = model(input_ids=ids, attention_mask=mask, labels=labels, model_weight=[1.0, 0.0], return_dict=True)
        check("CDM forward output not None", output_default is not None)
        if output_default is not None:
            check("CDM forward logits shape", tuple(output_default.logits.shape) == (2, 3, 37), str(tuple(output_default.logits.shape)))
            check("CDM forward stores per-model likelihoods", isinstance(output_default.likelihoods, list) and len(output_default.likelihoods) == 2)
            check("CDM forward likelihood shapes", all(tuple(x.shape) == (2, 3, 37) for x in output_default.likelihoods))
            check("CDM forward weighted branch selects first model", torch.allclose(output_weighted.logits, output_weighted.likelihoods[0], atol=1e-5))
        else:
            skip_checks(4, "CDM forward returned None")
    except Exception as exc:
        skip_checks(5, f"CDM forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"RESULT: passed={passed} failed={failed}")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
