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

        model_kwargs_ = {
            'decoder_input_ids': None,
            'output_attentions': None,
            'output_hidden_states': None,
            'use_cache': None,
            'attention_mask': None,
            'encoder_outputs': None
        }
        for model_idx in range(len(self.models)):
            model_kwarg = self.models[model_idx]._prepare_encoder_decoder_kwargs_for_generation(inputs_tensor, model_kwargs, model_input_name)
            if model_kwargs_['decoder_input_ids'] is None:
                model_kwargs_['decoder_input_ids'] = model_kwarg['decoder_input_ids']
            if model_kwargs_['output_attentions'] is None:
                model_kwargs_['output_attentions'] = model_kwarg['output_attentions']
            if model_kwargs_['output_hidden_states'] is None:
                model_kwargs_['output_hidden_states'] = model_kwarg['output_hidden_states']
            if model_kwargs_['use_cache'] is None:
                model_kwargs_['use_cache'] = model_kwarg['use_cache']
            if model_kwargs_['attention_mask'] is None:
                model_kwargs_['attention_mask'] = model_kwarg['attention_mask']
            if model_kwargs_['encoder_outputs'] is None:
                model_kwargs_['encoder_outputs'] = [model_kwarg['encoder_outputs']]
            else:
                model_kwargs_['encoder_outputs'].append(model_kwarg['encoder_outputs'])

        return model_kwargs_
    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: Optional[torch.LongTensor] = None,
            encoder_outputs: Optional[ModelOutput] = None,
            **model_kwargs,
    ):
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids = input_ids.index_select(0, expanded_return_idx)

        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            if encoder_outputs is None:
                raise ValueError("If `is_encoder_decoder` is True, make sure that `encoder_outputs` is defined.")
            reordered_encoder_outputs = [encoder_output for encoder_output in encoder_outputs]
            for encoder_output in reordered_encoder_outputs:
                encoder_output["last_hidden_state"] = encoder_output.last_hidden_state.index_select(
                    0, expanded_return_idx.to(encoder_output.last_hidden_state.device)
                )
            model_kwargs["encoder_outputs"] = reordered_encoder_outputs
        return input_ids, model_kwargs
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
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask

        likelihoods = []
        lm_logits = None
        output_past_key_values = []
        decoder_hidden_states = []
        decoder_attentions = []
        cross_attentions = []
        encoder_last_hidden_state = []
        encoder_hidden_states = []
        encoder_attentions = []

        if model_weight is None:
            model_weight = [1.0] * len(self.models)

        for model_idx, model in enumerate(self.models):
            model_output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                head_mask=None if head_mask is None else head_mask[model_idx],
                decoder_head_mask=None if decoder_head_mask is None else decoder_head_mask[model_idx],
                cross_attn_head_mask=None if cross_attn_head_mask is None else cross_attn_head_mask[model_idx],
                encoder_outputs=None if encoder_outputs is None else encoder_outputs[model_idx],
                past_key_values=None if past_key_values is None else past_key_values[model_idx],
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            likelihoods.append(model_output.logits)

            if lm_logits is None:
                lm_logits = (model_weight[model_idx] * model_output.logits)
            else:
                lm_logits = lm_logits + (model_weight[model_idx] * model_output.logits).to(lm_logits.device)

            output_past_key_values.append(model_output.past_key_values)

            decoder_hidden_states.append(model_output.decoder_hidden_states)

            decoder_attentions.append(model_output.decoder_attentions)

            cross_attentions.append(model_output.cross_attentions)

            encoder_last_hidden_state.append(model_output.encoder_last_hidden_state)

            encoder_hidden_states.append(model_output.encoder_hidden_states)

            encoder_attentions.append(model_output.encoder_attentions)


        return Seq2SeqLMSeriesOutput(
            likelihoods=likelihoods,
            logits=lm_logits,
            past_key_values=output_past_key_values,
            decoder_hidden_states=decoder_hidden_states,
            decoder_attentions=decoder_attentions,
            cross_attentions=cross_attentions,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attentions=encoder_attentions,
        )

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
