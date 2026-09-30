"""Ground-truth core model components for the MusicMagus benchmark.

This file consolidates the architecture-level zero-shot editing components used
by MusicMagus: Pix2Pix-Zero-style attention processors/losses, UNet processor
preparation, text embedding direction construction, and inversion regularizers.
"""

import inspect
from typing import Any, Callable, Dict, List, Optional, Union

from diffusers.pipelines.audioldm2.modeling_audioldm2 import AudioLDM2ProjectionModel, AudioLDM2UNet2DConditionModel

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F
from transformers import (
    ClapFeatureExtractor,
    ClapModel,
    GPT2Model,
    RobertaTokenizer,
    RobertaTokenizerFast,
    SpeechT5HifiGan,
    T5EncoderModel,
    T5Tokenizer,
    T5TokenizerFast,
)

from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    is_accelerate_available,
    is_accelerate_version,
    is_librosa_available,
    logging,
    replace_example_docstring,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import AudioPipelineOutput, DiffusionPipeline
from diffusers.models.attention_processor import Attention
import re

if is_librosa_available():
    import librosa

CURRENT_LOSS = None
CURRENT_TIMESTEP = None


# --- [Original file: audioldm2/p2p_pipeline.py] ---

def prepare_inputs_for_generation(
    inputs_embeds,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    if past_key_values is not None:
        # only last token for inputs_embeds if past is defined in kwargs
        inputs_embeds = inputs_embeds[:, -1:]

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "use_cache": kwargs.get("use_cache"),
    }


def prepare_unet(unet: AudioLDM2UNet2DConditionModel):
    """Modifies the UNet (`unet`) to perform Pix2Pix Zero optimizations."""
    """
    TODO: Reproduce MusicMagus/Pix2Pix-Zero UNet attention-processor preparation.

    Input:
        unet: AudioLDM2 UNet-like module exposing attn_processors, get_submodule,
            and set_attn_processor.

    Output:
        The same UNet object with every attention processor replaced.

"""
    pass


class Pix2PixZeroL2Loss:
    def __init__(self):
        self.loss = 0.0

    def compute_loss(self, predictions, targets):

        self.loss += ((predictions - targets) ** 2).sum((1, 2)).mean(0)


class Pix2PixZeroMSELoss:
    def __init__(self, seq_len, index, offset=1, weights=1.0, device="cuda"):
        self.loss = 0.0
        self.device = device
        self.seq_len = seq_len
        self.weights = torch.ones(seq_len, device=self.device)
        self.weights[index: index + offset] = weights

    def compute_loss(self, predictions, targets):
        """
        TODO: Reproduce the weighted cross-attention-map MSE accumulator.

        Input:
            predictions: attention map tensor, typically (heads_or_batch, query_tokens,
                key_tokens).
            targets: reference attention map tensor with the same shape.

        Output:
            No return value; update self.loss with a scalar tensor.

"""
        pass


class Pix2PixZeroAttnProcessor:
    """An attention processor class to store the attention weights.
    In Pix2Pix Zero, it happens during computations in the cross-attention blocks."""

    def __init__(self, is_pix2pix_zero=False):
        self.is_pix2pix_zero = is_pix2pix_zero
        if self.is_pix2pix_zero:
            self.reference_cross_attn_map = {}

    def __call__(
        self,
        attn: Attention,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        timestep=None,
        loss=None,
    ):
        """
        TODO: Reproduce the Pix2Pix-Zero attention processor call.

        Input:
            attn: diffusers Attention module with query/key/value projections and
                output projection helpers.
            hidden_states: (batch, query_tokens, channels) current latent tokens.
            encoder_hidden_states: optional (batch, key_tokens, channels) conditioning
                tokens; when absent, the call is self-attention.
            attention_mask: optional attention mask compatible with attn.
            timestep/loss: forwarded metadata; global CURRENT_TIMESTEP and CURRENT_LOSS
                control reference-map storage versus edit-loss computation.

        Output:
            (batch, query_tokens, channels) attended hidden states.

"""
        pass


class AudioLDM2Pipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-audio generation using AudioLDM2.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: ClapModel,
        text_encoder_2: T5EncoderModel,
        projection_model: AudioLDM2ProjectionModel,
        language_model: GPT2Model,
        tokenizer: Union[RobertaTokenizer, RobertaTokenizerFast],
        tokenizer_2: Union[T5Tokenizer, T5TokenizerFast],
        feature_extractor: ClapFeatureExtractor,
        unet: AudioLDM2UNet2DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        vocoder: SpeechT5HifiGan,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            projection_model=projection_model,
            language_model=language_model,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            unet=unet,
            scheduler=scheduler,
            vocoder=vocoder,
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    def generate_language_model(
        self,
        inputs_embeds: torch.Tensor = None,
        max_new_tokens: int = 8,
        **model_kwargs,
    ):
        """

        Generates a sequence of hidden-states from the language model, conditioned on the embedding inputs.

        Parameters:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence used as a prompt for the generation.
            max_new_tokens (`int`):
                Number of new tokens to generate.
            model_kwargs (`Dict[str, Any]`, *optional*):
                Ad hoc parametrization of additional model-specific kwargs that will be forwarded to the `forward`
                function of the model.

        Return:
            `inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                The sequence of generated hidden-states.
        """
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.language_model.config.max_new_tokens
        for _ in range(max_new_tokens):
            # prepare model inputs
            model_inputs = prepare_inputs_for_generation(inputs_embeds, **model_kwargs)

            # forward pass to get next hidden states
            output = self.language_model(**model_inputs, return_dict=True)

            next_hidden_states = output.last_hidden_state

            # Update the model input
            inputs_embeds = torch.cat([inputs_embeds, next_hidden_states[:, -1:, :]], dim=1)

            # Update generated hidden states, model inputs, and length for next step
            model_kwargs = self.language_model._update_model_kwargs_for_generation(output, model_kwargs)

        return inputs_embeds[:, -max_new_tokens:, :]

    def construct_direction(self, embs_source: torch.Tensor, embs_target: torch.Tensor):
        """Constructs the edit direction to steer the image generation process semantically."""
        """
        TODO: Reproduce the semantic edit direction construction.

        Input:
            embs_source: (num_source_prompts, ..., embed_dim) source concept embeddings.
            embs_target: (num_target_prompts, ..., embed_dim) target concept embeddings.

        Output:
            (1, ..., embed_dim) direction tensor for steering generation.

"""
        pass

    def get_epsilon(self, model_output: torch.Tensor, sample: torch.Tensor, timestep: int):
        """
        TODO: Reproduce scheduler-dependent epsilon recovery for inversion guidance.

        Input:
            model_output: latent noise/sample/velocity prediction tensor.
            sample: current noisy latent tensor with the same shape as model_output.
            timestep: scalar scheduler timestep index.

        Output:
            Tensor with the same shape as model_output representing epsilon noise.

"""
        pass

    def auto_corr_loss(self, hidden_states, generator=None):
        """
        TODO: Reproduce the latent-noise autocorrelation regularizer.

        Input:
            hidden_states: (batch, channels, height, width) latent noise tensor.
            generator: optional torch.Generator controlling random roll offsets.

        Output:
            Scalar tensor/accumulator penalizing spatial autocorrelation.

"""
        pass

    def kl_divergence(self, hidden_states):
        mean = hidden_states.mean()
        var = hidden_states.var()
        return var + mean**2 - 1 - torch.log(var + 1e-7)


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

    class DummyModule:
        def __init__(self):
            self.grad_flag = None

        def requires_grad_(self, flag):
            self.grad_flag = flag
            return self

    class DummyUNet:
        def __init__(self):
            self.attn_processors = {
                "block.attn1.processor": None,
                "block.attn2.processor": None,
            }
            self.modules = {
                "block.attn1": DummyModule(),
                "block.attn2": DummyModule(),
            }
            self.set_processors = None

        def get_submodule(self, name):
            return self.modules[name]

        def set_attn_processor(self, processors):
            self.set_processors = processors
            self.attn_processors = processors

    print("[Test 1/6] prepare_unet")
    try:
        unet = DummyUNet()
        returned = prepare_unet(unet)
        check("prepare_unet returns same object", returned is unet)
        check("prepare_unet installs processors", isinstance(unet.set_processors["block.attn2.processor"], Pix2PixZeroAttnProcessor))
        check("prepare_unet marks cross-attn pix2pix", unet.set_processors["block.attn2.processor"].is_pix2pix_zero is True)
        check("prepare_unet marks self-attn non-pix2pix", unet.set_processors["block.attn1.processor"].is_pix2pix_zero is False)
        check("prepare_unet grad flags", unet.modules["block.attn2"].grad_flag is True and unet.modules["block.attn1"].grad_flag is False)
    except Exception as exc:
        skip_checks(5, f"prepare_unet raised {type(exc).__name__}: {exc}")

    print("[Test 2/6] Pix2PixZeroMSELoss.compute_loss")
    try:
        loss_obj = Pix2PixZeroMSELoss(seq_len=4, index=1, offset=2, weights=3.0, device="cpu")
        predictions = torch.ones(2, 3, 4)
        targets = torch.zeros(2, 3, 4)
        loss_obj.compute_loss(predictions, targets)
        expected = torch.tensor(24.0)
        check("MSELoss accumulated tensor not None", loss_obj.loss is not None)
        check("MSELoss scalar shape", getattr(loss_obj.loss, "shape", torch.Size([])) == torch.Size([]))
        check("MSELoss finite", torch.isfinite(loss_obj.loss).item())
        check("MSELoss weighted token span", torch.allclose(loss_obj.loss, expected), f"{loss_obj.loss}")
        check("MSELoss weights initialized", torch.equal(loss_obj.weights, torch.tensor([1.0, 3.0, 3.0, 1.0])))
    except Exception as exc:
        skip_checks(5, f"Pix2PixZeroMSELoss.compute_loss raised {type(exc).__name__}: {exc}")

    class DummyAttention:
        def __init__(self, dim=4):
            self.norm_cross = False
            self.residual_connection = True
            self.rescale_output_factor = 1.0
            self.to_out = [torch.nn.Identity(), torch.nn.Identity()]

        def prepare_attention_mask(self, attention_mask, sequence_length, batch_size):
            return attention_mask

        def to_q(self, hidden_states):
            return hidden_states

        def to_k(self, encoder_hidden_states):
            return encoder_hidden_states

        def to_v(self, encoder_hidden_states):
            return encoder_hidden_states

        def head_to_batch_dim(self, states):
            return states

        def batch_to_head_dim(self, states):
            return states

        def get_attention_scores(self, query, key, attention_mask):
            scores = torch.bmm(query, key.transpose(1, 2)) / max(query.size(-1) ** 0.5, 1.0)
            if attention_mask is not None:
                scores = scores + attention_mask
            return scores.softmax(dim=-1)

    print("[Test 3/6] Pix2PixZeroAttnProcessor.__call__")
    try:
        attn = DummyAttention()
        processor = Pix2PixZeroAttnProcessor(is_pix2pix_zero=True)
        hidden_states = torch.randn(1, 3, 4)
        encoder_hidden_states = torch.randn(1, 5, 4)
        CURRENT_TIMESTEP = torch.tensor(7)
        CURRENT_LOSS = None
        reference_out = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)
        stored = 7 in processor.reference_cross_attn_map
        CURRENT_LOSS = Pix2PixZeroMSELoss(seq_len=5, index=2, offset=1, weights=2.0, device="cpu")
        edited_out = processor(attn, hidden_states, encoder_hidden_states=encoder_hidden_states)
        check("AttnProcessor output not None", reference_out is not None and edited_out is not None)
        if reference_out is not None and edited_out is not None:
            check("AttnProcessor output shape", tuple(reference_out.shape) == (1, 3, 4), str(tuple(reference_out.shape)))
            check("AttnProcessor finite output", torch.isfinite(reference_out).all().item() and torch.isfinite(edited_out).all().item())
            check("AttnProcessor stores reference map", stored and 7 not in processor.reference_cross_attn_map)
            check("AttnProcessor computes edit loss", CURRENT_LOSS.loss is not None and torch.isfinite(CURRENT_LOSS.loss).item())
        else:
            skip_checks(4, "Pix2PixZeroAttnProcessor returned None")
    except Exception as exc:
        skip_checks(5, f"Pix2PixZeroAttnProcessor.__call__ raised {type(exc).__name__}: {exc}")
    finally:
        CURRENT_TIMESTEP, CURRENT_LOSS = None, None

    print("[Test 4/6] AudioLDM2Pipeline.construct_direction")
    try:
        pipe = AudioLDM2Pipeline.__new__(AudioLDM2Pipeline)
        source = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        target = torch.tensor([[2.0, 5.0, 7.0], [4.0, 1.0, 5.0]])
        direction = pipe.construct_direction(source, target)
        check("construct_direction output not None", direction is not None)
        if direction is not None:
            check("construct_direction shape", tuple(direction.shape) == (1, 3), str(tuple(direction.shape)))
            check("construct_direction finite", torch.isfinite(direction).all().item())
            check("construct_direction equals mean delta", torch.allclose(direction, target.mean(0, keepdim=True) - source.mean(0, keepdim=True)))
            check("construct_direction batch axis preserved", direction.size(0) == 1)
        else:
            skip_checks(4, "construct_direction returned None")
    except Exception as exc:
        skip_checks(5, f"construct_direction raised {type(exc).__name__}: {exc}")

    print("[Test 5/6] AudioLDM2Pipeline.get_epsilon")
    try:
        class Config:
            prediction_type = "sample"

        class Scheduler:
            config = Config()
            alphas_cumprod = torch.tensor([0.25, 0.64, 0.81])

        pipe = AudioLDM2Pipeline.__new__(AudioLDM2Pipeline)
        pipe.inverse_scheduler = Scheduler()
        model_output = torch.ones(2, 2)
        sample = torch.full((2, 2), 3.0)
        eps_sample = pipe.get_epsilon(model_output, sample, 1)
        pipe.inverse_scheduler.config.prediction_type = "epsilon"
        eps_direct = pipe.get_epsilon(model_output, sample, 1)
        pipe.inverse_scheduler.config.prediction_type = "v_prediction"
        eps_v = pipe.get_epsilon(model_output, sample, 1)
        check("get_epsilon outputs not None", eps_sample is not None and eps_direct is not None and eps_v is not None)
        if eps_sample is not None and eps_direct is not None and eps_v is not None:
            check("get_epsilon output shape", tuple(eps_sample.shape) == (2, 2), str(tuple(eps_sample.shape)))
            check("get_epsilon finite outputs", torch.isfinite(eps_sample).all().item() and torch.isfinite(eps_v).all().item())
            check("get_epsilon epsilon branch identity", torch.equal(eps_direct, model_output))
            expected_sample = (sample - (0.64 ** 0.5) * model_output) / ((1 - 0.64) ** 0.5)
            check("get_epsilon sample branch formula", torch.allclose(eps_sample, expected_sample))
        else:
            skip_checks(4, "get_epsilon returned None")
    except Exception as exc:
        skip_checks(5, f"get_epsilon raised {type(exc).__name__}: {exc}")

    print("[Test 6/6] AudioLDM2Pipeline regularizers")
    try:
        pipe = AudioLDM2Pipeline.__new__(AudioLDM2Pipeline)
        hidden = torch.randn(2, 2, 6, 6, requires_grad=True)
        generator = torch.Generator().manual_seed(0)
        ac_loss = pipe.auto_corr_loss(hidden, generator=generator)
        kl_loss = pipe.kl_divergence(hidden)
        total = ac_loss + kl_loss
        check("regularizer outputs not None", ac_loss is not None and kl_loss is not None)
        if ac_loss is not None and kl_loss is not None:
            check("regularizer scalar shapes", getattr(ac_loss, "shape", torch.Size([])) == torch.Size([]) and getattr(kl_loss, "shape", torch.Size([])) == torch.Size([]))
            check("regularizer finite outputs", torch.isfinite(ac_loss).item() and torch.isfinite(kl_loss).item())
            total.backward()
            check("regularizer gradients reach hidden", hidden.grad is not None and hidden.grad.abs().sum().item() > 0)
            check("auto_corr_loss nonnegative", ac_loss.item() >= 0.0)
        else:
            skip_checks(4, "regularizers returned None")
    except Exception as exc:
        skip_checks(5, f"regularizers raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
