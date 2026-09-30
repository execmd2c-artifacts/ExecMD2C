# ============================================================
# ground_truth.py - TCVE Core Model Components (Source-faithful)
# Source: TCVE-main
#
# Contains ONLY model architecture definitions and direct dependencies.
# Model code above __main__ is copied from the source repository with only
# one-file consolidation import adjustments.
# No training, dataset, inference pipeline, CLI, or downloaded checkpoint code.
# ============================================================

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import os
import json
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.modeling_utils import ModelMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.attention import CrossAttention, FeedForward, AdaLayerNorm
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange, repeat

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# --- [Original file: tcve/models/attention.py] ---
@dataclass
class Transformer3DModelOutput(BaseOutput):
    sample: torch.FloatTensor


if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None


class Transformer3DModel(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.use_linear_projection = use_linear_projection
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim

        # Define input layers
        self.in_channels = in_channels

        self.norm = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=1e-6, affine=True)
        if use_linear_projection:
            self.proj_in = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_in = nn.Conv2d(in_channels, inner_dim, kernel_size=1, stride=1, padding=0)

        # Define transformers blocks
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
                for d in range(num_layers)
            ]
        )

        # 4. Define output layers
        if use_linear_projection:
            self.proj_out = nn.Linear(in_channels, inner_dim)
        else:
            self.proj_out = nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, return_dict: bool = True):
        # Input
        """
        TODO: Reproduce the source TCVE video transformer wrapper.

        Inputs:
        - hidden_states: (batch, channels, frames, height, width).
        - encoder_hidden_states: optional conditioning tensor aligned by video frames.

        Expected behavior:
        - Preserve the source implementation's 5D video validation, frame flattening, normalization, projection,
          transformer-block loop, residual addition, and restoration to video layout.
        - Honor return_dict exactly as in the original source.

"""
        pass


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        upcast_attention: bool = False,
    ):
        super().__init__()
        self.only_cross_attention = only_cross_attention
        self.use_ada_layer_norm = num_embeds_ada_norm is not None

        # SC-Attn
        self.attn1 = SparseCausalAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
        )
        self.norm1 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

        # Cross-Attn
        if cross_attention_dim is not None:
            self.attn2 = CrossAttention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
            )
        else:
            self.attn2 = None

        if cross_attention_dim is not None:
            self.norm2 = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)
        else:
            self.norm2 = None

        # Feed-forward
        self.ff = FeedForward(dim, dropout=dropout, activation_fn=activation_fn)
        self.norm3 = nn.LayerNorm(dim)

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = AdaLayerNorm(dim, num_embeds_ada_norm) if self.use_ada_layer_norm else nn.LayerNorm(dim)

    def set_use_memory_efficient_attention_xformers(self, use_memory_efficient_attention_xformers: bool):
        if not is_xformers_available():
            print("Here is how to install it")
            raise ModuleNotFoundError(
                "Refer to https://github.com/facebookresearch/xformers for more information on how to install"
                " xformers",
                name="xformers",
            )
        elif not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. xformers' memory efficient attention is only"
                " available for GPU "
            )
        else:
            try:
                # Make sure we can run the memory efficient attention
                _ = xformers.ops.memory_efficient_attention(
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                    torch.randn((1, 2, 40), device="cuda"),
                )
            except Exception as e:
                raise e
            self.attn1._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            if self.attn2 is not None:
                self.attn2._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers
            # self.attn_temp._use_memory_efficient_attention_xformers = use_memory_efficient_attention_xformers

    def forward(self, hidden_states, encoder_hidden_states=None, timestep=None, attention_mask=None, video_length=None):
        # SparseCausal-Attention
        """
        TODO: Reproduce the source TCVE transformer block forward pass.

        Inputs:
        - hidden_states: (batch * frames, spatial_tokens, channels).
        - encoder_hidden_states: optional cross-attention conditioning.
        - video_length: number of frames packed into the leading batch dimension.

        Expected behavior:
        - Preserve the source order of sparse causal attention, cross-attention, feed-forward, and temporal attention,
          including all residual paths and normalization behavior.

"""
        pass


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        """
        TODO: Reproduce the source TCVE sparse causal attention.

        Inputs:
        - hidden_states: (batch * frames, tokens, channels).
        - video_length: number of frames per video item.
        - encoder_hidden_states and attention_mask follow the source CrossAttention contract.

        Expected behavior:
        - Preserve source query/key/value projection behavior and sparse temporal context construction using first-frame
          and previous-frame information.
        - Preserve source attention mask handling and output projection.

"""
        pass


# --- [Original file: tcve/models/resnet.py] ---
def Upsample_1d(dim, dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv1d(dim, dim_out, 3, padding = 1)
    )

class InflatedConv3d(nn.Conv2d):
    def forward(self, x):
        video_length = x.shape[2]

        x = rearrange(x, "b c f h w -> (b f) c h w")
        x = super().forward(x)
        x = rearrange(x, "(b f) c h w -> b c f h w", f=video_length)

        return x


class Upsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, use_conv_transpose=False, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.upsample_TempUnet = Upsample_1d(self.channels, self.out_channels)  # upsample for temporal unet

        conv = None
        if use_conv_transpose:
            raise NotImplementedError
        elif use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, padding=1)
            conv_TempUnet = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(self, hidden_states, hidden_states_TempUnet, output_size=None):
        assert hidden_states.shape[1] == self.channels
        assert hidden_states_TempUnet.shape[1] == self.channels

        if self.use_conv_transpose:
            raise NotImplementedError

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)
            hidden_states_TempUnet = hidden_states_TempUnet.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if output_size is None:
            hidden_states = F.interpolate(hidden_states, scale_factor=[1.0, 2.0, 2.0], mode="nearest")
        else:
            hidden_states = F.interpolate(hidden_states, size=output_size, mode="nearest")



        # If the input is bfloat16, we cast back to bfloat16
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        hidden_states_TempUnet = self.upsample_TempUnet(hidden_states_TempUnet)

        return hidden_states, hidden_states_TempUnet


class Downsample3D(nn.Module):
    def __init__(self, channels, use_conv=False, out_channels=None, padding=1, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = InflatedConv3d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
            conv_TempUnet = nn.Conv1d(self.channels, self.out_channels, 3, stride=stride, padding=padding)
        else:
            raise NotImplementedError

        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
            self.Conv2d_0_TempUnet = conv_TempUnet
            self.conv_TempUnet = conv_TempUnet
        elif name == "Conv2d_0":
            self.conv = conv
            self.conv_TempUnet = conv_TempUnet
        else:
            self.conv = conv
            self.conv_TempUnet = conv_TempUnet

    def forward(self, hidden_states, hidden_states_TempUnet):
        assert hidden_states.shape[1] == self.channels
        assert hidden_states_TempUnet.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            raise NotImplementedError

        assert hidden_states.shape[1] == self.channels
        assert hidden_states_TempUnet.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)
        hidden_states_TempUnet = self.conv_TempUnet(hidden_states_TempUnet)

        return hidden_states, hidden_states_TempUnet


class ResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        temb_channels=512,
        groups=32,
        groups_out=None,
        pre_norm=True,
        eps=1e-6,
        non_linearity="swish",
        time_embedding_norm="default",
        output_scale_factor=1.0,
        use_in_shortcut=None,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.time_embedding_norm = time_embedding_norm
        self.output_scale_factor = output_scale_factor

        if groups_out is None:
            groups_out = groups

        self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)
        self.norm1_TempUnet = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = InflatedConv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv1_TempUnet = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                time_emb_proj_out_channels = out_channels
            elif self.time_embedding_norm == "scale_shift":
                time_emb_proj_out_channels = out_channels * 2
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")

            self.time_emb_proj = torch.nn.Linear(temb_channels, time_emb_proj_out_channels)
        else:
            self.time_emb_proj = None

        self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = InflatedConv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if non_linearity == "swish":
            self.nonlinearity = lambda x: F.silu(x)
        elif non_linearity == "mish":
            self.nonlinearity = Mish()
        elif non_linearity == "silu":
            self.nonlinearity = nn.SiLU()

        self.use_in_shortcut = self.in_channels != self.out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = InflatedConv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
            self.conv_shortcut_TempUnet = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, input_tensor, input_tensor_TempUnet, temb):
        """
        TODO: Reproduce the source TCVE paired spatial/TempUnet residual block.

        Inputs:
        - input_tensor: (batch, channels, frames, height, width).
        - input_tensor_TempUnet: (batch * height * width, channels, frames).
        - temb: optional timestep embedding.

        Expected behavior:
        - Preserve the source spatial branch, temporal TempUnet branch, timestep modulation, optional shortcuts,
          residual combination, and output scaling.

"""
        pass


class Mish(torch.nn.Module):
    def forward(self, hidden_states):
        return hidden_states * torch.tanh(torch.nn.functional.softplus(hidden_states))


# --- [Original file: tcve/models/unet_blocks.py] ---
def get_down_block(
    down_block_type,
    num_layers,
    in_channels,
    out_channels,
    temb_channels,
    add_downsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    downsample_padding=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownBlock3D":
        return DownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif down_block_type == "CrossAttnDownBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnDownBlock3D")
        return CrossAttnDownBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            temb_channels=temb_channels,
            add_downsample=add_downsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{down_block_type} does not exist.")


def get_up_block(
    up_block_type,
    num_layers,
    in_channels,
    out_channels,
    prev_output_channel,
    temb_channels,
    add_upsample,
    resnet_eps,
    resnet_act_fn,
    attn_num_head_channels,
    resnet_groups=None,
    cross_attention_dim=None,
    dual_cross_attention=False,
    use_linear_projection=False,
    only_cross_attention=False,
    upcast_attention=False,
    resnet_time_scale_shift="default",
):
    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpBlock3D":
        return UpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    elif up_block_type == "CrossAttnUpBlock3D":
        if cross_attention_dim is None:
            raise ValueError("cross_attention_dim must be specified for CrossAttnUpBlock3D")
        return CrossAttnUpBlock3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            prev_output_channel=prev_output_channel,
            temb_channels=temb_channels,
            add_upsample=add_upsample,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            cross_attention_dim=cross_attention_dim,
            attn_num_head_channels=attn_num_head_channels,
            dual_cross_attention=dual_cross_attention,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            upcast_attention=upcast_attention,
            resnet_time_scale_shift=resnet_time_scale_shift,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlock3DCrossAttn(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        output_scale_factor=1.0,
        cross_attention_dim=1280,
        dual_cross_attention=False,
        use_linear_projection=False,
        upcast_attention=False,
    ):
        super().__init__()

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)

        # there is always at least one resnet
        resnets = [
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    in_channels // attn_num_head_channels,
                    in_channels=in_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    upcast_attention=upcast_attention,
                )
            )
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, hidden_states_TempUnet, temb=None, encoder_hidden_states=None, attention_mask=None):
        hidden_states, hidden_states_TempUnet = self.resnets[0](hidden_states, hidden_states_TempUnet, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample
            hidden_states, hidden_states_TempUnet = resnet(hidden_states, hidden_states_TempUnet, temb)

        return hidden_states, hidden_states_TempUnet


class CrossAttnDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        downsample_padding=1,
        add_downsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, hidden_states_TempUnet, temb=None, encoder_hidden_states=None, attention_mask=None):
        output_states = ()
        output_states_TempUnet = ()

        for resnet, attn in zip(self.resnets, self.attentions):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states, hidden_states_TempUnet = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet),
                                                                                          hidden_states, hidden_states_TempUnet, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]
            else:
                hidden_states, hidden_states_TempUnet = resnet(hidden_states, hidden_states_TempUnet, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

            output_states += (hidden_states,)
            output_states_TempUnet += (hidden_states_TempUnet,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states, hidden_states_TempUnet = downsampler(hidden_states, hidden_states_TempUnet)

            output_states += (hidden_states,)
            output_states_TempUnet += (hidden_states_TempUnet,)

        return hidden_states, hidden_states_TempUnet, output_states, output_states_TempUnet


class DownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_downsample=True,
        downsample_padding=1,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    Downsample3D(
                        out_channels, use_conv=True, out_channels=out_channels, padding=downsample_padding, name="op"
                    )
                ]
            )
        else:
            self.downsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, hidden_states_TempUnet, temb=None):
        output_states = ()
        output_states_TempUnet = ()

        for resnet in self.resnets:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states, hidden_states_TempUnet = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet),
                                                                                         hidden_states, hidden_states_TempUnet, temb)
            else:
                hidden_states, hidden_states_TempUnet = resnet(hidden_states, hidden_states_TempUnet, temb)

            output_states += (hidden_states,)
            output_states_TempUnet += (hidden_states_TempUnet,)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states, hidden_states_TempUnet = downsampler(hidden_states, hidden_states_TempUnet)

            output_states += (hidden_states,)
            output_states_TempUnet += (hidden_states_TempUnet,)

        return hidden_states, hidden_states_TempUnet, output_states, output_states_TempUnet


class CrossAttnUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        prev_output_channel: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        attn_num_head_channels=1,
        cross_attention_dim=1280,
        output_scale_factor=1.0,
        add_upsample=True,
        dual_cross_attention=False,
        use_linear_projection=False,
        only_cross_attention=False,
        upcast_attention=False,
    ):
        super().__init__()
        resnets = []
        attentions = []

        self.has_cross_attention = True
        self.attn_num_head_channels = attn_num_head_channels

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )
            if dual_cross_attention:
                raise NotImplementedError
            attentions.append(
                Transformer3DModel(
                    attn_num_head_channels,
                    out_channels // attn_num_head_channels,
                    in_channels=out_channels,
                    num_layers=1,
                    cross_attention_dim=cross_attention_dim,
                    norm_num_groups=resnet_groups,
                    use_linear_projection=use_linear_projection,
                    only_cross_attention=only_cross_attention,
                    upcast_attention=upcast_attention,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        hidden_states_TempUnet,
        res_hidden_states_tuple,
        res_hidden_states_TempUnet_tuple,
        temb=None,
        encoder_hidden_states=None,
        upsample_size=None,
        attention_mask=None,
    ):
        for resnet, attn in zip(self.resnets, self.attentions):
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            res_hidden_states_TempUnet = res_hidden_states_TempUnet_tuple[-1]
            res_hidden_states_TempUnet_tuple = res_hidden_states_TempUnet_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states_TempUnet = torch.cat([hidden_states_TempUnet, res_hidden_states_TempUnet], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states, hidden_states_TempUnet = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet), hidden_states, hidden_states_TempUnet, temb)
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(attn, return_dict=False),
                    hidden_states,
                    encoder_hidden_states,
                )[0]
            else:
                hidden_states, hidden_states_TempUnet = resnet(hidden_states, hidden_states_TempUnet, temb)
                hidden_states = attn(hidden_states, encoder_hidden_states=encoder_hidden_states).sample

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states, hidden_states_TempUnet = upsampler(hidden_states, hidden_states_TempUnet, upsample_size)

        return hidden_states, hidden_states_TempUnet


class UpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        prev_output_channel: int,
        out_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor=1.0,
        add_upsample=True,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnet_in_channels = prev_output_channel if i == 0 else out_channels

            resnets.append(
                ResnetBlock3D(
                    in_channels=resnet_in_channels + res_skip_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList([Upsample3D(out_channels, use_conv=True, out_channels=out_channels)])
        else:
            self.upsamplers = None

        self.gradient_checkpointing = False

    def forward(self, hidden_states, hidden_states_TempUnet,
                res_hidden_states_tuple, res_hidden_states_TempUnet_tuple,
                temb=None, upsample_size=None):
        for resnet in self.resnets:
            # pop res hidden states
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            res_hidden_states_TempUnet = res_hidden_states_TempUnet_tuple[-1]
            res_hidden_states_TempUnet_tuple = res_hidden_states_TempUnet_tuple[:-1]
            hidden_states_TempUnet = torch.cat([hidden_states_TempUnet, res_hidden_states_TempUnet], dim=1)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                hidden_states, hidden_states_TempUnet = torch.utils.checkpoint.checkpoint(create_custom_forward(resnet),
                                                                                          hidden_states, hidden_states_TempUnet, temb)
            else:
                hidden_states, hidden_states_TempUnet = resnet(hidden_states, hidden_states_TempUnet, temb)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states, hidden_states_TempUnet = upsampler(hidden_states, hidden_states_TempUnet, upsample_size)

        return hidden_states, hidden_states_TempUnet

class STBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()

        # Temp-Attn
        self.attn_temp = CrossAttention(
            query_dim=dim,
            heads=8,
            dim_head=dim // 8,
            dropout=0.0,
            bias=False,
            upcast_attention=False,
        )
        nn.init.zeros_(self.attn_temp.to_out[0].weight.data)
        self.norm_temp = nn.LayerNorm(dim)

        self.norm = nn.GroupNorm(groups, dim_out)
        self.conv_3d = nn.Conv3d(dim, dim_out, (3, 3, 3), padding=(1, 1, 1))
        self.act = nn.SiLU()
        nn.init.zeros_(self.conv_3d.weight.data)
        nn.init.zeros_(self.conv_3d.bias.data)


    def forward(self, x_sd, x_temp):
        """
        TODO: Reproduce the source TCVE spatial-temporal fusion block.

        Inputs:
        - x_sd: (batch, channels, frames, height, width) spatial diffusion features.
        - x_temp: (batch, channels, frames, height, width) temporal branch features.

        Expected behavior:
        - Preserve temporal-size alignment, temporal attention over per-location frame sequences, fusion into x_sd,
          normalization, 3D convolution, activation, and residual addition.

"""
        pass


# --- [Original file: tcve/models/unet.py] ---
@dataclass
class UNet3DConditionOutput(BaseOutput):
    sample: torch.FloatTensor


class UNet3DConditionModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 4,
        out_channels: int = 4,
        center_input_sample: bool = False,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D",
        ),
        mid_block_type: str = "UNetMidBlock3DCrossAttn",
        up_block_types: Tuple[str] = (
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ),
        only_cross_attention: Union[bool, Tuple[bool]] = False,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        layers_per_block: int = 2,
        downsample_padding: int = 1,
        mid_block_scale_factor: float = 1,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        norm_eps: float = 1e-5,
        cross_attention_dim: int = 1280,
        attention_head_dim: Union[int, Tuple[int]] = 8,
        dual_cross_attention: bool = False,
        use_linear_projection: bool = False,
        class_embed_type: Optional[str] = None,
        num_class_embeds: Optional[int] = None,
        upcast_attention: bool = False,
        resnet_time_scale_shift: str = "default",
    ):
        super().__init__()

        self.sample_size = sample_size
        time_embed_dim = block_out_channels[0] * 4

        # input
        self.conv_in = InflatedConv3d(in_channels, block_out_channels[0], kernel_size=3, padding=(1, 1))
        self.conv_in_TempUnet = nn.Conv1d(in_channels, block_out_channels[0], kernel_size=7, padding=3) # init conv for temporal unet

        # time
        self.time_proj = Timesteps(block_out_channels[0], flip_sin_to_cos, freq_shift)
        timestep_input_dim = block_out_channels[0]

        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        # class embedding
        if class_embed_type is None and num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)
        elif class_embed_type == "timestep":
            self.class_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)
        elif class_embed_type == "identity":
            self.class_embedding = nn.Identity(time_embed_dim, time_embed_dim)
        else:
            self.class_embedding = None

        self.down_blocks = nn.ModuleList([])
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        ## STB for each stage
        self.down_stb_TempUnet = nn.ModuleList([])
        self.mid_stb_TempUnet = None
        self.up_stb_TempUnet = nn.ModuleList([])

        if isinstance(only_cross_attention, bool):
            only_cross_attention = [only_cross_attention] * len(down_block_types)

        if isinstance(attention_head_dim, int):
            attention_head_dim = (attention_head_dim,) * len(down_block_types)

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[i],
                downsample_padding=downsample_padding,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.down_blocks.append(down_block)
            self.down_stb_TempUnet.append(STBlock(output_channel, output_channel))

        # mid
        if mid_block_type == "UNetMidBlock3DCrossAttn":
            self.mid_block = UNetMidBlock3DCrossAttn(
                in_channels=block_out_channels[-1],
                temb_channels=time_embed_dim,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                output_scale_factor=mid_block_scale_factor,
                resnet_time_scale_shift=resnet_time_scale_shift,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=attention_head_dim[-1],
                resnet_groups=norm_num_groups,
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                upcast_attention=upcast_attention,
            )
        else:
            raise ValueError(f"unknown mid_block_type : {mid_block_type}")
        self.mid_stb_TempUnet = STBlock(block_out_channels[-1], block_out_channels[-1])

        # count how many layers upsample the videos
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_attention_head_dim = list(reversed(attention_head_dim))
        only_cross_attention = list(reversed(only_cross_attention))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=norm_eps,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                cross_attention_dim=cross_attention_dim,
                attn_num_head_channels=reversed_attention_head_dim[i],
                dual_cross_attention=dual_cross_attention,
                use_linear_projection=use_linear_projection,
                only_cross_attention=only_cross_attention[i],
                upcast_attention=upcast_attention,
                resnet_time_scale_shift=resnet_time_scale_shift,
            )
            self.up_blocks.append(up_block)
            self.up_stb_TempUnet.append(STBlock(output_channel, output_channel))
            prev_output_channel = output_channel


        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=norm_eps)
        self.conv_act = nn.SiLU()
        self.conv_out = InflatedConv3d(block_out_channels[0], out_channels, kernel_size=3, padding=1)

    def set_attention_slice(self, slice_size):
        r"""
        Enable sliced attention computation.

        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.

        Args:
            slice_size (`str` or `int` or `list(int)`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                `"max"`, maxium amount of memory will be saved by running only one slice at a time. If a number is
                provided, uses as many slices as `attention_head_dim // slice_size`. In this case, `attention_head_dim`
                must be a multiple of `slice_size`.
        """
        sliceable_head_dims = []

        def fn_recursive_retrieve_slicable_dims(module: torch.nn.Module):
            if hasattr(module, "set_attention_slice"):
                sliceable_head_dims.append(module.sliceable_head_dim)

            for child in module.children():
                fn_recursive_retrieve_slicable_dims(child)

        # retrieve number of attention layers
        for module in self.children():
            fn_recursive_retrieve_slicable_dims(module)

        num_slicable_layers = len(sliceable_head_dims)

        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = [dim // 2 for dim in sliceable_head_dims]
        elif slice_size == "max":
            # make smallest slice possible
            slice_size = num_slicable_layers * [1]

        slice_size = num_slicable_layers * [slice_size] if not isinstance(slice_size, list) else slice_size

        if len(slice_size) != len(sliceable_head_dims):
            raise ValueError(
                f"You have provided {len(slice_size)}, but {self.config} has {len(sliceable_head_dims)} different"
                f" attention layers. Make sure to match `len(slice_size)` to be {len(sliceable_head_dims)}."
            )

        for i in range(len(slice_size)):
            size = slice_size[i]
            dim = sliceable_head_dims[i]
            if size is not None and size > dim:
                raise ValueError(f"size {size} has to be smaller or equal to {dim}.")

        # Recursively walk through all the children.
        # Any children which exposes the set_attention_slice method
        # gets the message
        def fn_recursive_set_attention_slice(module: torch.nn.Module, slice_size: List[int]):
            if hasattr(module, "set_attention_slice"):
                module.set_attention_slice(slice_size.pop())

            for child in module.children():
                fn_recursive_set_attention_slice(child, slice_size)

        reversed_slice_size = list(reversed(slice_size))
        for module in self.children():
            fn_recursive_set_attention_slice(module, reversed_slice_size)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (CrossAttnDownBlock3D, DownBlock3D, CrossAttnUpBlock3D, UpBlock3D)):
            module.gradient_checkpointing = value

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        class_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[UNet3DConditionOutput, Tuple]:
        """
        TODO: Reproduce the source TCVE 3D conditional UNet forward pass.

        Inputs:
        - sample: (batch, in_channels, frames, height, width).
        - timestep: scalar or batch-shaped timestep input.
        - encoder_hidden_states: text/cross-attention conditioning.
        - class_labels, attention_mask, and return_dict follow the source signature.

        Expected behavior:
        - Preserve source timestep/class embedding logic, spatial and TempUnet preprocessing, down/mid/up block flow,
          skip handling for both branches, STBlock fusion, and output post-processing.

"""
        pass

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
        """
        TODO: Reproduce the source TCVE 2D-to-3D checkpoint loading method.

        Inputs:
        - pretrained_model_path: checkpoint directory containing config and weights.
        - subfolder: optional checkpoint subdirectory.

        Expected behavior:
        - Preserve source config loading, block-type conversion, model instantiation, weight-file loading, and insertion
          of newly introduced temporal/TempUnet parameters before loading the state dict.

"""
        pass


if __name__ == "__main__":
    torch.manual_seed(7)
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

    print("TCVE model reproduction benchmark checks")

    print("\nTest 1: SparseCausalAttention.forward")
    try:
        attn = SparseCausalAttention(query_dim=8, heads=2, dim_head=4)
        hidden = torch.randn(4, 5, 8)
        out = attn(hidden, video_length=4)
        check("sparse-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "SparseCausalAttention.forward returned None")
        else:
            check("sparse-shape-finite", out.shape == hidden.shape and torch.isfinite(out).all(), str(out.shape))
            shifted = hidden.clone()
            shifted[1] += 3.0
            out_shifted = attn(shifted, video_length=4)
            frame2_delta = (out_shifted[2] - out[2]).abs().mean().item()
            check("sparse-previous-frame-context", frame2_delta > 1e-5, f"delta={frame2_delta:.6f}")
    except Exception as exc:
        skip_checks(3, f"SparseCausalAttention.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 2: BasicTransformerBlock.forward")
    try:
        block = BasicTransformerBlock(dim=8, num_attention_heads=2, attention_head_dim=4, cross_attention_dim=8)
        hidden = torch.randn(4, 5, 8)
        encoder = torch.randn(4, 3, 8)
        out = block(hidden, encoder_hidden_states=encoder, video_length=4)
        check("block-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "BasicTransformerBlock.forward returned None")
        else:
            check("block-shape-finite", out.shape == hidden.shape and torch.isfinite(out).all(), str(out.shape))
            check("block-residual-transform", (out - hidden).abs().mean().item() > 1e-5)
    except Exception as exc:
        skip_checks(3, f"BasicTransformerBlock.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 3: Transformer3DModel.forward")
    try:
        transformer = Transformer3DModel(num_attention_heads=2, attention_head_dim=4, in_channels=8, num_layers=1, norm_num_groups=2, cross_attention_dim=8)
        video = torch.randn(1, 8, 4, 4, 4)
        encoder = torch.randn(1, 3, 8)
        result = transformer(video, encoder_hidden_states=encoder)
        out = result.sample if result is not None else None
        check("transformer-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "Transformer3DModel.forward returned None")
        else:
            check("transformer-shape-finite", out.shape == video.shape and torch.isfinite(out).all(), str(out.shape))
            check("transformer-residual-video-wrapper", (out - video).abs().mean().item() > 1e-5)
    except Exception as exc:
        skip_checks(3, f"Transformer3DModel.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 4: ResnetBlock3D.forward")
    try:
        resnet = ResnetBlock3D(in_channels=8, out_channels=8, temb_channels=32, groups=2)
        spatial = torch.randn(1, 8, 4, 4, 4)
        temporal = torch.randn(16, 8, 4)
        temb = torch.randn(1, 32)
        result = resnet(spatial, temporal, temb)
        check("resnet-output-not-none", result is not None)
        if result is None:
            skip_checks(2, "ResnetBlock3D.forward returned None")
        else:
            out_spatial, out_temporal = result
            check("resnet-shape-finite", out_spatial.shape == spatial.shape and out_temporal.shape == temporal.shape and torch.isfinite(out_spatial).all() and torch.isfinite(out_temporal).all(), f"{out_spatial.shape}, {out_temporal.shape}")
            check("resnet-temporal-branch-updates", (out_temporal - temporal).abs().mean().item() > 1e-5)
    except Exception as exc:
        skip_checks(3, f"ResnetBlock3D.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 5: STBlock.forward")
    try:
        stb = STBlock(dim=8, dim_out=8, groups=8)
        with torch.no_grad():
            stb.conv_3d.weight.zero_()
            stb.conv_3d.weight[:, 0, 1, 1, 1] = 0.1
            stb.conv_3d.bias.zero_()
        x_sd = torch.randn(1, 8, 4, 4, 4)
        x_temp = torch.randn(1, 8, 4, 4, 4)
        out = stb(x_sd, x_temp)
        x_temp_perturbed = x_temp.clone()
        x_temp_perturbed[:, 0, 1, :, :] += 2.0
        out_perturbed = stb(x_sd, x_temp_perturbed)
        check("stblock-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "STBlock.forward returned None")
        else:
            check("stblock-shape-finite", out.shape == x_sd.shape and torch.isfinite(out).all(), str(out.shape))
            check("stblock-temporal-propagation", out_perturbed is not None and (out_perturbed - out).abs().mean().item() > 1e-5)
    except Exception as exc:
        skip_checks(3, f"STBlock.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 6: UNet3DConditionModel.forward")
    small_config = {
        "sample_size": 8,
        "in_channels": 4,
        "out_channels": 4,
        "down_block_types": ("CrossAttnDownBlock3D", "DownBlock3D"),
        "up_block_types": ("UpBlock3D", "CrossAttnUpBlock3D"),
        "block_out_channels": (8, 16),
        "layers_per_block": 1,
        "cross_attention_dim": 8,
        "attention_head_dim": (2, 2),
        "norm_num_groups": 2,
    }
    try:
        model = UNet3DConditionModel(**small_config)
        sample = torch.randn(1, 4, 4, 8, 8)
        timestep = torch.tensor([10])
        encoder = torch.randn(1, 3, 8)
        result = model(sample, timestep, encoder)
        out = result.sample if result is not None else None
        check("unet-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "UNet3DConditionModel.forward returned None")
        else:
            check("unet-shape-finite", out.shape == sample.shape and torch.isfinite(out).all(), str(out.shape))
            perturbed = model(sample + 0.25, timestep, encoder)
            perturbed_out = perturbed.sample if perturbed is not None else None
            check("unet-input-dependent", perturbed_out is not None and (perturbed_out - out).abs().mean().item() > 1e-5)
    except Exception as exc:
        skip_checks(3, f"UNet3DConditionModel.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 7: UNet3DConditionModel.from_pretrained_2d")
    try:
        from diffusers.utils import WEIGHTS_NAME
        pretrained_config = dict(small_config)
        pretrained_config.update(
            {
                "down_block_types": (
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "CrossAttnDownBlock3D",
                    "DownBlock3D",
                ),
                "up_block_types": (
                    "UpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D",
                    "CrossAttnUpBlock3D",
                ),
                "block_out_channels": (8, 16, 16, 16),
                "attention_head_dim": (2, 2, 2, 2),
            }
        )
        base_model = UNet3DConditionModel(**pretrained_config)
        with tempfile.TemporaryDirectory() as tmpdir:
            serializable_config = dict(base_model.config)
            serializable_config["down_block_types"] = list(serializable_config["down_block_types"])
            serializable_config["up_block_types"] = list(serializable_config["up_block_types"])
            serializable_config["block_out_channels"] = list(serializable_config["block_out_channels"])
            serializable_config["attention_head_dim"] = list(serializable_config["attention_head_dim"])
            with open(os.path.join(tmpdir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(serializable_config, f)
            partial_state = {key: value for key, value in base_model.state_dict().items() if "_temp." not in key and "TempUnet" not in key}
            torch.save(partial_state, os.path.join(tmpdir, WEIGHTS_NAME))
            loaded = UNet3DConditionModel.from_pretrained_2d(tmpdir)
        check("pretrained-output-not-none", loaded is not None)
        if loaded is None:
            skip_checks(2, "UNet3DConditionModel.from_pretrained_2d returned None")
        else:
            temporal_keys = [key for key in loaded.state_dict() if "_temp." in key or "TempUnet" in key]
            check("pretrained-type-and-config", isinstance(loaded, UNet3DConditionModel) and loaded.config["in_channels"] == 4)
            check("pretrained-temporal-keys-initialized", len(temporal_keys) > 0)
    except Exception as exc:
        skip_checks(3, f"UNet3DConditionModel.from_pretrained_2d raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"\nResult: {passed}/{total} checks passed")
    if failed != 0:
        raise SystemExit(1)
