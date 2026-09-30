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
        assert hidden_states.dim() == 5, f"Expected hidden_states to have ndim=5, but got ndim={hidden_states.dim()}."
        video_length = hidden_states.shape[2]
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
        encoder_hidden_states = repeat(encoder_hidden_states, 'b n c -> (b f) n c', f=video_length)

        batch, channel, height, weight = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        if not self.use_linear_projection:
            hidden_states = self.proj_in(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
        else:
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch, height * weight, inner_dim)
            hidden_states = self.proj_in(hidden_states)

        # Blocks
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep=timestep,
                video_length=video_length
            )

        # Output
        if not self.use_linear_projection:
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )
            hidden_states = self.proj_out(hidden_states)
        else:
            hidden_states = self.proj_out(hidden_states)
            hidden_states = (
                hidden_states.reshape(batch, height, weight, inner_dim).permute(0, 3, 1, 2).contiguous()
            )

        output = hidden_states + residual

        output = rearrange(output, "(b f) c h w -> b c f h w", f=video_length)
        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)


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
        norm_hidden_states = (
            self.norm1(hidden_states, timestep) if self.use_ada_layer_norm else self.norm1(hidden_states)
        )

        if self.only_cross_attention:
            hidden_states = (
                self.attn1(norm_hidden_states, encoder_hidden_states, attention_mask=attention_mask) + hidden_states
            )
        else:
            hidden_states = self.attn1(norm_hidden_states, attention_mask=attention_mask, video_length=video_length) + hidden_states

        if self.attn2 is not None:
            # Cross-Attention
            norm_hidden_states = (
                self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
            )
            hidden_states = (
                self.attn2(
                    norm_hidden_states, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
                )
                + hidden_states
            )

        # Feed-forward
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        # Temporal-Attention
        d = hidden_states.shape[1]
        hidden_states = rearrange(hidden_states, "(b f) d c -> (b d) f c", f=video_length)
        norm_hidden_states = (
            self.norm_temp(hidden_states, timestep) if self.use_ada_layer_norm else self.norm_temp(hidden_states)
        )
        hidden_states = self.attn_temp(norm_hidden_states) + hidden_states
        hidden_states = rearrange(hidden_states, "(b d) f c -> (b f) d c", d=d)

        return hidden_states


class SparseCausalAttention(CrossAttention):
    def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, video_length=None):
        batch_size, sequence_length, _ = hidden_states.shape

        encoder_hidden_states = encoder_hidden_states

        if self.group_norm is not None:
            hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = self.to_q(hidden_states)
        dim = query.shape[-1]
        query = self.reshape_heads_to_batch_dim(query)

        if self.added_kv_proj_dim is not None:
            raise NotImplementedError

        encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        key = self.to_k(encoder_hidden_states)
        value = self.to_v(encoder_hidden_states)

        former_frame_index = torch.arange(video_length) - 1
        former_frame_index[0] = 0

        key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
        key = torch.cat([key[:, [0] * video_length], key[:, former_frame_index]], dim=2)
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
        value = torch.cat([value[:, [0] * video_length], value[:, former_frame_index]], dim=2)
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        if attention_mask is not None:
            if attention_mask.shape[-1] != query.shape[1]:
                target_length = query.shape[1]
                attention_mask = F.pad(attention_mask, (0, target_length), value=0.0)
                attention_mask = attention_mask.repeat_interleave(self.heads, dim=0)

        # attention, what we cannot get enough of
        if self._use_memory_efficient_attention_xformers:
            hidden_states = self._memory_efficient_attention_xformers(query, key, value, attention_mask)
            # Some versions of xformers return output in fp32, cast it back to the dtype of the input
            hidden_states = hidden_states.to(query.dtype)
        else:
            if self._slice_size is None or query.shape[0] // self._slice_size == 1:
                hidden_states = self._attention(query, key, value, attention_mask)
            else:
                hidden_states = self._sliced_attention(query, key, value, sequence_length, dim, attention_mask)

        # linear proj
        hidden_states = self.to_out[0](hidden_states)

        # dropout
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


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
        hidden_states = input_tensor
        hidden_states_TempUnet = input_tensor_TempUnet

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states_TempUnet = self.norm1_TempUnet(hidden_states_TempUnet)
        hidden_states_TempUnet = self.nonlinearity(hidden_states_TempUnet)

        hidden_states = self.conv1(hidden_states)
        hidden_states_TempUnet = self.conv1_TempUnet(hidden_states_TempUnet)

        if temb is not None:
            temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None, None]


        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)
            input_tensor_TempUnet = self.conv_shortcut_TempUnet(input_tensor_TempUnet)

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor
        output_tensor_TempUnet = (input_tensor_TempUnet + hidden_states_TempUnet) / self.output_scale_factor # residual for temporal unet

        return output_tensor, output_tensor_TempUnet


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
        if x_temp.size(2) != x_sd.size(2):
            x_temp = F.interpolate(x_temp, size=x_sd.shape[2:], mode="nearest")

        x_temp = rearrange(x_temp, "b c f h w -> (b h w) f c")  # reshape tensor to fit temp_attn
        norm_x_temp = self.norm_temp(x_temp)
        x_temp = self.attn_temp(norm_x_temp) + x_temp
        x_temp = rearrange(x_temp, "(b h w) f c -> b c f h w", b=x_sd.size(0), c=x_sd.size(1), f=x_sd.size(2), h=x_sd.size(3))

        x = x_sd + 0.1*x_temp  # add temp info to 2d repr
        x = self.norm(x)
        x = self.conv_3d(x)

        return self.act(x) + x_sd


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
        r"""
        Args:
            sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
            timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
            encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

        Returns:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
            [`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.
        """
        # By default samples have to be AT least a multiple of the overall upsampling factor.
        # The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
        # However, the upsampling interpolation output size can be forced to fit any upsampling size
        # on the fly if necessary.
        default_overall_up_factor = 2**self.num_upsamplers

        # upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
        forward_upsample_size = False
        upsample_size = None

        if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
            # logger.info("Forward upsample size to force interpolation output size.")
            forward_upsample_size = True

        # prepare attention_mask
        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # center input if necessary
        if self.config.center_input_sample:
            sample = 2 * sample - 1.0

        # time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb)

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")

            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)

            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        # pre-process
        sample_TempUnet = rearrange(sample, "b c f h w -> (b h w) c f")  # reshape tensor to fit TempUnet
        sample = self.conv_in(sample)
        sample_TempUnet = self.conv_in_TempUnet(sample_TempUnet)

        origin_h = sample.size(3)

        # down
        down_block_res_samples = (sample,)
        down_block_res_samples_TempUnet = (sample_TempUnet,)
        for downsample_block, downsample_stb_TempUnet in zip(self.down_blocks, self.down_stb_TempUnet):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, sample_TempUnet, res_samples, res_samples_TempUnet = downsample_block(
                    hidden_states=sample,
                    hidden_states_TempUnet=sample_TempUnet,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                )
            else:
                sample, sample_TempUnet, res_samples, res_samples_TempUnet = downsample_block(hidden_states=sample,
                                                                                              hidden_states_TempUnet=sample_TempUnet,
                                                                                              temb=emb)

            down_block_res_samples += res_samples
            down_block_res_samples_TempUnet += res_samples_TempUnet

        # mid
        sample, sample_TempUnet = self.mid_block(
            sample, sample_TempUnet, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
        )

        temp_info = rearrange(sample_TempUnet, "(b h w) c f -> b c f h w", b=sample.size(0), h=origin_h)  # reshape tensor into video tensor
        sample = self.mid_stb_TempUnet(sample, temp_info)  # STB Modeling

        # up
        for i, upsample_block in enumerate(self.up_blocks):
            upsample_stb_TempUnet = self.up_stb_TempUnet[i]
            is_final_block = i == len(self.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            res_samples_TempUnet = down_block_res_samples_TempUnet[-len(upsample_block.resnets):]
            down_block_res_samples_TempUnet = down_block_res_samples_TempUnet[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block and forward_upsample_size:
                upsample_size = down_block_res_samples[-1].shape[2:]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample, sample_TempUnet = upsample_block(
                    hidden_states=sample,
                    hidden_states_TempUnet=sample_TempUnet,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    res_hidden_states_TempUnet_tuple=res_samples_TempUnet,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                    attention_mask=attention_mask,
                )
            else:
                sample, sample_TempUnet = upsample_block(
                    hidden_states=sample,
                    hidden_states_TempUnet=sample_TempUnet,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    res_hidden_states_TempUnet_tuple=res_samples_TempUnet,
                    upsample_size=upsample_size
                )
            temp_info = rearrange(sample_TempUnet, "(b h w) c f -> b c f h w", b=sample.size(0), h=sample.size(3))  # reshape tensor into video tensor
            sample = upsample_stb_TempUnet(sample, temp_info)  # STB Modeling


        # post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        if not return_dict:
            return (sample,)

        return UNet3DConditionOutput(sample=sample)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)
        config["_class_name"] = cls.__name__
        config["down_block_types"] = [
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "CrossAttnDownBlock3D",
            "DownBlock3D"
        ]
        config["up_block_types"] = [
            "UpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D",
            "CrossAttnUpBlock3D"
        ]

        from diffusers.utils import WEIGHTS_NAME
        model = cls.from_config(config)
        model_file = os.path.join(pretrained_model_path, WEIGHTS_NAME)
        if not os.path.isfile(model_file):
            raise RuntimeError(f"{model_file} does not exist")
        state_dict = torch.load(model_file, map_location="cpu")
        for k, v in model.state_dict().items():
            if '_temp.' in k or 'TempUnet' in k:
                state_dict.update({k: v})
        model.load_state_dict(state_dict)

        return model


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
