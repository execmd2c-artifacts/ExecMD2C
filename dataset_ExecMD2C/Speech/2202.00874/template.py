# ============================================================
# ground_truth.py - HTS-Audio-Transformer Core Model Components
# Source:
#   Speech/HTS-Audio-Transformer-main/utils.py
#   Speech/HTS-Audio-Transformer-main/model/layers.py
#   Speech/HTS-Audio-Transformer-main/model/htsat.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import logging
import pdb
import math
import random
import warnings
import collections.abc
from itertools import repeat
from typing import List

import numpy as np
from numpy.core.fromnumeric import clip, reshape
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn.init import _calculate_fan_in_and_fan_out

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation


# --- [Original file: utils.py] ---
def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)

    Returns:
      out: (batch_size, ...)
    """
    out = (x.transpose(0,-1) * mixup_lambda + torch.flip(x, dims = [0]).transpose(0,-1) * (1 - mixup_lambda)).transpose(0,-1)
    return out


def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the 
    resolution reduction in downsampling of a CNN.
    
    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate

    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


# --- [Original file: model/layers.py] ---
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None, flatten=True, patch_stride = 16):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        padding = ((patch_size[0] - patch_stride[0]) // 2, (patch_size[1] - patch_stride[1]) // 2)

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


# --- [Original file: model/htsat.py] ---
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """[TODO] Reproduce window multi-head self-attention.

        Input:
            x: Tensor with shape (num_windows * batch, tokens_per_window, dim).
            mask: Optional Tensor with shape
                (num_windows, tokens_per_window, tokens_per_window) containing
                zero values for valid pairs and large negative values for
                blocked shifted-window pairs.

        Output:
            Tuple of:
                - attended features with shape
                  (num_windows * batch, tokens_per_window, dim)
                - attention weights with shape
                  (num_windows * batch, num_heads, tokens_per_window,
                  tokens_per_window)

"""
        pass

    def extra_repr(self):
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, norm_before_mlp='ln'):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.norm_before_mlp == 'ln':
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == 'bn':
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        """[TODO] Reproduce one Swin Transformer block for HTS-AT tokens.

        Input:
            x: Tensor with shape (batch, height * width, dim), where
               (height, width) equals self.input_resolution.

        Output:
            Tuple of:
                - block features with shape (batch, height * width, dim)
                - attention weights from the window attention module.

"""
        pass

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 norm_before_mlp='ln'):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer, norm_before_mlp=norm_before_mlp)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        """[TODO] Reproduce one hierarchical Swin stage.

        Input:
            x: Tensor with shape (batch, tokens, dim), where tokens correspond
               to self.input_resolution.

        Output:
            Tuple of:
                - stage features. If downsample is present, shape is
                  (batch, tokens / 4, 2 * dim); otherwise it is
                  (batch, tokens, dim).
                - in evaluation mode, the mean attention map across blocks.

"""
        pass

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class HTSAT_Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(self, spec_size=256, patch_size=4, patch_stride=(4,4), 
                in_chans=1, num_classes=527,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[4, 8, 16, 32],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, 
                 ape=False, patch_norm=True,
                 use_checkpoint=False, norm_before_mlp='ln', config = None, **kwargs):
        super(HTSAT_Swin_Transformer, self).__init__()

        self.config = config
        self.spec_size = spec_size 
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))
        
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = 'hann'
        center = True
        pad_mode = 'reflect'
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32     # Downsampled ratio
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=config.window_size, hop_length=config.hop_size, 
            win_length=config.window_size, window=window, center=center, pad_mode=pad_mode, 
            freeze_parameters=True)
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=config.sample_rate, n_fft=config.window_size, 
            n_mels=config.mel_bins, fmin=config.fmin, fmax=config.fmax, ref=ref, amin=amin, top_db=top_db, 
            freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2) # 2 2
        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)


        # split spctrogram into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size, patch_size=self.patch_size, in_chans=self.in_chans, 
            embed_dim=self.embed_dim, norm_layer=self.norm_layer, patch_stride = patch_stride)

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, self.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(self.embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                    patches_resolution[1] // (2 ** i_layer)),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp)
            self.layers.append(layer)

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        
        if self.config.enable_tscam:
            SF = self.spec_size // (2 ** (len(self.depths) - 1)) // self.patch_stride[0] // self.freq_ratio
            self.tscam_conv = nn.Conv2d(
                in_channels = self.num_features,
                out_channels = self.num_classes,
                kernel_size = (SF,3),
                padding = (0,1)
            )
            self.head = nn.Linear(num_classes, num_classes)
        else:
            self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x):
        """[TODO] Reproduce HTS-AT feature extraction and prediction heads.

        Input:
            x: Tensor with shape (batch, 1, spec_size, spec_size), representing
               the spectrogram image consumed by the Swin-style patch embedder.

        Output:
            Dictionary containing at least:
                - framewise_output: Tensor with shape
                  (batch, frame_steps, num_classes)
                - clipwise_output: Tensor with shape (batch, num_classes)
              When token-semantic CAM is enabled, also include:
                - latent_output: Tensor with shape (batch, num_features)

"""
        pass

    def crop_wav(self, x, crop_size, spe_pos = None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos:crop_pos + crop_size,:]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)
        x = x.permute(0,1,3,2).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2], self.freq_ratio, x.shape[3] // self.freq_ratio)
        # print(x.shape)
        x = x.permute(0,1,3,2,4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x
    
    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert T <= target_T and F <= target_F, "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(x, (target_T, x.shape[3]), mode="bicubic", align_corners=True)
        if F < target_F:
            x = nn.functional.interpolate(x, (x.shape[2], target_F), mode="bicubic", align_corners=True)  
        x = x.permute(0,1,3,2).contiguous() # B C F T
        x = x[:,:,:,cur_pos:cur_pos + self.spec_size]
        x = x.repeat(repeats = (1,1,4,1))
        return x

    def forward(self, x: torch.Tensor, mixup_lambda = None, infer_mode = False):# out_feat_keys: List[str] = None):
        """[TODO] Reproduce the full HTS-AT waveform forward path.

        Input:
            x: Tensor with shape (batch, audio_samples), raw waveform samples.
            mixup_lambda: Optional Tensor with shape (batch,) used only during
                training-time mixup.
            infer_mode: Boolean controlling variable-length inference handling.

        Output:
            Dictionary following the same contract as forward_features:
            framewise_output, clipwise_output, and optionally latent_output.

"""
        pass


# ============================================================
# __main__: Automated test suite for 5 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)

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

    class TinyConfig:
        sample_rate = 1600
        window_size = 64
        hop_size = 32
        mel_bins = 8
        fmin = 20
        fmax = 700
        enable_tscam = True
        htsat_attn_heatmap = False
        loss_type = "clip_bce"
        enable_repeat_mode = False

    class FakeSpectrogram(nn.Module):
        def __init__(self, mel_bins):
            super().__init__()
            self.mel_bins = mel_bins

        def forward(self, waveform):
            return waveform[:, None, :, None].repeat(1, 1, 1, self.mel_bins)

    class FakeLogmel(nn.Module):
        def forward(self, x):
            return x

    def build_tiny_model():
        cfg = TinyConfig()
        model = HTSAT_Swin_Transformer(
            spec_size=32,
            patch_size=2,
            patch_stride=(2, 2),
            in_chans=1,
            num_classes=5,
            embed_dim=8,
            depths=[1, 1],
            num_heads=[2, 4],
            window_size=2,
            mlp_ratio=2.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.0,
            ape=False,
            patch_norm=True,
            use_checkpoint=False,
            config=cfg,
        )
        return model

    print("=" * 70)
    print("HTS-AT: Hierarchical Token-Semantic Audio Transformer")
    print("Automated reproduction benchmark - 5 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/5: WindowAttention.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] WindowAttention.forward - window MSA with relative position bias")
    try:
        module = WindowAttention(dim=8, window_size=(2, 2), num_heads=2, attn_drop=0.0, proj_drop=0.0).to(device)
        x = torch.randn(3, 4, 8, device=device, requires_grad=True)
        out, attn = module(x)
        check("WindowAttention output not None", out is not None and attn is not None)
        if out is not None and attn is not None:
            check("WindowAttention output shape", tuple(out.shape) == (3, 4, 8), f"expected (3, 4, 8), got {tuple(out.shape)}")
            check("WindowAttention attention shape", tuple(attn.shape) == (3, 2, 4, 4), f"expected (3, 2, 4, 4), got {tuple(attn.shape)}")
            check("WindowAttention output finite", torch.isfinite(out).all().item() and torch.isfinite(attn).all().item())
            out.sum().backward()
            check("WindowAttention learns relative position bias", module.relative_position_bias_table.grad is not None and module.relative_position_bias_table.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "WindowAttention.forward returned None output(s)")
    except Exception as e:
        skip_checks(5, f"WindowAttention.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/5: SwinTransformerBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] SwinTransformerBlock.forward - shifted window transformer block")
    try:
        module = SwinTransformerBlock(
            dim=8, input_resolution=(4, 4), num_heads=2,
            window_size=2, shift_size=1, mlp_ratio=2.0,
            drop=0.0, attn_drop=0.0, drop_path=0.0
        ).to(device)
        x = torch.randn(2, 16, 8, device=device, requires_grad=True)
        out, attn = module(x)
        check("SwinTransformerBlock output not None", out is not None and attn is not None)
        if out is not None and attn is not None:
            check("SwinTransformerBlock output shape", tuple(out.shape) == (2, 16, 8), f"expected (2, 16, 8), got {tuple(out.shape)}")
            check("SwinTransformerBlock attention shape", tuple(attn.shape) == (8, 2, 4, 4), f"expected (8, 2, 4, 4), got {tuple(attn.shape)}")
            check("SwinTransformerBlock shifted mask exists", module.attn_mask is not None and tuple(module.attn_mask.shape) == (4, 4, 4))
            (out ** 2).sum().backward()
            check("SwinTransformerBlock routes gradient to attention qkv", module.attn.qkv.weight.grad is not None and module.attn.qkv.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "SwinTransformerBlock.forward returned None output(s)")
    except Exception as e:
        skip_checks(5, f"SwinTransformerBlock.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/5: BasicLayer.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] BasicLayer.forward - Swin stage assembly with attention aggregation")
    try:
        module = BasicLayer(
            dim=8, input_resolution=(4, 4), depth=2, num_heads=2,
            window_size=2, mlp_ratio=2.0, drop=0.0,
            attn_drop=0.0, drop_path=[0.0, 0.0],
            downsample=PatchMerging, use_checkpoint=False
        ).to(device)
        module.eval()
        x = torch.randn(2, 16, 8, device=device, requires_grad=True)
        out, attn = module(x)
        check("BasicLayer output not None", out is not None and attn is not None)
        if out is not None and attn is not None:
            check("BasicLayer downsampled output shape", tuple(out.shape) == (2, 4, 16), f"expected (2, 4, 16), got {tuple(out.shape)}")
            check("BasicLayer aggregated attention shape", tuple(attn.shape) == (8, 2, 4, 4), f"expected (8, 2, 4, 4), got {tuple(attn.shape)}")
            check("BasicLayer output finite", torch.isfinite(out).all().item() and torch.isfinite(attn).all().item())
            (out ** 2).sum().backward()
            check("BasicLayer routes gradient through first block", module.blocks[0].attn.qkv.weight.grad is not None and module.blocks[0].attn.qkv.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "BasicLayer.forward returned None output(s)")
    except Exception as e:
        skip_checks(5, f"BasicLayer.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/5: HTSAT_Swin_Transformer.forward_features
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] HTSAT_Swin_Transformer.forward_features - token-semantic frame/clip heads")
    try:
        model = build_tiny_model().to(device)
        model.eval()
        x = torch.randn(2, 1, 32, 32, device=device, requires_grad=True)
        out = model.forward_features(x)
        check("forward_features output not None", out is not None)
        if out is not None:
            required_keys = {"framewise_output", "clipwise_output", "latent_output"}
            check("forward_features output keys", required_keys.issubset(set(out.keys())))
            check("forward_features clipwise shape", tuple(out["clipwise_output"].shape) == (2, 5), f"expected (2, 5), got {tuple(out['clipwise_output'].shape)}")
            check("forward_features framewise and latent shapes", tuple(out["framewise_output"].shape) == (2, 512, 5) and tuple(out["latent_output"].shape) == (2, 16))
            loss = out["clipwise_output"].sum() + out["framewise_output"].sum()
            loss.backward()
            check("forward_features routes gradient to TSCAM conv", model.tscam_conv.weight.grad is not None and model.tscam_conv.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "HTSAT_Swin_Transformer.forward_features returned None")
    except Exception as e:
        skip_checks(5, f"HTSAT_Swin_Transformer.forward_features raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 5/5: HTSAT_Swin_Transformer.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] HTSAT_Swin_Transformer.forward - waveform to HTS-AT predictions")
    try:
        model = build_tiny_model().to(device)
        model.spectrogram_extractor = FakeSpectrogram(model.config.mel_bins)
        model.logmel_extractor = FakeLogmel()
        model.eval()
        waveform = torch.randn(2, 128, device=device, requires_grad=True)
        out = model(waveform)
        check("HTSAT forward output not None", out is not None)
        if out is not None:
            check("HTSAT forward output keys", {"framewise_output", "clipwise_output", "latent_output"}.issubset(set(out.keys())))
            check("HTSAT forward clipwise shape", tuple(out["clipwise_output"].shape) == (2, 5), f"expected (2, 5), got {tuple(out['clipwise_output'].shape)}")
            check("HTSAT forward framewise shape", tuple(out["framewise_output"].shape) == (2, 512, 5), f"expected (2, 512, 5), got {tuple(out['framewise_output'].shape)}")
            total = out["clipwise_output"].sum() + out["framewise_output"].sum()
            total.backward()
            check("HTSAT forward routes gradient to patch embedding", model.patch_embed.proj.weight.grad is not None and model.patch_embed.proj.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "HTSAT_Swin_Transformer.forward returned None")
    except Exception as e:
        skip_checks(5, f"HTSAT_Swin_Transformer.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Final Score
    # ==============================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some ablated functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
