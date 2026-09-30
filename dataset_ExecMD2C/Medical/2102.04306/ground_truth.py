"""
ground_truth.py for TransUNet core model components.

Source-consolidated from:
- networks/vit_seg_configs.py
- networks/vit_seg_modeling_resnet_skip.py
- networks/vit_seg_modeling.py

Only the TransUNet model architecture, hybrid ResNetV2 stem, Transformer
encoder, decoder, segmentation head, model configuration helpers, and
pretrained model-specific loading logic are included. Non-model orchestration,
CLI wrappers, and metric utilities are intentionally excluded.
"""

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
from collections import OrderedDict
from os.path import join as pjoin

import ml_collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


# --- [Original file: networks/vit_seg_configs.py] ---
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1

    config.classifier = 'seg'
    config.representation_size = None
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_16.npz'
    config.patch_size = 16

    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_testing():
    """Returns a minimal configuration for testing."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 1
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None
    return config


def get_r50_b16_config():
    """Returns the Resnet50 + ViT-B/16 configuration."""
    config = get_b16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.n_skip = 3
    config.activation = 'softmax'

    return config


def get_b32_config():
    """Returns the ViT-B/32 configuration."""
    config = get_b16_config()
    config.patches.size = (32, 32)
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-B_32.npz'
    return config


def get_l16_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 1024
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 4096
    config.transformer.num_heads = 16
    config.transformer.num_layers = 24
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.representation_size = None

    # custom
    config.classifier = 'seg'
    config.resnet_pretrained_path = None
    config.pretrained_path = '../model/vit_checkpoint/imagenet21k/ViT-L_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_r50_l16_config():
    """Returns the Resnet50 + ViT-L/16 configuration. customized """
    config = get_l16_config()
    config.patches.grid = (16, 16)
    config.resnet = ml_collections.ConfigDict()
    config.resnet.num_layers = (3, 4, 9)
    config.resnet.width_factor = 1

    config.classifier = 'seg'
    config.resnet_pretrained_path = '../model/vit_checkpoint/imagenet21k/R50+ViT-B_16.npz'
    config.decoder_channels = (256, 128, 64, 16)
    config.skip_channels = [512, 256, 64, 16]
    config.n_classes = 2
    config.activation = 'softmax'
    return config


def get_l32_config():
    """Returns the ViT-L/32 configuration."""
    config = get_l16_config()
    config.patches.size = (32, 32)
    return config


def get_h14_config():
    """Returns the ViT-L/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (14, 14)})
    config.hidden_size = 1280
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 5120
    config.transformer.num_heads = 16
    config.transformer.num_layers = 32
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'token'
    config.representation_size = None

    return config


# --- [Original file: networks/vit_seg_modeling_resnet_skip.py] ---
def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


class StdConv2d(nn.Conv2d):

    def forward(self, x):
        w = self.weight
        v, m = torch.var_mean(w, dim=[1, 2, 3], keepdim=True, unbiased=False)
        w = (w - m) / torch.sqrt(v + 1e-5)
        return F.conv2d(x, w, self.bias, self.stride, self.padding,
                        self.dilation, self.groups)


def conv3x3(cin, cout, stride=1, groups=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=3, stride=stride,
                     padding=1, bias=bias, groups=groups)


def conv1x1(cin, cout, stride=1, bias=False):
    return StdConv2d(cin, cout, kernel_size=1, stride=stride,
                     padding=0, bias=bias)


class PreActBottleneck(nn.Module):
    """Pre-activation (v2) bottleneck block.
    """

    def __init__(self, cin, cout=None, cmid=None, stride=1):
        super().__init__()
        cout = cout or cin
        cmid = cmid or cout//4

        self.gn1 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv1 = conv1x1(cin, cmid, bias=False)
        self.gn2 = nn.GroupNorm(32, cmid, eps=1e-6)
        self.conv2 = conv3x3(cmid, cmid, stride, bias=False)  # Original code has it on conv1!!
        self.gn3 = nn.GroupNorm(32, cout, eps=1e-6)
        self.conv3 = conv1x1(cmid, cout, bias=False)
        self.relu = nn.ReLU(inplace=True)

        if (stride != 1 or cin != cout):
            # Projection also with pre-activation according to paper.
            self.downsample = conv1x1(cin, cout, stride, bias=False)
            self.gn_proj = nn.GroupNorm(cout, cout)

    def forward(self, x):

        # Residual branch
        residual = x
        if hasattr(self, 'downsample'):
            residual = self.downsample(x)
            residual = self.gn_proj(residual)

        # Unit's branch
        y = self.relu(self.gn1(self.conv1(x)))
        y = self.relu(self.gn2(self.conv2(y)))
        y = self.gn3(self.conv3(y))

        y = self.relu(residual + y)
        return y

    def load_from(self, weights, n_block, n_unit):
        conv1_weight = np2th(weights[pjoin(n_block, n_unit, "conv1/kernel")], conv=True)
        conv2_weight = np2th(weights[pjoin(n_block, n_unit, "conv2/kernel")], conv=True)
        conv3_weight = np2th(weights[pjoin(n_block, n_unit, "conv3/kernel")], conv=True)

        gn1_weight = np2th(weights[pjoin(n_block, n_unit, "gn1/scale")])
        gn1_bias = np2th(weights[pjoin(n_block, n_unit, "gn1/bias")])

        gn2_weight = np2th(weights[pjoin(n_block, n_unit, "gn2/scale")])
        gn2_bias = np2th(weights[pjoin(n_block, n_unit, "gn2/bias")])

        gn3_weight = np2th(weights[pjoin(n_block, n_unit, "gn3/scale")])
        gn3_bias = np2th(weights[pjoin(n_block, n_unit, "gn3/bias")])

        self.conv1.weight.copy_(conv1_weight)
        self.conv2.weight.copy_(conv2_weight)
        self.conv3.weight.copy_(conv3_weight)

        self.gn1.weight.copy_(gn1_weight.view(-1))
        self.gn1.bias.copy_(gn1_bias.view(-1))

        self.gn2.weight.copy_(gn2_weight.view(-1))
        self.gn2.bias.copy_(gn2_bias.view(-1))

        self.gn3.weight.copy_(gn3_weight.view(-1))
        self.gn3.bias.copy_(gn3_bias.view(-1))

        if hasattr(self, 'downsample'):
            proj_conv_weight = np2th(weights[pjoin(n_block, n_unit, "conv_proj/kernel")], conv=True)
            proj_gn_weight = np2th(weights[pjoin(n_block, n_unit, "gn_proj/scale")])
            proj_gn_bias = np2th(weights[pjoin(n_block, n_unit, "gn_proj/bias")])

            self.downsample.weight.copy_(proj_conv_weight)
            self.gn_proj.weight.copy_(proj_gn_weight.view(-1))
            self.gn_proj.bias.copy_(proj_gn_bias.view(-1))


class ResNetV2(nn.Module):
    """Implementation of Pre-activation (v2) ResNet mode."""

    def __init__(self, block_units, width_factor):
        super().__init__()
        width = int(64 * width_factor)
        self.width = width

        self.root = nn.Sequential(OrderedDict([
            ('conv', StdConv2d(3, width, kernel_size=7, stride=2, bias=False, padding=3)),
            ('gn', nn.GroupNorm(32, width, eps=1e-6)),
            ('relu', nn.ReLU(inplace=True)),
            # ('pool', nn.MaxPool2d(kernel_size=3, stride=2, padding=0))
        ]))

        self.body = nn.Sequential(OrderedDict([
            ('block1', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width, cout=width*4, cmid=width))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*4, cout=width*4, cmid=width)) for i in range(2, block_units[0] + 1)],
                ))),
            ('block2', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*4, cout=width*8, cmid=width*2, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*8, cout=width*8, cmid=width*2)) for i in range(2, block_units[1] + 1)],
                ))),
            ('block3', nn.Sequential(OrderedDict(
                [('unit1', PreActBottleneck(cin=width*8, cout=width*16, cmid=width*4, stride=2))] +
                [(f'unit{i:d}', PreActBottleneck(cin=width*16, cout=width*16, cmid=width*4)) for i in range(2, block_units[2] + 1)],
                ))),
        ]))

    def forward(self, x):
        features = []
        b, c, in_size, _ = x.size()
        x = self.root(x)
        features.append(x)
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)(x)
        for i in range(len(self.body)-1):
            x = self.body[i](x)
            right_size = int(in_size / 4 / (i+1))
            if x.size()[2] != right_size:
                pad = right_size - x.size()[2]
                assert pad < 3 and pad > 0, "x {} should {}".format(x.size(), right_size)
                feat = torch.zeros((b, x.size()[1], right_size, right_size), device=x.device)
                feat[:, :, 0:x.size()[2], 0:x.size()[3]] = x[:]
            else:
                feat = x
            features.append(feat)
        x = self.body[-1](x)
        return x, features[::-1]


# --- [Original file: networks/vit_seg_modeling.py] ---
logger = logging.getLogger(__name__)


ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids):
        embedding_output, features = self.embeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)  # (B, n_patch, hidden)
        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class SegmentationHead(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip):  # re-select the skip channels according to n_skip
                skip_channels[3-i]=0

        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x, attn_weights, features = self.transformer(x)  # (B, n_patch, hidden)
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))

            self.transformer.encoder.encoder_norm.weight.copy_(np2th(weights["Transformer/encoder_norm/scale"]))
            self.transformer.encoder.encoder_norm.bias.copy_(np2th(weights["Transformer/encoder_norm/bias"]))

            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  # th2np
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            # Encoder whole
            for bname, block in self.transformer.encoder.named_children():
                for uname, unit in block.named_children():
                    unit.load_from(weights, n_block=uname)

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': get_b16_config(),
    'ViT-B_32': get_b32_config(),
    'ViT-L_16': get_l16_config(),
    'ViT-L_32': get_l32_config(),
    'ViT-H_14': get_h14_config(),
    'R50-ViT-B_16': get_r50_b16_config(),
    'R50-ViT-L_16': get_r50_l16_config(),
    'testing': get_testing(),
}


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

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
    print("TransUNet core model component benchmark")
    print("=" * 70)

    def _make_tiny_config(img_size=32, hidden_size=8, mlp_dim=16, num_heads=2, num_layers=2, n_classes=3):
        config = ml_collections.ConfigDict()
        config.patches = ml_collections.ConfigDict({'size': (16, 16)})
        config.hidden_size = hidden_size
        config.transformer = ml_collections.ConfigDict()
        config.transformer.mlp_dim = mlp_dim
        config.transformer.num_heads = num_heads
        config.transformer.num_layers = num_layers
        config.transformer.attention_dropout_rate = 0.0
        config.transformer.dropout_rate = 0.0
        config.classifier = 'seg'
        config.representation_size = None
        config.decoder_channels = (16, 8, 4, 2)
        config.skip_channels = [0, 0, 0, 0]
        config.n_classes = n_classes
        config.n_skip = 0
        config.activation = 'softmax'
        return config

    def _make_tiny_weights(config, img_size=32, posemb_tokens=10):
        weights = {}
        hidden = config.hidden_size
        mlp_dim = config.transformer["mlp_dim"]
        patch = config.patches["size"][0]
        weights["embedding/kernel"] = np.random.randn(patch, patch, 3, hidden).astype(np.float32)
        weights["embedding/bias"] = np.random.randn(hidden).astype(np.float32)
        weights["Transformer/encoder_norm/scale"] = np.ones(hidden, dtype=np.float32)
        weights["Transformer/encoder_norm/bias"] = np.zeros(hidden, dtype=np.float32)
        weights["Transformer/posembed_input/pos_embedding"] = np.random.randn(1, posemb_tokens, hidden).astype(np.float32)
        for block_id in range(config.transformer["num_layers"]):
            root = f"Transformer/encoderblock_{block_id}"
            for name in [ATTENTION_Q, ATTENTION_K, ATTENTION_V, ATTENTION_OUT]:
                weights[pjoin(root, name, "kernel")] = np.random.randn(hidden, hidden).astype(np.float32)
                weights[pjoin(root, name, "bias")] = np.random.randn(hidden).astype(np.float32)
            weights[pjoin(root, FC_0, "kernel")] = np.random.randn(hidden, mlp_dim).astype(np.float32)
            weights[pjoin(root, FC_0, "bias")] = np.random.randn(mlp_dim).astype(np.float32)
            weights[pjoin(root, FC_1, "kernel")] = np.random.randn(mlp_dim, hidden).astype(np.float32)
            weights[pjoin(root, FC_1, "bias")] = np.random.randn(hidden).astype(np.float32)
            weights[pjoin(root, ATTENTION_NORM, "scale")] = np.ones(hidden, dtype=np.float32)
            weights[pjoin(root, ATTENTION_NORM, "bias")] = np.zeros(hidden, dtype=np.float32)
            weights[pjoin(root, MLP_NORM, "scale")] = np.ones(hidden, dtype=np.float32)
            weights[pjoin(root, MLP_NORM, "bias")] = np.zeros(hidden, dtype=np.float32)
        return weights

    device = torch.device("cpu")
    tiny_config = _make_tiny_config()

    print("-" * 60)
    print("[Group 1] StdConv2d and PreActBottleneck")
    try:
        std = StdConv2d(3, 32, kernel_size=3, padding=1, bias=False).to(device)
        x_std = torch.randn(2, 3, 16, 16, device=device)
        y_std = std(x_std)
        check("StdConv2d output not None", y_std is not None)
        check("StdConv2d output shape", y_std is not None and tuple(y_std.shape) == (2, 32, 16, 16),
              f"got {None if y_std is None else tuple(y_std.shape)}")
        check("StdConv2d output finite", y_std is not None and torch.isfinite(y_std).all().item())

        block = PreActBottleneck(cin=32, cout=64, cmid=32, stride=2).to(device)
        x_block = torch.randn(2, 32, 16, 16, device=device)
        y_block = block(x_block)
        check("PreActBottleneck output not None", y_block is not None)
        check("PreActBottleneck output shape", y_block is not None and tuple(y_block.shape) == (2, 64, 8, 8),
              f"got {None if y_block is None else tuple(y_block.shape)}")
        check("PreActBottleneck output finite", y_block is not None and torch.isfinite(y_block).all().item())
        check("PreActBottleneck nonnegative after relu", y_block is not None and (y_block >= 0).all().item())
    except Exception as exc:
        skip_checks(7, f"resnet stem/bottleneck raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 2] ResNetV2 hybrid feature extractor")
    try:
        resnet = ResNetV2(block_units=(1, 1, 1), width_factor=0.5).to(device)
        x_res = torch.randn(2, 3, 64, 64, device=device)
        y_res, features = resnet(x_res)
        check("ResNetV2 output not None", y_res is not None)
        check("ResNetV2 feature list length", isinstance(features, list) and len(features) == 3)
        check("ResNetV2 output shape", y_res is not None and tuple(y_res.shape) == (2, 512, 4, 4),
              f"got {None if y_res is None else tuple(y_res.shape)}")
        check("ResNetV2 first skip shape", len(features) > 0 and tuple(features[0].shape) == (2, 256, 8, 8),
              f"got {None if not features else tuple(features[0].shape)}")
        check("ResNetV2 last skip shape", len(features) > 2 and tuple(features[2].shape) == (2, 32, 32, 32),
              f"got {None if len(features) <= 2 else tuple(features[2].shape)}")
        check("ResNetV2 values finite", y_res is not None and torch.isfinite(y_res).all().item())
    except Exception as exc:
        skip_checks(6, f"ResNetV2 raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 3] Attention")
    try:
        attn = Attention(tiny_config, vis=True).to(device)
        x_attn = torch.randn(2, 5, 8, device=device)
        y_attn, weights = attn(x_attn)
        check("Attention output not None", y_attn is not None)
        check("Attention output shape", y_attn is not None and tuple(y_attn.shape) == (2, 5, 8),
              f"got {None if y_attn is None else tuple(y_attn.shape)}")
        check("Attention weights shape", weights is not None and tuple(weights.shape) == (2, 2, 5, 5),
              f"got {None if weights is None else tuple(weights.shape)}")
        check("Attention weights sum to one", weights is not None and torch.allclose(weights.sum(dim=-1), torch.ones(2, 2, 5), atol=1e-5))
        check("Attention output finite", y_attn is not None and torch.isfinite(y_attn).all().item())
    except Exception as exc:
        skip_checks(5, f"Attention raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 4] Embeddings")
    try:
        emb = Embeddings(tiny_config, img_size=32).to(device)
        x_emb = torch.randn(2, 3, 32, 32, device=device)
        y_emb, emb_features = emb(x_emb)
        check("Embeddings output not None", y_emb is not None)
        check("Embeddings output shape", y_emb is not None and tuple(y_emb.shape) == (2, 4, 8),
              f"got {None if y_emb is None else tuple(y_emb.shape)}")
        check("Embeddings no-hybrid features", emb_features is None)
        check("Embeddings position shape", tuple(emb.position_embeddings.shape) == (1, 4, 8))
        check("Embeddings output finite", y_emb is not None and torch.isfinite(y_emb).all().item())
    except Exception as exc:
        skip_checks(5, f"Embeddings raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 5] Block and Encoder")
    try:
        enc_block = Block(tiny_config, vis=True).to(device)
        x_blk = torch.randn(2, 4, 8, device=device)
        y_blk, w_blk = enc_block(x_blk)
        check("Block output not None", y_blk is not None)
        check("Block output shape", y_blk is not None and tuple(y_blk.shape) == (2, 4, 8),
              f"got {None if y_blk is None else tuple(y_blk.shape)}")
        check("Block attention weights returned", w_blk is not None and tuple(w_blk.shape) == (2, 2, 4, 4))

        encoder = Encoder(tiny_config, vis=True).to(device)
        y_enc, w_enc = encoder(x_blk)
        check("Encoder output not None", y_enc is not None)
        check("Encoder output shape", y_enc is not None and tuple(y_enc.shape) == (2, 4, 8),
              f"got {None if y_enc is None else tuple(y_enc.shape)}")
        check("Encoder attention list length", isinstance(w_enc, list) and len(w_enc) == 2)
        check("Encoder output finite", y_enc is not None and torch.isfinite(y_enc).all().item())
    except Exception as exc:
        skip_checks(7, f"Block/Encoder raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 6] DecoderBlock and DecoderCup")
    try:
        dec_block = DecoderBlock(in_channels=16, out_channels=8, skip_channels=4).to(device)
        x_dec = torch.randn(2, 16, 8, 8, device=device)
        skip_dec = torch.randn(2, 4, 16, 16, device=device)
        y_dec = dec_block(x_dec, skip=skip_dec)
        check("DecoderBlock output not None", y_dec is not None)
        check("DecoderBlock output shape", y_dec is not None and tuple(y_dec.shape) == (2, 8, 16, 16),
              f"got {None if y_dec is None else tuple(y_dec.shape)}")
        check("DecoderBlock output finite", y_dec is not None and torch.isfinite(y_dec).all().item())

        decoder = DecoderCup(tiny_config).to(device)
        hidden = torch.randn(2, 4, 8, device=device)
        y_cup = decoder(hidden, features=None)
        check("DecoderCup output not None", y_cup is not None)
        check("DecoderCup output shape", y_cup is not None and tuple(y_cup.shape) == (2, 2, 32, 32),
              f"got {None if y_cup is None else tuple(y_cup.shape)}")
        check("DecoderCup output finite", y_cup is not None and torch.isfinite(y_cup).all().item())
    except Exception as exc:
        skip_checks(6, f"Decoder raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 7] VisionTransformer forward")
    try:
        model = VisionTransformer(tiny_config, img_size=32, num_classes=tiny_config.n_classes, vis=True).to(device)
        x_model = torch.randn(2, 1, 32, 32, device=device)
        logits = model(x_model)
        check("VisionTransformer output not None", logits is not None)
        check("VisionTransformer logits shape", logits is not None and tuple(logits.shape) == (2, 3, 32, 32),
              f"got {None if logits is None else tuple(logits.shape)}")
        check("VisionTransformer logits finite", logits is not None and torch.isfinite(logits).all().item())
        check("VisionTransformer grayscale repeat branch", model.transformer.embeddings.patch_embeddings.in_channels == 3)
        check("VisionTransformer has segmentation head", isinstance(model.segmentation_head, SegmentationHead))
    except Exception as exc:
        skip_checks(5, f"VisionTransformer forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 8] VisionTransformer load_from")
    try:
        load_config = _make_tiny_config(num_layers=1)
        load_model = VisionTransformer(load_config, img_size=32, num_classes=load_config.n_classes, vis=False).to(device)
        weights = _make_tiny_weights(load_config, img_size=32, posemb_tokens=10)
        load_model.load_from(weights)
        posemb = load_model.transformer.embeddings.position_embeddings
        check("load_from position embedding shape", tuple(posemb.shape) == (1, 4, 8),
              f"got {tuple(posemb.shape)}")
        check("load_from position embedding finite", torch.isfinite(posemb).all().item())
        check("load_from patch bias copied", torch.allclose(load_model.transformer.embeddings.patch_embeddings.bias,
                                                           torch.from_numpy(weights["embedding/bias"]), atol=1e-6))
        check("load_from encoder norm finite", torch.isfinite(load_model.transformer.encoder.encoder_norm.weight).all().item())
        check("load_from resized posemb nonzero", posemb.abs().sum().item() > 0)
    except Exception as exc:
        skip_checks(5, f"load_from raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some target functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
