# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Consolidated benchmark answer key for Long Short-Term Transformer (LSTR).

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Registry(dict):

    def register(self, name):
        def decorator(obj):
            self[name] = obj
            return obj
        return decorator


def layer_norm(d_model, condition=True):
    return nn.LayerNorm(d_model) if condition else None


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class DotProductAttention(nn.Module):

    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()

        self.dropout = dropout

    def forward(self, q, k, v, attn_mask=None):
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, v)
        return attn_output


class DotProductAttentionStream(DotProductAttention):

    def __init__(self, dropout=0.0):
        super(DotProductAttentionStream, self).__init__(dropout)

        ############################
        # Cache for stream inference
        ############################
        self.k_weights_cache = None
        self.k_pos_weights_cache = None

    def stream_inference(self, q, k, v, k_pos, v_pos, attn_mask=None):
        if self.k_weights_cache is not None:
            k_weights_new = torch.bmm(q, k[:, [-1]].transpose(1, 2))
            k_weights = torch.cat((self.k_weights_cache[:, :, 1:], k_weights_new), dim=-1)
            self.k_weights_cache = k_weights
            k_pos_weights = self.k_pos_weights_cache
        else:
            k_weights = torch.bmm(q, k.transpose(1, 2))
            self.k_weights_cache = k_weights
            k_pos_weights = torch.bmm(q, k_pos.transpose(1, 2))
            self.k_pos_weights_cache = k_pos_weights
        attn_output_weights = k_weights + k_pos_weights

        if attn_mask is not None:
            attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights,
                                        p=self.dropout,
                                        training=self.training)
        attn_output = torch.bmm(attn_output_weights, (v + v_pos))
        return attn_output


class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        if self._qkv_same_embed_dim:
            self.in_proj_weight = nn.Parameter(torch.empty(3 * embed_dim, embed_dim))
        else:
            raise RuntimeError('Do not support q, k, v have different dimensions')

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.in_proj_weight)

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)

        self.dotproductattention = DotProductAttention(dropout)

    def forward(self, q, k, v, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        _b = self.in_proj_bias
        _start = None
        _end = embed_dim
        _w = self.in_proj_weight[:_end, :]
        if _b is not None:
            _b = _b[:_end]
        q = F.linear(q, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim
        _end = embed_dim * 2
        _w = self.in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        k = F.linear(k, _w, _b)

        _b = self.in_proj_bias
        _start = embed_dim * 2
        _end = None
        _w = self.in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        v = F.linear(v, _w, _b)

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output = self.dotproductattention(q, k, v, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None


class MultiheadAttentionStream(MultiheadAttention):

    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, kdim=None, vdim=None):
        super(MultiheadAttentionStream, self).__init__(embed_dim, num_heads, dropout, bias, kdim, vdim)

        self.dotproductattention = DotProductAttentionStream(dropout)

        ############################
        # Cache for stream inference
        ############################
        self.q_cache = None
        self.k_cache = None
        self.v_cache = None
        self.k_pos_cache = None
        self.v_pos_cache = None

    def stream_inference(self, q, k, v, pos, attn_mask=None, key_padding_mask=None):
        tsz, bsz, embed_dim = q.shape[0], q.shape[1], q.shape[2]

        head_dim = embed_dim // self.num_heads
        assert head_dim * self.num_heads == embed_dim, \
            'embed_dim must be divisible by num_heads'
        scaling = float(head_dim) ** -0.5

        if self.q_cache is not None:
            q = self.q_cache
        else:
            _b = self.in_proj_bias
            _start = None
            _end = embed_dim
            _w = self.in_proj_weight[:_end, :]
            if _b is not None:
                _b = _b[:_end]
            q = F.linear(q, _w, _b)
            self.q_cache = q

        assert (self.k_cache is None) == (self.k_pos_cache is None)
        if self.k_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k_new = F.linear(k[[-1]], _w, None)
            k = torch.cat((self.k_cache[1:], k_new))
            self.k_cache = k
            k_pos = self.k_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = self.in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(k, _w, None)
            self.k_cache = k
            k_pos = F.linear(pos, _w, _b)
            self.k_pos_cache = k_pos

        assert (self.v_cache is None) == (self.v_pos_cache is None)
        if self.v_cache is not None:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v_new = F.linear(v[[-1]], _w, None)
            v = torch.cat((self.v_cache[1:], v_new))
            self.v_cache = v
            v_pos = self.v_pos_cache
        else:
            _b = self.in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = self.in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(v, _w, None)
            self.v_cache = v
            v_pos = F.linear(pos, _w, _b)
            self.v_pos_cache = v_pos

        q = q * scaling

        q = q.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        k_pos = k_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)
        v_pos = v_pos.contiguous().view(-1, bsz * self.num_heads, head_dim).transpose(0, 1)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0).repeat(bsz, 1, 1)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            attn_mask = attn_mask.reshape(-1, *attn_mask.shape[2:])

        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, tsz, 1)
            key_padding_mask = key_padding_mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            key_padding_mask = key_padding_mask.reshape(-1, *key_padding_mask.shape[2:])

        if attn_mask is not None and key_padding_mask is not None:
            mask = attn_mask + key_padding_mask
        elif attn_mask is not None:
            mask = attn_mask
        elif key_padding_mask is not None:
            mask = key_padding_mask
        else:
            mask = None

        attn_output = self.dotproductattention.stream_inference(q, k, v, k_pos, v_pos, mask)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tsz, bsz,
                                                                    self.embed_dim)
        return self.out_proj(attn_output), None


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation='relu', custom_encoder=None, custom_decoder=None):
        super(Transformer, self).__init__()

        if custom_encoder is not None:
            self.encoder = custom_encoder
        else:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            encoder_norm = nn.LayerNorm(d_model)
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if custom_decoder is not None:
            self.decoder = custom_decoder
        else:
            decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation)
            decoder_norm = nn.LayerNorm(d_model)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.d_model = d_model
        self.nhead = nhead

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if src.size(1) != tgt.size(1):
            raise RuntimeError('the batch number of src and tgt must be equal')

        if src.size(2) != self.d_model or tgt.size(2) != self.d_model:
            raise RuntimeError('the feature number of src and tgt must be equal to d_model')

        memory = self.encoder(src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        return output


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()

        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        output = src

        for mod in self.layers:
            output = mod(output, src_mask=src_mask,
                         src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()

        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def stream_inference(self, tgt, memory, pos, tgt_mask=None,
                         memory_mask=None, tgt_key_padding_mask=None,
                         memory_key_padding_mask=None):
        output = tgt

        if len(self.layers) != 1:
            raise RuntimeError('Number of layers cannot larger than 1 for stream inference')

        output = self.layers[0].stream_inference(output, memory, pos,
                                                 tgt_mask=tgt_mask,
                                                 memory_mask=memory_mask,
                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                 memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiheadAttentionStream(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiheadAttentionStream(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttentionStream(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

        ############################
        # Cache for stream inference
        ############################
        self.tgt_cache = None

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def stream_inference(self, tgt, memory, pos, tgt_mask=None, memory_mask=None,
                         tgt_key_padding_mask=None, memory_key_padding_mask=None):
        if self.tgt_cache is None:
            tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)
            self.tgt_cache = tgt
        else:
            tgt = self.tgt_cache
        tgt2 = self.multihead_attn.stream_inference(tgt, memory, memory, pos, attn_mask=memory_mask,
                                                    key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    if activation == 'relu':
        return F.relu
    elif activation == 'gelu':
        return F.gelu

    raise RuntimeError('activation should be relu/gelu, not {}'.format(activation))


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, padding=0):
        x = x + self.pe[padding: padding + x.shape[0], :]
        return self.dropout(x)


FEATURE_HEADS = Registry()
FEATURE_SIZES = {
    'rgb_anet_resnet50': 2048,
    'flow_anet_resnet50': 2048,
    'rgb_kinetics_bninception': 1024,
    'flow_kinetics_bninception': 1024,
    'rgb_kinetics_resnet50': 2048,
    'flow_kinetics_resnet50': 2048,
}


@FEATURE_HEADS.register('THUMOS')
@FEATURE_HEADS.register('TVSeries')
class BaseFeatureHead(nn.Module):

    def __init__(self, cfg):
        super(BaseFeatureHead, self).__init__()

        if cfg.INPUT.MODALITY in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in cfg.INPUT.MODALITY
            self.with_motion = 'visual' not in cfg.INPUT.MODALITY
        else:
            raise RuntimeError('Unknown modality of {}'.format(cfg.INPUT.MODALITY))

        if self.with_visual and self.with_motion:
            visual_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
            motion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = FEATURE_SIZES[cfg.INPUT.VISUAL_FEATURE]
        elif self.with_motion:
            fusion_size = FEATURE_SIZES[cfg.INPUT.MOTION_FEATURE]

        self.d_model = fusion_size

        if cfg.MODEL.FEATURE_HEAD.LINEAR_ENABLED:
            if cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES != -1:
                self.d_model = cfg.MODEL.FEATURE_HEAD.LINEAR_OUT_FEATURES
            self.input_linear = nn.Sequential(
                nn.Linear(fusion_size, self.d_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_linear = nn.Identity()

    def forward(self, visual_input, motion_input):
        if self.with_visual and self.with_motion:
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
        elif self.with_visual:
            fusion_input = visual_input
        elif self.with_motion:
            fusion_input = motion_input
        fusion_input = self.input_linear(fusion_input)
        return fusion_input


def build_feature_head(cfg):
    feature_head = FEATURE_HEADS[cfg.DATA.DATA_NAME]
    return feature_head(cfg)


META_ARCHITECTURES = Registry()


class LSTR(nn.Module):

    def __init__(self, cfg):
        super(LSTR, self).__init__()

        # Build long feature heads
        self.long_memory_num_samples = cfg.MODEL.LSTR.LONG_MEMORY_NUM_SAMPLES
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg)

        # Build work feature head
        self.work_memory_num_samples = cfg.MODEL.LSTR.WORK_MEMORY_NUM_SAMPLES
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = build_feature_head(cfg)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.activation = cfg.MODEL.LSTR.ACTIVATION
        self.num_classes = cfg.DATA.NUM_CLASSES

        # Build position encoding
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for param in cfg.MODEL.LSTR.ENC_MODULE:
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(TransformerDecoder(
                        enc_layer, param[1], layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(TransformerEncoder(
                        enc_layer, param[1], layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        if self.long_enabled:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = TransformerDecoder(
                dec_layer, param[1], layer_norm(self.d_model, param[2]))
        else:
            param = cfg.MODEL.LSTR.DEC_MODULE
            dec_layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = TransformerEncoder(
                dec_layer, param[1], layer_norm(self.d_model, param[2]))

        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

    def forward(self, visual_inputs, motion_inputs, memory_key_padding_mask=None):
        if self.long_enabled:
            # Compute long memories
            long_memories = self.pos_encoding(self.feature_head_long(
                visual_inputs[:, :self.long_memory_num_samples],
                motion_inputs[:, :self.long_memory_num_samples],
            ).transpose(0, 1))

            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                        memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                visual_inputs[:, self.long_memory_num_samples:],
                motion_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)


@META_ARCHITECTURES.register('LSTR')
class LSTRStream(LSTR):

    def __init__(self, cfg):
        super(LSTRStream, self).__init__(cfg)

        ############################
        # Cache for stream inference
        ############################
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None

    def stream_inference(self,
                         long_visual_inputs,
                         long_motion_inputs,
                         work_visual_inputs,
                         work_motion_inputs,
                         memory_key_padding_mask=None):
        assert self.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(self.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'

        if (long_visual_inputs is not None) and (long_motion_inputs is not None):
            # Compute long memories
            long_memories = self.feature_head_long(
                long_visual_inputs,
                long_motion_inputs,
            ).transpose(0, 1)

            if self.long_memories_cache is None:
                self.long_memories_cache = long_memories
            else:
                self.long_memories_cache = torch.cat((
                    self.long_memories_cache[1:], long_memories
                ))

            long_memories = self.long_memories_cache
            pos = self.pos_encoding.pe[:self.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask)
            self.compressed_long_memories_cache = long_memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = self.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                work_visual_inputs,
                work_motion_inputs,
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)


if __name__ == "__main__":
    failed = []
    skipped = []

    def check(name, condition, details=""):
        if bool(condition):
            print(f"  [{name}] PASS")
        else:
            failed.append(f"{name}: {details}" if details else name)
            suffix = f" - {details}" if details else ""
            print(f"  [{name}] FAIL{suffix}")

    def skip_checks(group, reason, count):
        for idx in range(count):
            skipped.append(f"{group} #{idx + 1}: {reason}")
        print(f"  [skipped {count} check(s)] FAIL - {group}: {reason}")

    class Node:
        ...

    def node(**kwargs):
        obj = Node()
        for key, value in kwargs.items():
            setattr(obj, key, value)
        return obj

    def make_cfg():
        cfg = node()
        cfg.DATA = node(DATA_NAME='THUMOS', NUM_CLASSES=5)
        cfg.INPUT = node(
            MODALITY='twostream',
            VISUAL_FEATURE='rgb_kinetics_bninception',
            MOTION_FEATURE='flow_kinetics_bninception',
        )
        cfg.MODEL = node()
        cfg.MODEL.FEATURE_HEAD = node(LINEAR_ENABLED=True, LINEAR_OUT_FEATURES=16)
        cfg.MODEL.LSTR = node(
            LONG_MEMORY_NUM_SAMPLES=4,
            WORK_MEMORY_NUM_SAMPLES=3,
            NUM_HEADS=4,
            DIM_FEEDFORWARD=32,
            DROPOUT=0.0,
            ACTIVATION='relu',
            ENC_MODULE=[[2, 1, True]],
            DEC_MODULE=[-1, 1, True],
        )
        return cfg

    print("=" * 70)
    print("Long Short-Term Transformer benchmark: streaming LSTR components")
    print("Automated Test Suite - 5 ablated functions")
    print("=" * 70)
    print()

    torch.manual_seed(7)

    print("-" * 60)
    print("[Test 1/5] DotProductAttentionStream.stream_inference - cached dot-product attention")
    try:
        attn = DotProductAttentionStream(dropout=0.0)
        attn.eval()
        q = torch.randn(2, 2, 4)
        k = torch.randn(2, 3, 4)
        v = torch.randn(2, 3, 4)
        k_pos = torch.randn(2, 3, 4)
        v_pos = torch.randn(2, 3, 4)
        out = attn.stream_inference(q, k, v, k_pos, v_pos)
        out_second = attn.stream_inference(q, k + 0.1, v - 0.1, k_pos, v_pos)
        check("dot stream output shape", out.shape == (2, 2, 4), str(tuple(out.shape)))
        check("dot stream output finite", torch.isfinite(out).all().item())
        check("dot stream cache shape", attn.k_weights_cache.shape == (2, 2, 3), str(tuple(attn.k_weights_cache.shape)))
        check("dot stream cache reused shape", out_second.shape == out.shape, str(tuple(out_second.shape)))
        check("dot stream positional cache present", attn.k_pos_weights_cache is not None)
    except Exception as exc:
        failed.append(f"dot stream checks raised {type(exc).__name__}: {exc}")
        print(f"  [dot stream checks] FAIL - raised {type(exc).__name__}: {exc}")
        skip_checks("dot stream dependent checks", "exception in stream attention", 4)
    print()

    print("-" * 60)
    print("[Test 2/5] MultiheadAttentionStream.stream_inference - projection and cache reuse")
    try:
        mha = MultiheadAttentionStream(embed_dim=16, num_heads=4, dropout=0.0)
        mha.eval()
        q = torch.randn(2, 1, 16)
        k = torch.randn(4, 1, 16)
        v = torch.randn(4, 1, 16)
        pos = torch.randn(4, 1, 16)
        out, weights = mha.stream_inference(q, k, v, pos)
        out_second, _ = mha.stream_inference(q + 0.01, k + 0.01, v - 0.01, pos)
        check("multihead stream output shape", out.shape == (2, 1, 16), str(tuple(out.shape)))
        check("multihead stream returns no weights", weights is None)
        check("multihead stream output finite", torch.isfinite(out).all().item())
        check("multihead stream key cache shape", mha.k_cache.shape == (4, 1, 16), str(tuple(mha.k_cache.shape)))
        check("multihead stream repeated call shape", out_second.shape == out.shape, str(tuple(out_second.shape)))
    except Exception as exc:
        failed.append(f"multihead stream checks raised {type(exc).__name__}: {exc}")
        print(f"  [multihead stream checks] FAIL - raised {type(exc).__name__}: {exc}")
        skip_checks("multihead dependent checks", "exception in multihead stream attention", 4)
    print()

    print("-" * 60)
    print("[Test 3/5] TransformerDecoderLayer.stream_inference - streaming decoder layer")
    try:
        layer = TransformerDecoderLayer(d_model=16, nhead=4, dim_feedforward=32, dropout=0.0)
        layer.eval()
        tgt = torch.randn(2, 1, 16)
        memory = torch.randn(4, 1, 16)
        pos = torch.randn(4, 1, 16)
        out = layer.stream_inference(tgt, memory, pos)
        out_second = layer.stream_inference(tgt + 0.1, memory - 0.1, pos)
        check("decoder stream output shape", out.shape == (2, 1, 16), str(tuple(out.shape)))
        check("decoder stream output finite", torch.isfinite(out).all().item())
        check("decoder stream target cache shape", layer.tgt_cache.shape == (2, 1, 16), str(tuple(layer.tgt_cache.shape)))
        check("decoder stream repeated call shape", out_second.shape == out.shape, str(tuple(out_second.shape)))
    except Exception as exc:
        failed.append(f"decoder stream checks raised {type(exc).__name__}: {exc}")
        print(f"  [decoder stream checks] FAIL - raised {type(exc).__name__}: {exc}")
        skip_checks("decoder dependent checks", "exception in decoder stream layer", 3)
    print()

    print("-" * 60)
    print("[Test 4/5] LSTR.forward - offline long/work memory classification")
    try:
        cfg = make_cfg()
        model = LSTRStream(cfg)
        model.train()
        visual = torch.randn(2, 7, 1024)
        motion = torch.randn(2, 7, 1024)
        score = model(visual, motion)
        loss = score.sum()
        loss.backward()
        check("lstr forward score shape", score.shape == (2, 3, 5), str(tuple(score.shape)))
        check("lstr forward finite scores", torch.isfinite(score).all().item())
        check("lstr forward classifier gradient", model.classifier.weight.grad is not None)
        check("lstr forward long feature gradient", model.feature_head_long.input_linear[0].weight.grad is not None)
        model.eval()
        with torch.no_grad():
            eval_score = model(visual, motion)
        check("lstr eval score shape", eval_score.shape == (2, 3, 5), str(tuple(eval_score.shape)))
    except Exception as exc:
        failed.append(f"lstr forward checks raised {type(exc).__name__}: {exc}")
        print(f"  [lstr forward checks] FAIL - raised {type(exc).__name__}: {exc}")
        skip_checks("lstr forward dependent checks", "exception in LSTR forward", 4)
    print()

    print("-" * 60)
    print("[Test 5/5] LSTRStream.stream_inference - cached online inference")
    try:
        cfg = make_cfg()
        model = LSTRStream(cfg)
        model.eval()
        long_visual = torch.randn(1, 4, 1024)
        long_motion = torch.randn(1, 4, 1024)
        work_visual = torch.randn(1, 3, 1024)
        work_motion = torch.randn(1, 3, 1024)
        with torch.no_grad():
            stream_score = model.stream_inference(long_visual, long_motion, work_visual, work_motion)
            cached_score = model.stream_inference(None, None, work_visual + 0.05, work_motion - 0.05)
        check("lstr stream score shape", stream_score.shape == (1, 3, 5), str(tuple(stream_score.shape)))
        check("lstr stream score finite", torch.isfinite(stream_score).all().item())
        check("lstr stream long cache shape", model.long_memories_cache.shape == (4, 1, 16), str(tuple(model.long_memories_cache.shape)))
        check("lstr stream compressed cache shape", model.compressed_long_memories_cache.shape == (2, 1, 16), str(tuple(model.compressed_long_memories_cache.shape)))
        check("lstr stream cached branch shape", cached_score.shape == (1, 3, 5), str(tuple(cached_score.shape)))
    except Exception as exc:
        failed.append(f"lstr stream checks raised {type(exc).__name__}: {exc}")
        print(f"  [lstr stream checks] FAIL - raised {type(exc).__name__}: {exc}")
        skip_checks("lstr stream dependent checks", "exception in LSTR stream inference", 4)
    print()

    total_checks = 24
    passed = total_checks - len(failed) - len(skipped)
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {total_checks} checks passed")
    print("=" * 70)
    if failed or skipped:
        print(f"{len(failed) + len(skipped)} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
        print("=" * 70)
        raise SystemExit(1)
    print("All tests PASSED! The model code is complete and correct.")
    print("=" * 70)
