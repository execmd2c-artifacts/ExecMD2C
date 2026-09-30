"""
Self-contained Q2T benchmark answer key.

This file consolidates the core model components from Query2Triple:
- graph/distance-biased BERT attention from models/modeling_bert.py
- query-token embedding and Query2Triple encoding from models/query2triple.py
- KGE scoring heads used by the encoded (head, relation) triple

Training loops, datasets, evaluation loops, CLI wrappers, and the auxiliary
ssl-relation-prediction pretraining submodule are intentionally excluded.
"""

from typing import Optional, Tuple, Union, List

import collections
import logging
import math
import time
import os

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn as nn, Tensor
from torch.cuda.amp import GradScaler
from torch.nn.init import xavier_normal_, kaiming_uniform_, kaiming_normal_
from torch_geometric.data import Data
from torch_geometric.nn import MLP
from transformers import BertConfig, BertPreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions,     BaseModelOutputWithPoolingAndCrossAttentions
from transformers.pytorch_utils import prune_linear_layer, find_pruneable_heads_and_indices, apply_chunking_to_forward

import torch.nn.functional as F


TYPE_TO_IDX = {
    '1p': 0,
    '2i': 1,
    '3i': 2,
    '2p': 3,
    '3p': 4,
    '2in': 5,
    '3in': 6,
    'inp': 7,
    'pin': 8,
    'pni': 9
}

TYPE_TO_SMOOTH = {
    '1p': 0.1,
    '2i': 0.6,
    '3i': 0.6,
    '2p': 0.8,
    '3p': 0.8,
    '2in': 0.8,
    '3in': 0.8,
    'inp': 0.8,
    'pin': 0.8,
    'pni': 0.8,
}
PAD = 0
VAR = 1
TGT = 2
ENT_CLS = 3
REL_CLS = 4
OFFSET = 5


# --- [Original file: models/modeling_bert.py] ---
# Consolidation note: the internal HuggingFace-style BertEncoder class is named
# BertTransformerEncoder here to avoid colliding with Q2T's wrapper BertEncoder
# from models/query2triple.py.
class BertSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.enc_dist = config.enc_dist

        if self.enc_dist == 'u':
            self.dist_bias = nn.Embedding(12, self.num_attention_heads)
            self.offset = 0
        elif self.enc_dist == 'd':
            # 12 + 11
            self.dist_bias = nn.Embedding(23, self.num_attention_heads)
            self.offset = 11
        elif self.enc_dist == 'n':
            self.dist_bias = None
        elif self.enc_dist == 'no':
            self.dist_bias = None

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            dist_mat=None,
            negs=None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # neg_idxes = torch.where(negs == 1)
        # hidden_states[neg_idxes] = self.neg_proj(hidden_states[neg_idxes])

        query_layer = self.query(hidden_states)
        query_layer = self.transpose_for_scores(query_layer)

        key_layer = self.key(hidden_states)
        key_layer = self.transpose_for_scores(key_layer)

        value_layer = self.value(hidden_states)
        value_layer = self.transpose_for_scores(value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [n_batch, n_head, len, len]

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if self.dist_bias:
            # [b, len, len, n_head] -> [b, n_head, len, len]
            dist_bias = self.dist_bias(dist_mat + self.offset).permute(0, 3, 1, 2)
            attention_scores += dist_bias

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        # dist_mat_with_offset = dist_mat + 11
        #
        # # consider rel key
        # # dist_mat[i, j] : i -> j, dist_mat.T[i, j] : j -> i,
        # dist_key_layer = self.d_key(dist_mat_with_offset.transpose(1, 2))
        # new_x_shape = dist_key_layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # # batch, len, len, n_head, dim
        # dist_key_layer = dist_key_layer.view(new_x_shape)
        # # batch, n_head, len, len, dim
        # dist_key_layer = dist_key_layer.transpose(2, 3).transpose(1, 2)
        #
        # # 1, 1, 1, 1, 1
        # rep = [1 for _ in range(len(dist_key_layer.shape))]
        # # 1, 1, len, 1, 1
        # rep[-3] = dist_key_layer.shape[-2]
        # # batch, n_head, len, len, dim
        # key_layer = key_layer.unsqueeze(-3).repeat(rep) * dist_key_layer
        # query_layer = query_layer.unsqueeze(-1)  # b, n_head, len, dim, 1
        # attention_scores = torch.matmul(key_layer, query_layer).squeeze()
        #
        # attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # if attention_mask is not None:
        #     # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        #     attention_scores = attention_scores + attention_mask
        #
        # # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        #
        # # # This is actually dropping out entire tokens to attend to, which might
        # # # seem a bit unusual, but is taken from the original Transformer paper.
        # attention_probs = self.dropout(attention_probs)
        #
        # # Mask heads if we want to
        # if head_mask is not None:
        #     attention_probs = attention_probs * head_mask

        # # dist_mat[i, j] : i -> j, dist_mat.T[i, j] : j -> i,
        # rel_val_layer = self.d_value(dist_mat_with_offset.transpose(1, 2))
        # new_x_shape = rel_val_layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # # batch, len, len, n_head, dim
        # rel_val_layer = rel_val_layer.view(new_x_shape)
        # # batch, n_head, len, len, dim
        # rel_val_layer = rel_val_layer.transpose(2, 3).transpose(1, 2)
        # # batch, n_head, len, len, dim
        # value_layer = value_layer.unsqueeze(-3).repeat(rep) * rel_val_layer
        #
        # # batch, n_head, len, de
        # # context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer = attention_probs.unsqueeze(-1) * value_layer
        # context_layer = context_layer.sum(-2)

        # batch, len, n_head, de
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        # batch, len, n_head * de
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        self.self = BertSelfAttention(config, position_embedding_type=position_embedding_type)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            dist_mat=None,
            negs=None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            dist_mat,
            negs,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            dist_mat=None,
            negs=None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            dist_mat,
            negs,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertTransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        # self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None,
            dist_mat=None,
            negs=None,
            head_mask: Optional[torch.FloatTensor] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = False,
            output_hidden_states: Optional[bool] = False,
            return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None

        # neg_idxes = torch.where(negs == 1)
        # hidden_states[neg_idxes] = self.neg_pre_layer(
        #     hidden_states[neg_idxes],
        #     attention_mask[neg_idxes],
        #     dist_mat[neg_idxes],
        # )[0]

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                # if use_cache:
                #     logger.warning(
                #         "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                #     )
                #     use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    dist_mat,
                    negs,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class BertModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in [Attention is
    all you need](https://arxiv.org/abs/1706.03762) by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.encoder = BertTransformerEncoder(config)
        self.type = torch.float16 if config.fp16 else torch.float32

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
            self,
            inputs_embeds: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            dist_mat=None,
            negs=None,
            head_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
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

        use_cache = False

        input_shape = inputs_embeds.size()[:-1]

        batch_size, seq_length = input_shape
        device = inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape,
                                                                                 dtype=self.type)

        encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = inputs_embeds

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            dist_mat=dist_mat,
            negs=negs,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


# --- [Original file: models/query2triple.py] ---
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.1,
                 reduction="mean"):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def reduce_loss(self, loss):
        return loss.mean() if self.reduction == 'mean' else loss.sum() \
            if self.reduction == 'sum' else loss

    def linear_combination(self, x, y, smoothing=None):
        if smoothing is None:
            smoothing = self.smoothing

        return smoothing * x + (1 - smoothing) * y

    def forward(self, preds, target, query_types=None):
        if query_types is not None:
            smoothing = torch.ones(query_types.shape, device=query_types.device)
            for type_ in TYPE_TO_SMOOTH.keys():
                idx = TYPE_TO_IDX[type_]
                smoothing[query_types == idx] = TYPE_TO_SMOOTH[type_]
        else:
            assert 0 <= self.smoothing < 1
            smoothing = self.smoothing

        n = preds.size(-1)
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1)) / n
        nll = F.nll_loss(
            log_preds, target, reduction=self.reduction
        )
        return self.linear_combination(loss, nll, smoothing), log_preds


class Query2Triple(nn.Module):
    def __init__(self, num_ents, num_rels, hidden_dim, edge_to_entities,
                 **kwargs):
        super(Query2Triple, self).__init__()

        # basic setting
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.edge_to_entities = edge_to_entities
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']

        model = {
            'tucker': TuckER,
            'complex': Complex,
            'cp': CP,
            'rescal': RESCAL,
            'distmult': DistMult
        }[kwargs['geo'].lower()]
        self.kge_model = model(self.num_ents, self.num_rels,
                               self.dim_ent_embedding, self.dim_rel_embedding,
                               kwargs)

        self.kge_ckpt_path = kwargs['kge_ckpt_path']
        if self.kge_ckpt_path:
            self.kge_model.load_from_ckpt_path(self.kge_ckpt_path)

        self.ent_embedding = self.kge_model.ent_embedding
        self.rel_embedding = self.kge_model.rel_embedding
        kge_requires_grad = True if kwargs['not_freeze_kge'] else False

        if not kge_requires_grad:
            for name, param in self.kge_model.named_parameters():
                param.requires_grad = False

        logging.info(f'KGE requires_grad: {kge_requires_grad}')

        if self.kge_ckpt_path and not kge_requires_grad:
            self.use_kge_to_pred_1p = True
        else:
            self.use_kge_to_pred_1p = False
        logging.info(f'use_kge_to_pred_1p: {self.use_kge_to_pred_1p}')

        # var, tgt, CLS
        self.sp_token_embedding = nn.Embedding(OFFSET, self.dim_ent_embedding)
        # prompt
        self.query_encoder = BertEncoder(kwargs)

        self.fp16 = kwargs['fp16']
        if self.fp16:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.loss_fct = LabelSmoothingLoss(smoothing=kwargs['label_smoothing'], reduction='none')

        self.init_weight()

    def init_weight(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding) and module.weight.requires_grad:
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def init_seq_embedding(self, seq: torch.tensor):
        """
        :param seq:
        :param embedding1:
        :param embedding2:
        :return:
        """
        #  init sqe embedding
        shape = seq.shape + (self.ent_embedding.embedding_dim,)
        node_embeddings = torch.zeros(shape, device=self.device)

        # ent idx
        tup_idx = torch.where(seq >= OFFSET)
        node_embeddings[tup_idx] = self.ent_embedding(seq[tup_idx] - OFFSET)

        # 0: var, 1: tgt, 2: ENT_CLS, 3: REL_CLS
        tup_idx = torch.where((seq >= 0) & (seq < OFFSET))
        node_embeddings[tup_idx] = self.sp_token_embedding(seq[tup_idx])

        # rel idx
        tup_idx = torch.where(seq < 0)
        node_embeddings[tup_idx] = self.rel_embedding(torch.abs(seq[tup_idx]) - 1)

        return node_embeddings

    def forward(self, batch: Data, tgt_ent_idx=None):
        """
        """
        x = batch['x']

        node_embedding = self.init_seq_embedding(x)
        h, r, ws = self.query_encoder(node_embedding, graph=batch)

        t = self.kge_model(h, r)
        pred = self.kge_model.get_preds(t, tgt_ent_idx)

        return pred, ws

    def pred(self, batch, query_type):
        x = batch['x']
        node_embedding = self.init_seq_embedding(x)

        if query_type in ['1p', '2u-DNF'] and self.use_kge_to_pred_1p:
            # test 1p
            h = node_embedding[:, 2]
            r = node_embedding[:, 3]
            t = self.kge_model(h, r)

            pred = self.kge_model.get_preds(t)

        else:
            h, r, _ = self.query_encoder(node_embedding, graph=batch)
            t = self.kge_model(h, r)
            pred = self.kge_model.get_preds(t)

        return pred



class TokenEmbedding(nn.Module):
    def __init__(self, kwargs):
        super(TokenEmbedding, self).__init__()
        self.kwargs = kwargs
        # [1,2,3]
        if len(kwargs['token_embeddings']):
            self.token_embeds = [int(_) for _ in kwargs['token_embeddings'].split('.')]
        else:
            self.token_embeds = []
        self.hidden_size = kwargs['hidden_size']
        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.p_dropout = kwargs['hidden_dropout_prob']

        self.type_embeddings = nn.Embedding(2, self.hidden_size) if 1 in self.token_embeds else None
        self.layer_embeddings = nn.Embedding(8, self.hidden_size) if 2 in self.token_embeds else None
        self.op_embeddings = nn.Embedding(2, self.hidden_size) if 3 in self.token_embeds else None
        self.in_embeddings = nn.Embedding(8, self.hidden_size) if 4 in self.token_embeds else None
        self.out_embeddings = nn.Embedding(8, self.hidden_size) if 5 in self.token_embeds else None

        self.proj = nn.Linear(self.dim_ent_embedding, self.hidden_size)

        self.n_neg_proj = 1
        self.neg_proj = nn.ModuleList(
            [MLP(channel_list=[self.hidden_size, self.hidden_size])
             for _ in range(self.n_neg_proj)]
        )

        self.norm = nn.LayerNorm(self.hidden_size, eps=kwargs['layer_norm_eps'])
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, node_embeddings, graph):
        node_embeddings = self.proj(node_embeddings)

        if self.type_embeddings:
            node_embeddings += self.type_embeddings(graph['node_types'])

        if self.layer_embeddings:
            node_embeddings += self.layer_embeddings(graph['layers'])

        if self.op_embeddings:
            node_embeddings += self.op_embeddings(graph['operators'])

        if self.in_embeddings:
            node_embeddings += self.in_embeddings(graph['in_degs'])

        if self.out_embeddings:
            node_embeddings += self.out_embeddings(graph['out_degs'])

        for i in range(self.n_neg_proj):
            idxes = torch.where(graph['negs'] == i + 1)
            node_embeddings[idxes] = self.neg_proj[i](node_embeddings[idxes])

        node_embeddings = self.norm(node_embeddings)
        node_embeddings = self.dropout(node_embeddings)

        return node_embeddings


class BertEncoder(nn.Module):
    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

        self.dim_ent_embedding = kwargs['dim_ent_embedding']
        self.dim_rel_embedding = kwargs['dim_rel_embedding']
        self.hidden_size = kwargs['hidden_size']
        self.num_heads = kwargs['num_attention_heads']
        self.head_dim = kwargs['hidden_size'] // self.num_heads
        self.device = torch.device('cuda') if kwargs['cuda'] else torch.device('cpu')

        self.embedding = TokenEmbedding(kwargs)

        config = BertConfig(
            num_hidden_layers=kwargs['num_hidden_layers'],
            hidden_size=kwargs['hidden_size'],
            num_attention_heads=kwargs['num_attention_heads'],
            intermediate_size=kwargs['intermediate_size'],
            hidden_dropout_prob=kwargs['hidden_dropout_prob'],
            attention_probs_dropout_prob=kwargs['hidden_dropout_prob'],
            fp16=kwargs['fp16'],
            enc_dist=(kwargs['enc_dist'])
        )
        self.bert = BertModel(config)

        self.rev_proj1 = nn.Linear(self.hidden_size, self.dim_ent_embedding)
        self.rev_proj2 = nn.Linear(self.hidden_size, self.dim_rel_embedding)

    def forward(self, initial_node_embeddings,
                graph):
        # [b, l, dim]
        node_embeddings = self.embedding(initial_node_embeddings,
                                         graph)
        # node_embeddings = self.norm(node_embeddings)

        batch, length, dim = node_embeddings.shape

        hidden_states = self.bert(
            inputs_embeds=node_embeddings,
            attention_mask=graph['attention_mask'],
            dist_mat=graph['dist_mat'],
            negs=graph['negs']
        ).last_hidden_state

        # batch, hd
        cls1 = hidden_states[:, 0]
        cls2 = hidden_states[:, 1]
        # tgts = hidden_states[torch.where(graph['targets'] == 1)]

        h = self.rev_proj1(cls1)
        r = self.rev_proj2(cls2)

        return h, r, None


class KGE(nn.Module):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__()
        self.ent_embedding = None
        self.rel_embedding = None

        self.dim_rel_embedding = dim_rel_embedding
        self.dim_ent_embedding = dim_ent_embedding
        self.num_ents = num_ents
        self.num_rels = num_rels
        self.kwargs = kwargs
        self.init_size = 1e-3

    def forward(self, lhs, rel):
        raise NotImplemented

    def load_from_ckpt_path(self, ckpt_path):
        raise NotImplemented

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        raise NotImplemented

    @staticmethod
    def calc_preds(pred_embedding, ent_embedding, tgt_ent_idx=None):
        if tgt_ent_idx is None:
            # [dim, n_ent]
            tgt_ent_embedding = ent_embedding.weight.transpose(0, 1)

            # [n_batch, n_ent]
            scores = pred_embedding @ tgt_ent_embedding

        else:
            # [n_batch, neg, dim]
            tgt_ent_embedding = ent_embedding(tgt_ent_idx)

            # [n_batch, dim, 1]
            pred_embedding = pred_embedding.unsqueeze(-1)

            scores = torch.bmm(tgt_ent_embedding, pred_embedding)
            scores = scores.squeeze(-1)

        return scores


class Complex(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)
        # embedding and W
        self.rank = dim_ent_embedding // 2

        self.ent_embedding = nn.Embedding(num_ents, 2 * self.rank)
        self.rel_embedding = nn.Embedding(num_rels, 2 * self.rank)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

        self.embeddings = [self.ent_embedding, self.rel_embedding]

    def forward_emb(self, lhs, rel, to_score_idx=None):
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]

        if not to_score_idx:
            to_score = self.embeddings[0].weight
        else:
            to_score = self.embeddings[0](to_score_idx)

        to_score = to_score[:, :self.rank], to_score[:, self.rank:]
        return ((lhs[0] * rel[0] - lhs[1] * rel[1]) @ to_score[0].transpose(0, 1) +
                (lhs[0] * rel[1] + lhs[1] * rel[0]) @ to_score[1].transpose(0, 1))

    def forward(self, lhs, rel):
        lhs = torch.chunk(lhs, 2, -1)
        rel = torch.chunk(rel, 2, -1)

        output = ([lhs[0] * rel[0] - lhs[1] * rel[1], lhs[0] * rel[1] + lhs[1] * rel[0]])
        output = torch.cat(output, dim=-1)

        return output

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def get_factor(self, x):
        lhs = self.ent_embedding(x[0])
        rel = self.rel_embedding(x[1])
        rhs = self.ent_embedding(x[2])
        lhs = lhs[:, :self.rank], lhs[:, self.rank:]
        rel = rel[:, :self.rank], rel[:, self.rank:]
        rhs = rhs[:, :self.rank], rhs[:, self.rank:]
        return (torch.sqrt(lhs[0] ** 2 + lhs[1] ** 2),
                torch.sqrt(rel[0] ** 2 + rel[1] ** 2),
                torch.sqrt(rhs[0] ** 2 + rhs[1] ** 2))

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading Complex params from {ckpt_path}')

        try:
            self.embeddings[0].weight.data = params['embeddings.0.weight']
            self.embeddings[1].weight.data = params['embeddings.1.weight']
        except:
            self.embeddings[0].weight.data = params['_entity_embedding.weight']
            self.embeddings[1].weight.data = params['_relation_embedding.weight']

        self.ent_embedding_norm_mean = self.embeddings[0].weight.data.norm(p=2, dim=1).mean().item()
        self.rel_embedding_norm_mean = self.embeddings[1].weight.data.norm(p=2, dim=1).mean().item()

        self.embeddings[0].weight.data /= self.ent_embedding_norm_mean
        self.embeddings[1].weight.data /= self.rel_embedding_norm_mean


class TuckER(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.E = torch.nn.Embedding(num_ents, dim_ent_embedding)
        self.R = torch.nn.Embedding(num_rels, dim_rel_embedding)
        self.W = torch.nn.Parameter(
            torch.tensor(np.random.uniform(-1, 1, (dim_rel_embedding, dim_ent_embedding, dim_ent_embedding)),
                         dtype=torch.float))

        self.input_dropout = torch.nn.Dropout(0.3)
        self.hidden_dropout1 = torch.nn.Dropout(0.4)
        self.hidden_dropout2 = torch.nn.Dropout(0.5)

        self.bn0 = torch.nn.BatchNorm1d(dim_ent_embedding)
        self.bn1 = torch.nn.BatchNorm1d(dim_ent_embedding)

    def init(self):
        xavier_normal_(self.E.weight.data)
        xavier_normal_(self.R.weight.data)

    def forward(self, lhs, rel):
        x = self.bn0(lhs)
        x = self.input_dropout(x)
        x = x.view(-1, 1, self.dim_ent_embedding)

        W_mat = torch.mm(rel, self.W.view(self.dim_rel_embedding, -1))
        W_mat = W_mat.view(-1, self.dim_ent_embedding, self.dim_ent_embedding)
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)
        x = x.view(-1, self.dim_ent_embedding)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        return x

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.E, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path))

        self.ent_embedding = self.E
        self.rel_embedding = self.R


class CP(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)
        self.ent_embedding1 = nn.Embedding(num_ents, self.dim_ent_embedding)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size
        self.ent_embedding1.weight.data *= self.init_size

    def forward(self, lhs, rel):
        return lhs * rel

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding1, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading CP params from {ckpt_path}')

        self.ent_embedding.weight.data = params['lhs.weight']
        self.rel_embedding.weight.data = params['rel.weight']
        self.ent_embedding1.weight.data = params['rhs.weight']


class DistMult(KGE):
    def __init__(self, num_ents, num_rels,
                 dim_ent_embedding, dim_rel_embedding,
                 kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        self.ent_embedding = nn.Embedding(num_ents, self.dim_ent_embedding)
        self.rel_embedding = nn.Embedding(num_rels, self.dim_rel_embedding)

        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

    def forward(self, lhs, rel):
        return lhs * rel

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading DistMult params from {ckpt_path}')

        self.ent_embedding.weight.data = params['entity.weight']
        self.rel_embedding.weight.data = params['relation.weight']


class RESCAL(KGE):
    def __init__(self, num_ents, num_rels, dim_ent_embedding, dim_rel_embedding, kwargs):
        super().__init__(num_ents, num_rels,
                         dim_ent_embedding, dim_rel_embedding,
                         kwargs)

        assert dim_rel_embedding == dim_ent_embedding
        self.rank = dim_ent_embedding

        self.ent_embedding = nn.Embedding(num_ents, self.rank)
        self.rel_embedding = nn.Embedding(num_rels, self.rank * self.rank)
        self.ent_embedding.weight.data *= self.init_size
        self.rel_embedding.weight.data *= self.init_size

    def forward(self, lhs, rel):
        rel = rel.view(-1, self.rank, self.rank)
        lhs_proj = lhs.view(-1, 1, self.rank)
        lhs_proj = torch.bmm(lhs_proj, rel).view(-1, self.rank)
        return lhs_proj

    def get_preds(self, pred_embedding, tgt_ent_idx=None):
        return KGE.calc_preds(pred_embedding, self.ent_embedding, tgt_ent_idx)

    def load_from_ckpt_path(self, ckpt_path):
        params = torch.load(ckpt_path)
        logging.info(f'loading RESCAL params from {ckpt_path}')

        self.ent_embedding.weight.data = params['entity.weight']
        self.rel_embedding.weight.data = params['relation.weight']

# ============================================================
# __main__: Automated test suite for 5 ablated functions
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

    def tiny_kwargs(token_embeddings="1.2.3.4.5"):
        return {
            "cuda": False,
            "dim_ent_embedding": 8,
            "dim_rel_embedding": 8,
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-12,
            "token_embeddings": token_embeddings,
            "fp16": False,
            "enc_dist": "d",
            "geo": "complex",
            "kge_ckpt_path": None,
            "not_freeze_kge": True,
            "label_smoothing": 0.1,
        }

    def tiny_graph(batch=2, length=5):
        dist = torch.zeros(batch, length, length, dtype=torch.long)
        dist[:, :, -1] = 1
        return {
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
            "dist_mat": dist,
            "negs": torch.zeros(batch, length, dtype=torch.long),
            "node_types": torch.zeros(batch, length, dtype=torch.long),
            "layers": torch.arange(length).unsqueeze(0).repeat(batch, 1) % 8,
            "operators": torch.zeros(batch, length, dtype=torch.long),
            "in_degs": torch.arange(length).unsqueeze(0).repeat(batch, 1) % 8,
            "out_degs": torch.arange(length).flip(0).unsqueeze(0).repeat(batch, 1) % 8,
        }

    print("=" * 70)
    print("Q2T benchmark: Query2Triple core model checks")
    print("=" * 70)

    # ============================================================== 
    # Test 1/5: BertSelfAttention.forward
    # ============================================================== 
    print("-" * 60)
    print("[Test 1/5] BertSelfAttention.forward")
    try:
        cfg = BertConfig(
            hidden_size=8,
            num_attention_heads=2,
            intermediate_size=16,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            num_hidden_layers=1,
            fp16=False,
            enc_dist="d",
        )
        attention = BertSelfAttention(cfg)
        attention.eval()
        hidden = torch.randn(2, 4, 8)
        mask = torch.zeros(2, 1, 1, 4)
        mask[..., -1] = -10000.0
        dist = torch.zeros(2, 4, 4, dtype=torch.long)
        outputs = attention(hidden, attention_mask=mask, dist_mat=dist, output_attentions=True)
        check("attention output not None", outputs is not None and isinstance(outputs, tuple))
        if outputs is None or not isinstance(outputs, tuple):
            skip_checks(5, "attention output missing")
        else:
            context, probs = outputs[0], outputs[1]
            check("attention context shape", tuple(context.shape) == (2, 4, 8), f"got {tuple(context.shape)}")
            check("attention context finite", torch.isfinite(context).all().item())
            check("attention probs shape", tuple(probs.shape) == (2, 2, 4, 4), f"got {tuple(probs.shape)}")
            check("attention mask suppresses masked key", probs[..., -1].max().item() < 1e-4)
            with torch.no_grad():
                attention.dist_bias.weight.zero_()
                attention.dist_bias.weight[attention.offset + 1, 0] = 4.0
            biased_dist = torch.zeros(2, 4, 4, dtype=torch.long)
            biased_dist[:, :, -1] = 1
            unbiased = attention(hidden, dist_mat=torch.zeros_like(biased_dist), output_attentions=True)[1]
            biased = attention(hidden, dist_mat=biased_dist, output_attentions=True)[1]
            check("distance bias changes attention", not torch.allclose(unbiased[:, 0], biased[:, 0]))
    except Exception as exc:
        skip_checks(6, f"BertSelfAttention.forward raised {type(exc).__name__}: {exc}")

    # ============================================================== 
    # Test 2/5: TokenEmbedding.forward
    # ============================================================== 
    print("-" * 60)
    print("[Test 2/5] TokenEmbedding.forward")
    try:
        kwargs = tiny_kwargs()
        embedder = TokenEmbedding(kwargs)
        embedder.eval()
        graph = tiny_graph(batch=2, length=5)
        node_embeddings = torch.zeros(2, 5, 8)
        encoded = embedder(node_embeddings, graph)
        check("token embedding output not None", encoded is not None)
        if encoded is None:
            skip_checks(4, "token embedding output missing")
        else:
            check("token embedding shape", tuple(encoded.shape) == (2, 5, 8), f"got {tuple(encoded.shape)}")
            check("token embedding finite", torch.isfinite(encoded).all().item())
            graph_alt = tiny_graph(batch=2, length=5)
            graph_alt["node_types"] = torch.ones(2, 5, dtype=torch.long)
            encoded_alt = embedder(node_embeddings, graph_alt)
            check("type features affect embeddings", not torch.allclose(encoded, encoded_alt))
            loss = encoded.sum()
            loss.backward()
            grad_ok = embedder.proj.weight.grad is not None and torch.isfinite(embedder.proj.weight.grad).all().item()
            check("token embedding gradient finite", grad_ok)
    except Exception as exc:
        skip_checks(5, f"TokenEmbedding.forward raised {type(exc).__name__}: {exc}")

    # ============================================================== 
    # Test 3/5: BertEncoder.forward
    # ============================================================== 
    print("-" * 60)
    print("[Test 3/5] BertEncoder.forward")
    try:
        kwargs = tiny_kwargs()
        encoder = BertEncoder(kwargs)
        encoder.eval()
        graph = tiny_graph(batch=2, length=5)
        initial = torch.randn(2, 5, 8, requires_grad=True)
        result = encoder(initial, graph)
        check("query encoder output not None", result is not None)
        if result is None:
            skip_checks(4, "query encoder output missing")
        else:
            h, r, ws = result
            check("query encoder h shape", tuple(h.shape) == (2, 8), f"got {tuple(h.shape)}")
            check("query encoder r shape", tuple(r.shape) == (2, 8), f"got {tuple(r.shape)}")
            check("query encoder outputs finite", torch.isfinite(h).all().item() and torch.isfinite(r).all().item())
            (h.sum() + r.sum()).backward()
            check("query encoder backpropagates to inputs", initial.grad is not None and initial.grad.abs().sum().item() > 0)
    except Exception as exc:
        skip_checks(5, f"BertEncoder.forward raised {type(exc).__name__}: {exc}")

    # ============================================================== 
    # Test 4/5: Query2Triple.init_seq_embedding
    # ============================================================== 
    print("-" * 60)
    print("[Test 4/5] Query2Triple.init_seq_embedding")
    try:
        kwargs = tiny_kwargs(token_embeddings="0")
        model = Query2Triple(num_ents=6, num_rels=4, hidden_dim=8, edge_to_entities={}, **kwargs)
        seq = torch.tensor([[ENT_CLS, REL_CLS, OFFSET + 1, -2, VAR], [TGT, OFFSET + 2, -1, PAD, REL_CLS]])
        seq_embeddings = model.init_seq_embedding(seq)
        check("sequence embedding output not None", seq_embeddings is not None)
        if seq_embeddings is None:
            skip_checks(5, "sequence embedding output missing")
        else:
            check("sequence embedding shape", tuple(seq_embeddings.shape) == (2, 5, 8), f"got {tuple(seq_embeddings.shape)}")
            check("sequence embedding finite", torch.isfinite(seq_embeddings).all().item())
            check("entity ids use entity embeddings", torch.allclose(seq_embeddings[0, 2], model.ent_embedding.weight[1]))
            check("relation ids use relation embeddings", torch.allclose(seq_embeddings[0, 3], model.rel_embedding.weight[1]))
            check("special ids use special embeddings", torch.allclose(seq_embeddings[0, 0], model.sp_token_embedding.weight[ENT_CLS]))
    except Exception as exc:
        skip_checks(6, f"Query2Triple.init_seq_embedding raised {type(exc).__name__}: {exc}")

    # ============================================================== 
    # Test 5/5: Query2Triple.forward
    # ============================================================== 
    print("-" * 60)
    print("[Test 5/5] Query2Triple.forward")
    try:
        kwargs = tiny_kwargs()
        model = Query2Triple(num_ents=6, num_rels=4, hidden_dim=8, edge_to_entities={}, **kwargs)
        model.eval()
        batch = tiny_graph(batch=2, length=5)
        batch["x"] = torch.tensor([[ENT_CLS, REL_CLS, OFFSET + 1, -2, VAR], [TGT, OFFSET + 2, -1, PAD, REL_CLS]])
        pred, ws = model(batch)
        check("Query2Triple prediction not None", pred is not None)
        if pred is None:
            skip_checks(4, "Query2Triple prediction missing")
        else:
            check("Query2Triple prediction shape", tuple(pred.shape) == (2, 6), f"got {tuple(pred.shape)}")
            check("Query2Triple prediction finite", torch.isfinite(pred).all().item())
            check("Query2Triple auxiliary output preserved", ws is None)
            pred.sum().backward()
            grad_ok = model.query_encoder.rev_proj1.weight.grad is not None and model.query_encoder.rev_proj1.weight.grad.abs().sum().item() > 0
            check("Query2Triple gradients reach triple projection", grad_ok)
    except Exception as exc:
        skip_checks(5, f"Query2Triple.forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
