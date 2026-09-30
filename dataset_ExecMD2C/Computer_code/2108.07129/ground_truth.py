# ============================================================
# ground_truth.py - autoencoder_program_synthesis Core Model Components
# Source: Computer_Code/autoencoder_program_synthesis-main
#
# Contains only the tree VAE/RNN model components and direct model helpers.
# No training loops, datasets, CLI wrappers, checkpoint I/O, clang parsing, or BLEU evaluation.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from random import choice
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, Sequential
from torch.nn.functional import log_softmax
from typing import List, Sequence


# --- [Original file: autoencoder_program_synthesis/model_utils/distance_functions.py] ---
class CosineDistance(nn.Module):
    """
        Returns cosine distance between x1 and x2 computed along dim.
        Uses torch.nn.functional.cosine_similarity to calculate cosine similarity.
        The cosine distance is then defined as 1 - cosine similarity.
    """

    def __init__(self, dim: int =1, eps: float = 1e-8) -> None:
        super().__init__()

        self.dim = dim
        self.eps = eps

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return 1.0 - F.cosine_similarity(x1, x2, dim=self.dim, eps=self.eps)


# --- [Original file: autoencoder_program_synthesis/model_utils/modules.py] ---
class AddGate(nn.Module):
    """
        Add gate similar to LSTM add gate: y = sigmoid(W_mul * inp + b_mul) * tanh(W_add * inp + b_add)

        Outputs information that can be added to some state
        where the network learns: if and how much of the input should be added
    """

    def __init__(self, dim):
        super().__init__()

        self.W_mul = nn.Linear(dim, dim, bias=True)
        self.W_add = nn.Linear(dim, dim, bias=True)

        self.sigmoid = nn.Sigmoid()


    def forward(self, inp):
        out_mul = self.sigmoid(self.W_mul(inp))
        out_add = torch.tanh(self.W_add(inp))

        return out_mul * out_add


class PredictiveHidden(nn.Module):
    """
        Computes a combined predictive hidden state from two hidden states: y = tanh(W1 * x1 + W2 * x2)
    """

    def __init__(self, dim):
        super().__init__()

        # Learnable parameter weights1 -> for calculating: W1 * inp1
        self.W1 = nn.Linear(dim, dim, bias=True)

        # Learnable parameter weights2 -> for calculating: W2 * inp2
        self.W2 = nn.Linear(dim, dim, bias=True)


    def forward(self, inp1, inp2):
        # predictive hidden state: tanh(W1 * inp1 + W2 * inp2)
        h_pred = torch.tanh(self.W1(inp1) + self.W2(inp2))

        return h_pred


class TreeTopologyPred(nn.Module):
    """
        Computes logits for depth, width and res predictions with linear transformations: dim -> 1
    """

    def __init__(self, dim):
        super().__init__()

        # For topology prediction, we predict whether there are children
        self.depth_pred = nn.Linear(dim, 1)

        # For topology prediction, we predict whether there are successor siblings
        self.width_pred = nn.Linear(dim, 1)

        # For predicting whether a token is a reserved keyword of c++ or not
        self.res_pred = nn.Linear(dim, 1)

    def forward(self, inp):
        depth_pred = self.depth_pred(inp)
        width_pred = self.width_pred(inp)
        res_pred = self.res_pred(inp)

        return depth_pred, width_pred, res_pred


class LstmAttention(nn.Module):
    """
        ATTENTION-BASED LSTM FOR PSYCHOLOGICAL STRESS DETECTION FROM SPOKEN
        LANGUAGE USING DISTANT SUPERVISION

        https://arxiv.org/abs/1805.12307
    """

    def __init__(self, dim):
        super().__init__()

        self.attention_weights = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp):
        u = torch.tanh(self.attention_weights(inp))

        a = self.softmax(u)

        v = torch.sum(a * inp, dim=-1)

        return u * inp


class MultiLayerLSTMCell(nn.Module):
    """
        A long short-term memory (LSTM) cell with support for multiple layers.

        input_size: The number of expected features in the input
        hidden_size: The number of features in the hidden state
        num_layers: Number of recurrent layers.
                    E.g., setting num_layers=2 would mean stacking two LSTM cells together
                    to form a stacked LSTM cell, with the second LSTM cell taking in outputs of
                    the first LSTM cell and computing the final results. Default: 1
    """

    def __init__(self, input_size, hidden_size, num_layers = 1, recurrent_dropout=0):
        super().__init__()

        self.num_layers = num_layers
        self.rnns = nn.ModuleList([])
        self.dropout = nn.Dropout(recurrent_dropout)

        # Initialize RNNs with num layers
        for i in range(num_layers):
            if i == 0:
                self.rnns.append(nn.LSTMCell(input_size, hidden_size))
            else:
                self.rnns.append(nn.LSTMCell(hidden_size, hidden_size))


    def forward(self, input, hidden_states):
        new_hidden_states = []

        for i in range(self.num_layers):
            if i == 0:
                h, c = self.rnns[i](input, hidden_states[i])
            else:
                h, c = self.rnns[i](h, hidden_states[i])

            # apply recurrent dropout on the outputs of each LSTM cell hidden except the last layer
            if i < self.num_layers - 1:
                h = self.dropout(h)


            new_hidden_states.append((h, c))

        return new_hidden_states


# --- [Original file: autoencoder_program_synthesis/utils/Sampling.py] ---
class Sampling:
    @classmethod
    def _get_sample(cls, logits):
        """

        """

        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, 1)


    @classmethod
    def _filter_top_k(cls, logits, top_k):
        """
            Set logits below the top-k logits to -inf so softmax sets their probability to 0
        """

        # safety check
        top_k = min(top_k, logits.size(-1))

        if top_k > 0:
            # Get the indices of the logits below the top-k logits
            indices_below_top_k = logits < torch.topk(logits, top_k)[0][..., -1, None]

            # Set the logits below the top-k logits to -infinity so softmax gives them probability 0
            logits[indices_below_top_k] = float('-inf')

        return logits


    @classmethod
    def _filter_top_p(cls, logits, top_p):
        """
            Set logits above the top-p proability mass to -inf so softmax sets their probability to 0
        """

        if top_p > 0.0:
            # Sort the logits
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)

            # Run through softmax to get probabilities and then get the cumulative probabilities
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p

            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing and get the indices of the logits below the top-p logits
            indices_above_top_p = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)

            # Set the logits below the top-p logits to -infinity so softmax gives them probability 0
            logits[indices_above_top_p] = float('-inf')

        return logits


    @classmethod
    def sample(cls, logits, temperature=1.0, top_k=0, top_p=0.0):
        """
            Get sample from logits with temperature control and nucleus filtering (top-k and top-p)
        """

        if logits.shape[0] > 0:
            logits /= temperature if temperature > 0 else 1

            logits = cls._filter_top_k(logits, top_k)
            logits = cls._filter_top_p(logits, top_p)

            if temperature == 0: # greedy sampling:
                return torch.argmax(logits, dim=-1)
            else:
                return cls._get_sample(logits)
        else:
            # Return empty long tensor if logits are empty
            return torch.empty(0, dtype=torch.long, device=logits.device)


# --- [Original file: autoencoder_program_synthesis/models/tree_lstm.py] ---
class TreeLSTM(torch.nn.Module):
    '''PyTorch TreeLSTM model that implements efficient batching.
    '''
    def __init__(self, in_features, out_features):
        '''TreeLSTM class initializer

        Takes in int sizes of in_features and out_features and sets up model Linear network layers.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # bias terms are only on the W layers for efficiency
        self.W_iou = torch.nn.Linear(self.in_features, 3 * self.out_features)
        self.U_iou = torch.nn.Linear(self.out_features, 3 * self.out_features, bias=False)

        # f terms are maintained seperate from the iou terms because they involve sums over child nodes
        # while the iou terms do not
        self.W_f = torch.nn.Linear(self.in_features, self.out_features)
        self.U_f = torch.nn.Linear(self.out_features, self.out_features, bias=False)

    def forward(self, features, node_order, adjacency_list, edge_order, h, c):
        '''Run TreeLSTM model on a tree data structure with node features

        Takes Tensors encoding node features, a tree node adjacency_list, and the order in which
        the tree processing should proceed in node_order and edge_order.
        '''

        # populate the h and c states respecting computation order
        for n in range(node_order.max() + 1):
            self._run_lstm(n, h, c, features, node_order, adjacency_list, edge_order)

        return h, c

    def _run_lstm(self, iteration, h, c, features, node_order, adjacency_list, edge_order):
        '''Helper function to evaluate all tree nodes currently able to be evaluated.
        '''
        # N is the number of nodes in the tree
        # n is the number of nodes to be evaluated on in the current iteration
        # E is the number of edges in the tree
        # e is the number of edges to be evaluated on in the current iteration
        # F is the number of features in each node
        # M is the number of hidden neurons in the network

        # node_order is a tensor of size N x 1
        # edge_order is a tensor of size E x 1
        # features is a tensor of size N x F
        # adjacency_list is a tensor of size E x 2

        # node_mask is a tensor of size N x 1
        node_mask = node_order == iteration
        # edge_mask is a tensor of size E x 1
        edge_mask = edge_order == iteration

        # x is a tensor of size n x F
        x = features[node_mask, :]

        # At iteration 0 none of the nodes should have children
        # Otherwise, select the child nodes needed for current iteration
        # and sum over their hidden states
        if iteration == 0:
            iou = self.W_iou(x)
        else:
            # adjacency_list is a tensor of size e x 2
            adjacency_list = adjacency_list[edge_mask, :]

            # parent_indexes and child_indexes are tensors of size e x 1
            # parent_indexes and child_indexes contain the integer indexes needed to index into
            # the feature and hidden state arrays to retrieve the data for those parent/child nodes.
            parent_indexes = adjacency_list[:, 0]
            child_indexes = adjacency_list[:, 1]
            # child_h and child_c are tensors of size e x 1
            child_h = h[child_indexes, :]
            child_c = c[child_indexes, :]

            # Add child hidden states to parent offset locations
            _, child_counts = torch.unique_consecutive(parent_indexes, return_counts=True)
            child_counts = tuple(child_counts)

            parent_children = torch.split(child_h, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            h_sum = torch.stack(parent_list)
            iou = self.W_iou(x) + self.U_iou(h_sum)

        # i, o and u are tensors of size n x M
        i, o, u = torch.split(iou, iou.size(1) // 3, dim=1)
        i = torch.sigmoid(i)
        o = torch.sigmoid(o)
        u = torch.tanh(u)

        # At iteration 0 none of the nodes should have children
        # Otherwise, calculate the forget states for each parent node and child node
        # and sum over the child memory cell states
        if iteration == 0:
            c[node_mask, :] = i * u
        else:
            # f is a tensor of size e x M
            f = self.W_f(features[parent_indexes, :]) + self.U_f(child_h)
            f = torch.sigmoid(f)

            # fc is a tensor of size e x M
            fc = f * child_c

            # Add the calculated f values to the parent's memory cell state
            parent_children = torch.split(fc, child_counts)
            parent_list = [item.sum(0) for item in parent_children]

            c_sum = torch.stack(parent_list)
            c[node_mask, :] = i * u + c_sum

        h[node_mask, :] = o * torch.tanh(c[node_mask])


# --- [Original file: autoencoder_program_synthesis/models/TreeLstmEncoderComplete.py] ---
class TreeLstmEncoderComplete(nn.Module):
    def __init__(self,
                device,
                embedding_layers,
                embedding_dim,
                rnn_hidden_size,
                latent_dim,
                use_cell_output_lstm,
                vae,
                num_rnn_layers_enc,
                recurrent_dropout,
                indiv_embed_layers):

        super().__init__()

        self.device = device
        self.embedding_dim = embedding_dim
        self.rnn_hidden_size = rnn_hidden_size
        self.vae = vae
        self.num_rnn_layers_enc = num_rnn_layers_enc
        self.indiv_embed_layers = indiv_embed_layers
        self.use_cell_output_lstm = use_cell_output_lstm


        self.embedding_layers = embedding_layers
        self.tree_lstms = nn.ModuleList([])
        self.leaf_lstms = nn.ModuleDict({})
        self.attention = LstmAttention(rnn_hidden_size)

        self.recurrent_dropout = nn.Dropout(recurrent_dropout)

        for i in range(num_rnn_layers_enc):
            if i == 0:
                self.tree_lstms.append(TreeLSTM(embedding_dim, rnn_hidden_size))
            else:
                self.tree_lstms.append(TreeLSTM(rnn_hidden_size, rnn_hidden_size))


        if use_cell_output_lstm:
            rnn_hidden_size = rnn_hidden_size * 2
            latent_dim = latent_dim * 2

        self.z_mean = nn.Linear(rnn_hidden_size, latent_dim)
        self.z_log_var = nn.Linear(rnn_hidden_size, latent_dim)


    def forward(self, inp):
        batch_size = len(inp['tree_sizes'])
        node_order = inp['node_order_bottomup']
        adj_list = inp['adjacency_list']
        edge_order = inp['edge_order_bottomup']
        vocabs = inp['vocabs']
        total_nodes = node_order.shape[0]

        features = torch.zeros(total_nodes, self.embedding_dim, device=self.device)
        h = []
        c = []

        for _ in range(self.num_rnn_layers_enc):
            h.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))
            c.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))


        if self.indiv_embed_layers:
            for k, embedding_layer in self.embedding_layers.items():
                features_vocab = embedding_layer(inp['features'][vocabs == k].long()).view(-1, self.embedding_dim)
                features[vocabs == k] = features_vocab

        else:
            features_vocab = self.embedding_layers['ALL'](inp['features_combined'].long()).view(-1, self.embedding_dim)
            features = features_vocab

        for i in range(self.num_rnn_layers_enc):
            hidden, cell = self.tree_lstms[i](features,
                                        node_order,
                                        adj_list,
                                        edge_order,
                                        h[i],
                                        c[i])

            if i < self.num_rnn_layers_enc - 1:
                hidden = self.recurrent_dropout(hidden)

            features = hidden


        # hidden roots: (batch_size, hidden_size (*2 if using cell state of LSTM output))
        hidden_roots = torch.zeros(batch_size,
                                   self.rnn_hidden_size *2 if self.use_cell_output_lstm else self.rnn_hidden_size,
                                   device=self.device)

        hidden = self.attention(hidden)


        # Offset to check in hidden state, start at zero, increase by tree size each time
        # Example: hidden  [1, 3, 5, 1, 5, 2] and tree_sizes = [4, 2] we want hidden[0] and hidden[4] -> 1, 5
        # offset = 0
        # for i in range(len(inp['tree_sizes'])):
        #     if self.use_cell_output_lstm:
        #         hidden_roots[i] = torch.cat([hidden[offset], cell[offset]])
        #     else:
        #         hidden_roots[i] = hidden[offset]
        #     offset += inp['tree_sizes'][i]

        # tree_states = torch.cat([hidden, cell], dim=-1)

        for idx, tree_state in enumerate(torch.split(hidden, inp['tree_sizes'])):
            hidden_roots[idx] = torch.max(tree_state, dim=0)[0]
            # hidden_roots[idx] = tree_state[-1]

        # Get z_mean and z_log_var from hidden (parent roots only)
        z_mean = self.z_mean(hidden_roots)
        z_log_var = self.z_log_var(hidden_roots)


        # Parameterization trick
        z = self.reparameterize(z_mean, z_log_var)


        if self.vae:
            kl_loss = (0.5 * torch.sum(z_log_var.exp() - z_log_var - 1 + z_mean.pow(2)))
        else:
            kl_loss = torch.tensor([0], device=self.device)

        # Return latent vector, and mean and variance
        return z, kl_loss


    def reparameterize(self, mu, log_var):
        if self.training and self.vae:
            std = torch.exp(0.5 * log_var)
            eps = std.data.new(std.size()).normal_()

            return eps.mul(std).add_(mu)
        else:
            return mu


# --- [Original file: autoencoder_program_synthesis/model_utils/adaptive_softmax_pytorch.py] ---
_ASMoutput = namedtuple('_ASMoutput', ['output', 'loss'])


class AdaptiveLogSoftmaxWithLoss(Module):
    in_features: int
    n_classes: int
    cutoffs: List[int]
    div_value: float
    head_bias: bool
    head: Linear
    tail: ModuleList

    def __init__(
        self,
        in_features: int,
        n_classes: int,
        cutoffs: Sequence[int],
        div_value: float = 4.,
        head_bias: bool = False,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaptiveLogSoftmaxWithLoss, self).__init__()

        cutoffs = list(cutoffs)

        if (cutoffs != sorted(cutoffs)) \
                or (min(cutoffs) <= 0) \
                or (max(cutoffs) > (n_classes - 1)) \
                or (len(set(cutoffs)) != len(cutoffs)) \
                or any([int(c) != c for c in cutoffs]):

            raise ValueError("cutoffs should be a sequence of unique, positive "
                             "integers sorted in an increasing order, where "
                             "each value is between 1 and n_classes-1")

        self.in_features = in_features
        self.n_classes = n_classes
        self.cutoffs = cutoffs + [n_classes]
        self.div_value = div_value
        self.head_bias = head_bias

        self.shortlist_size = self.cutoffs[0]
        self.n_clusters = len(self.cutoffs) - 1
        self.head_size = self.shortlist_size + self.n_clusters

        self.head = Linear(self.in_features, self.head_size, bias=self.head_bias)
        self.tail = ModuleList()

        for i in range(self.n_clusters):

            hsz = int(self.in_features // (self.div_value ** (i + 1)))
            osz = self.cutoffs[i + 1] - self.cutoffs[i]

            projection = Sequential(
                Linear(self.in_features, hsz, bias=False),
                Linear(hsz, osz, bias=False),
            )

            self.tail.append(projection)

    def reset_parameters(self) -> None:
        self.head.reset_parameters()
        for i2h, h2o in self.tail:
            i2h.reset_parameters()
            h2o.reset_parameters()

    def forward(self, input: Tensor, target: Tensor) -> _ASMoutput:
        if input.size(0) != target.size(0):
            raise RuntimeError('Input and target should have the same size '
                               'in the batch dimension.')

        used_rows = 0
        batch_size = target.size(0)

        output = input.new_zeros(batch_size)
        gather_inds = target.new_empty(batch_size)

        cutoff_values = [0] + self.cutoffs
        for i in range(len(cutoff_values) - 1):

            low_idx = cutoff_values[i]
            high_idx = cutoff_values[i + 1]

            target_mask = (target >= low_idx) & (target < high_idx)
            row_indices = target_mask.nonzero().squeeze()

            if row_indices.numel() == 0:
                continue

            if i == 0:
                gather_inds.index_copy_(0, row_indices, target[target_mask])

            else:
                relative_target = target[target_mask] - low_idx
                input_subset = input.index_select(0, row_indices)

                cluster_output = self.tail[i - 1](input_subset)
                cluster_index = self.shortlist_size + i - 1

                gather_inds.index_fill_(0, row_indices, cluster_index)

                cluster_logprob = log_softmax(cluster_output, dim=1)
                local_logprob = cluster_logprob.gather(1, relative_target.unsqueeze(1))
                output.index_copy_(0, row_indices, local_logprob.squeeze(1))

            used_rows += row_indices.numel()

        if used_rows != batch_size:
            raise RuntimeError("Target values should be in [0, {}], "
                               "but values in range [{}, {}] "
                               "were found. ".format(self.n_classes - 1,
                                                     target.min().item(),
                                                     target.max().item()))

        head_output = self.head(input)
        head_logprob = log_softmax(head_output, dim=1)
        output += head_logprob.gather(1, gather_inds.unsqueeze(1)).squeeze()
        loss = (-output)

        return _ASMoutput(output, loss)

    def _get_full_log_prob(self, input, head_output):
        out = input.new_empty((head_output.size(0), self.n_classes))
        head_logprob = log_softmax(head_output, dim=1)

        out[:, :self.shortlist_size] = head_logprob[:, :self.shortlist_size]

        for i, (start_idx, stop_idx) in enumerate(zip(self.cutoffs, self.cutoffs[1:])):
            cluster_output = self.tail[i](input)
            cluster_logprob = log_softmax(cluster_output, dim=1)
            output_logprob = cluster_logprob + head_logprob[:, self.shortlist_size + i].unsqueeze(1)

            out[:, start_idx:stop_idx] = output_logprob

        return out

    def log_prob(self, input: Tensor) -> Tensor:
        head_output = self.head(input)
        return self._get_full_log_prob(input, head_output)


    def predict(self, input: Tensor) -> Tensor:
        head_output = self.head(input)
        output = torch.argmax(head_output, dim=1)
        not_in_shortlist = (output >= self.shortlist_size)
        all_in_shortlist = not (not_in_shortlist.any())

        if all_in_shortlist:
            return output

        elif not_in_shortlist.all():
            log_prob = self._get_full_log_prob(input, head_output)
            return torch.argmax(log_prob, dim=1)

        else:
            log_prob = self._get_full_log_prob(input[not_in_shortlist],
                                               head_output[not_in_shortlist])
            output[not_in_shortlist] = torch.argmax(log_prob, dim=1)
            return output


# --- [Original file: autoencoder_program_synthesis/models/TreeLstmDecoderComplete.py] ---
class TreeLstmDecoderComplete(nn.Module):
    def __init__(self,
                 device,
                 embedding_layers,
                 vocabulary,
                 loss_weights,
                 embedding_dim,
                 rnn_hidden_size,
                 latent_dim,
                 use_cell_output_lstm,
                 num_rnn_layers_dec,
                 dropout,
                 recurrent_dropout,
                 indiv_embed_layers,
                 max_name_tokens
                 ):
        super().__init__()

        self.device = device
        self.vocabulary = vocabulary
        self.loss_weights = loss_weights
        self.rnn_hidden_size = rnn_hidden_size
        self.latent_dim = latent_dim
        self.embedding_dim = embedding_dim
        self.use_cell_output_lstm = use_cell_output_lstm
        self.num_rnn_layers_dec = num_rnn_layers_dec
        self.indiv_embed_layers = indiv_embed_layers
        self.max_name_tokens = max_name_tokens

        # Shared embedding layers (shared with encoder)
        self.embedding_layers = embedding_layers

        # Latent to hidden layer -> transform z from latent dim to hidden size
        self.latent2hidden = nn.Linear(self.latent_dim, rnn_hidden_size)

        # Doubly recurrent network layers: parent and sibling RNNs
        self.rnns_parent = MultiLayerLSTMCell(embedding_dim, rnn_hidden_size, num_rnn_layers_dec, recurrent_dropout)
        self.rnns_sibling = MultiLayerLSTMCell(embedding_dim, rnn_hidden_size, num_rnn_layers_dec, recurrent_dropout)

        self.pred_hidden_state = PredictiveHidden(rnn_hidden_size)

        self.add_gate = AddGate(rnn_hidden_size)

        self.tree_topology_pred = TreeTopologyPred(rnn_hidden_size)

        # Learnable weights to incorporate topology information in label prediction
        self.offset_parent = nn.Linear(1, 1, bias=True)
        self.offset_sibling = nn.Linear(1, 1, bias=True)


        self.name_weights = nn.Linear(rnn_hidden_size, self.embedding_dim, bias=True)

        # Leaf lstms, if we want individual RNNs for leaf nodes
        self.leaf_lstms_sibling = nn.ModuleDict({})

        # Prediction layers for each vocab
        self.prediction_layers = nn.ModuleDict({})

        # Binary cross entropy loss for computing loss of topology
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='sum')

        # Loss functions for labels for each vocab so we can specify weights if needed
        self.label_losses = nn.ModuleDict({})

        self.dropout = nn.Dropout(dropout)


        # For evaluation/testing/generating:

        # sigmoid to predict topology of the tree (width and depth)
        self.sigmoid = nn.Sigmoid()

        # softmax to get probability distribution for labels of nodes
        self.softmax = nn.LogSoftmax(dim=-1)

        # Cosine similarity to get the similarity between name declarations and references
        self.dist_function = CosineDistance(dim=-1)

        self.triplet_loss = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function, reduction='sum', margin=1)

        # Initialize layers
        self.init_layers()


    def init_layers(self):
        # Initialize prediction layers
        for vocab_name in self.vocabulary.token_counts.keys():
            if vocab_name == 'LITERAL':
                # Adaptive softmax loss does not need prediction layer, much faster approach
                    # for calculating softmax with highly imbalanced vocabs
                    # cutoffs: 10: 71.1%, 11-100: 18.8%, 101-1000: 7.8%, rest: 2.2%
                    self.label_losses[vocab_name] = AdaptiveLogSoftmaxWithLoss(
                                                        self.rnn_hidden_size,
                                                        self.vocabulary.get_vocab_size(vocab_name),
                                                        cutoffs=[10, 100, 300],
                                                        div_value=3.0)

            elif vocab_name == 'NAME':
                self.label_losses[vocab_name] = None
            else:
                # Prediction layer
                self.prediction_layers[vocab_name] = nn.Linear(self.rnn_hidden_size, self.vocabulary.get_vocab_size(vocab_name))
                # cross entropy loss
                self.label_losses[vocab_name] = nn.CrossEntropyLoss(weight=self.loss_weights[vocab_name], reduction='sum')


    def forward(self, z, inp=None, names_token2index=None, temperature=None, top_k=None, top_p=None, generate=False):
        """
        @param z: (batch_size, LATENT_DIM) -> latent vector(s)

        @param target: dictionary containing tree information -> node_order_topdown, edge_order_topdown,
                                                  features, adjacency_list and vocabs

        @param idx_to_label: dictionary containing mapping from ids to labels, needed for evaluation
        """

        # We are training and we can do teacher forcing and batch processing
        if inp is not None:
            # Keep track of the loss
            total_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
            individual_losses = {}
            accuracies = {}
            loss_types = list(self.label_losses.keys()) + \
                ['PARENT', 'SIBLING', 'IS_RES']

            for loss_type in loss_types:
                individual_losses[loss_type] = 0
                accuracies[loss_type] = 0

            node_order = inp['node_order_topdown']

            # Total number of nodes of all trees in the batch
            total_nodes = node_order.shape[0]

            # Hidden and cell states of RNNs for parent and sibling
            h_p = []
            c_p = []

            h_s = []
            c_s = []

            for _ in range(self.num_rnn_layers_dec):
                # h and c states for every node in the batch for parent lstm
                h_p.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))
                c_p.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))

                # h and c states for every node in the batch for sibling lstm
                h_s.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))
                c_s.append(torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device))

            # Save predictive hidden states for name clustering
            # h_pred_states = torch.zeros(total_nodes, self.rnn_hidden_size, device=self.device)

            declared_names = [{} for _ in range(len(inp['tree_sizes']))]

            # Iterate over the levels of the tree -> top down
            for iteration in range(node_order.max() + 1):
                loss = self.decode_train(iteration, z, inp, h_p, c_p, h_s, c_s, declared_names, individual_losses, accuracies)
                total_loss += loss

            for loss_type in loss_types:
                if loss_type in ['PARENT', 'SIBLING', 'IS_RES']:
                    accuracies[loss_type] = accuracies[loss_type] / (total_nodes - z.shape[0])
                    # print(loss_type, accuracies[loss_type])
                else:
                    if loss_type == 'RES':
                        # Correct for root node -> is not predicted
                        accuracies[loss_type] = accuracies[loss_type] / (sum(inp['vocabs'] == loss_type) - z.shape[0])
                    elif loss_type == 'NAME':
                        accuracies[loss_type] = accuracies[loss_type] / (sum(inp['vocabs'] == loss_type) - sum([len(tree) for tree in declared_names]))
                    else:
                        accuracies[loss_type] = accuracies[loss_type] / sum(inp['vocabs'] == loss_type)

            return total_loss, individual_losses, accuracies


    def decode_train(self, iteration, z, data, h_p, c_p, h_s, c_s, declared_names, individual_losses, accuracies):
        loss = 0

        node_order = data['node_order_topdown']
        edge_order = data['edge_order_topdown']
        edge_order_sib = data['edge_order_topdown_sib']
        features = data['features']
        features_combined = data['features_combined']
        adjacency_list = data['adjacency_list'].cpu()
        adjacency_list_sib = data['adjacency_list_sib'].cpu()
        vocabs = np.array(data['vocabs'])
        tree_indices = data['tree_indices']

        if iteration == 0:
            hidden = self.latent2hidden(z)

            if self.use_cell_output_lstm:
                h_parent, c_parent = torch.split(hidden, int(hidden.shape[-1]/2), dim=-1)
            else:
                h_parent = hidden
                c_parent = torch.zeros((z.shape[0], self.rnn_hidden_size), device=self.device)


             # TODO this should be what we are supposed to do, but what we do below seems to be working better
            # emb_label = self.embedding_layers['RES'](tree_state.features[tree_state.node_order == 0]).view(-1, self.params['EMBEDDING_DIM'])

            # hidden_states = [(h_parent, c_parent) for _ in range(self.params['NUM_LSTM_LAYERS'])]
            # new_parent_states = self.rnns_parent(emb_label, hidden_states)

            # for idx, state in enumerate(new_parent_states):
            #     tree_state.h_p[idx][tree_state.node_order == 0] = state[0]
            #     tree_state.c_p[idx][tree_state.node_order == 0] = state[1]


            # FOR SOME REASON THIS SEEMS TO BE WORKING BETTER
            for i in range(self.num_rnn_layers_dec):
                # Update the hidden and cell values matrices
                h_p[i][node_order == 0] = h_parent
                c_p[i][node_order == 0] = c_parent


        else:
            # node_mask is a tensor of size N x 1
            node_mask = (node_order == iteration).cpu()
            # edge_mask is a tensor of size E x 1
            edge_mask = edge_order == iteration - 1

            edge_mask_sib = edge_order_sib == iteration - 1

            parent_indices = adjacency_list[edge_mask, :][:, 0]

            prev_sib_indices = adjacency_list_sib[edge_mask_sib, :][:, 0]

            curr_h_p = []
            curr_c_p = []

            for i in range(self.num_rnn_layers_dec):
                curr_h_p.append(h_p[i][parent_indices, :])
                curr_c_p.append(c_p[i][parent_indices, :])


            curr_h_s = []
            curr_c_s = []

            for i in range(self.num_rnn_layers_dec):
                curr_h_s.append(h_s[i][prev_sib_indices, :])
                curr_c_s.append(c_s[i][prev_sib_indices, :])


            # Combine hidden and cell states from last RNN layer
            if self.use_cell_output_lstm:
                parent_state = torch.cat([curr_h_p[-1], curr_c_p[-1]], dim=-1)
                sibling_state = torch.cat([curr_h_s[-1], curr_c_s[-1]], dim=-1)
            else:
                parent_state = curr_h_p[-1]
                sibling_state = curr_h_s[-1]

            h_pred = self.pred_hidden_state(parent_state, sibling_state)

            h_pred = self.dropout(h_pred)

            parent_state_updated = parent_state + self.add_gate(sibling_state)

            if self.use_cell_output_lstm:
                curr_h_p[-1], curr_c_p[-1] = torch.split(parent_state_updated, int(parent_state_updated.shape[-1]/2), dim=-1)
            else:
                curr_h_p[-1] = parent_state_updated

            depth_pred, width_pred, res_pred = self.tree_topology_pred(h_pred)

            is_parent = torch.tensor(np.isin(np.argwhere(node_mask == True), adjacency_list[:, 0]).flatten(), dtype=torch.float16, device=self.device).unsqueeze(-1)
            has_sibling = torch.tensor(np.isin(np.argwhere(node_mask == True), adjacency_list_sib[:, 0]).flatten(), dtype=torch.float16, device=self.device).unsqueeze(-1)
            is_res = torch.tensor(vocabs[node_mask] == 'RES', dtype=torch.float16, device=self.device).unsqueeze(-1)

            # Calculate parent and sibling loss
            parent_loss = self.bce_loss(depth_pred, is_parent)
            sibling_loss = self.bce_loss(width_pred, has_sibling)

            # calculate res prediction loss
            res_loss = self.bce_loss(res_pred, is_res)


            loss += parent_loss + sibling_loss + res_loss

            individual_losses['PARENT'] += parent_loss.item()
            individual_losses['SIBLING'] += sibling_loss.item()
            individual_losses['IS_RES'] += res_loss.item()

            accuracies['PARENT'] += torch.sum(
                (self.sigmoid(depth_pred) >= 0.5) == (is_parent == 1)).item()
            accuracies['SIBLING'] += torch.sum(
                (self.sigmoid(width_pred) >= 0.5) == (has_sibling == 1)).item()

            accuracies['IS_RES'] += torch.sum(
                (self.sigmoid(res_pred) >= 0.5) == (is_res == 1)).item()

            label = features[node_mask].long()


            # Iterate over possible node types and predict labels for each node type
            for k, prediction_layer in list(self.prediction_layers.items()) + [('LITERAL', '_'), ('NAME', '_')]:
                # Only do actual calculations when the amount of nodes for this node type > 0
                if len(h_pred[vocabs[node_mask] == k]) > 0:
                    # Get label predictions
                    if k == 'LITERAL':
                        # TODO look into how we can incorporate offset parent and offset sibling here (see below)
                        # or if it even influences our loss since we work with the vocab types so maybe we can remove it altogether
                        label_pred, label_loss = self.label_losses[k](
                            h_pred[vocabs[node_mask] == k], label[vocabs[node_mask] == k].view(-1))

                        accuracies[k] += torch.sum(self.label_losses[k].predict(h_pred[vocabs[node_mask] == k])
                                            == label[vocabs[node_mask] == k].view(-1)).item()


                        # Reduction is set to None for adaptive softmax with logit loss
                        label_loss = torch.sum(label_loss)

                        loss += label_loss
                        individual_losses[k] += label_loss.item()
                    elif k == 'NAME':
                        # Get the indices of the current name nodes we are evaluating
                        node_mask_indices = torch.where(torch.tensor(vocabs[node_mask] == k))[0]

                        # Loop over the name labels we are evaluating
                        for idx, l in zip(node_mask_indices, label[node_mask_indices]):
                            # Get the current node index (from all nodes, not just the selection)
                            current_node_idx = torch.where(node_mask)[0][idx]

                            # Get the declared names of the program we are evaluating
                            program_declared_names = declared_names[tree_indices[node_mask][idx]]

                            # Remove the declared names that are defined after the current node
                            program_declared_names_filt = {k: v for k, v in program_declared_names.items() if program_declared_names[k][0] < current_node_idx}

                            # If the name is already declared -> calculate loss: triplet between declaration as positive
                            #  and between other declarations as negative
                            if l.item() in program_declared_names_filt:
                                positive = self.name_weights(h_pred[idx]).unsqueeze(0)

                                if len(program_declared_names_filt) > 1:
                                    # create a name prediction state to make the h_pred of the name similar to the h_pred of the declaration
                                    anchor = program_declared_names_filt[l.item()][1]
                                    # negative = torch.stack([v[1] for k, v in program_declared_names_filt.items() if k != l.item()])
                                    negative = choice([v[1] for k, v in program_declared_names_filt.items() if k != l.item()]).unsqueeze(0)

                                    name_ref_loss = self.triplet_loss(anchor, positive, negative)

                                    loss += name_ref_loss
                                    individual_losses[k] += name_ref_loss.item()


                                distances = self.dist_function(positive, torch.stack([v[1] for v in program_declared_names_filt.values()]))
                                most_similar = list(program_declared_names_filt.keys())[torch.argmin(distances)]

                                # topk = min(3, len(program_declared_names))
                                # most_similar = torch.tensor(list(program_declared_names.keys()))[torch.topk(distances, k=topk, largest=False)[1]]
                                accuracies[k] +=  l.item() == most_similar
                            else:
                                program_declared_names[l.item()] = (current_node_idx, self.name_weights(h_pred[idx]))

                    else:
                        label_pred = prediction_layer(h_pred[vocabs[node_mask] == k])

                        # Calculate cross entropy loss of label prediction
                        label_loss = self.label_losses[k]((label_pred
                                                        # + self.offset_parent(is_parent[vocabs_mask == k])
                                                        # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                        ), label[vocabs[node_mask] == k].view(-1)) # / sum(tree_state.vocabs == k)


                        accuracies[k] += torch.sum(torch.argmax(self.softmax(label_pred
                                                                    # + self.offset_parent(is_parent[vocabs_mask == k])
                                                                    # + self.offset_sibling(has_sibling[vocabs_mask == k])
                                                                    ), dim=-1) == label[vocabs[node_mask] == k].view(-1)).item()

                        loss += label_loss #* max(-100 * iteration + 200, 1)

                        individual_losses[k] += label_loss.item()

                    if self.indiv_embed_layers:
                        self.update_rnn_state(h_p, c_p, h_s, c_s, features, k, curr_h_p, curr_c_p, curr_h_s, curr_c_s, vocabs, node_mask)

            if not self.indiv_embed_layers:
                self.update_rnn_state(h_p, c_p, h_s, c_s, features_combined, 'ALL', curr_h_p, curr_c_p, curr_h_s, curr_c_s, vocabs, node_mask)


        return loss


    def update_rnn_state(self, h_p, c_p, h_s, c_s, features, node_type, curr_h_p, curr_c_p, curr_h_s, curr_c_s, vocabs, node_mask):
        # Get true label values
        label = features[node_mask].long()

        if self.indiv_embed_layers:
            mask = vocabs[node_mask] == node_type
        else:
            mask = [True] * len(vocabs[node_mask])


        # Calculate embedding of true label -> teacher forcing
        emb_label = self.embedding_layers[node_type](label[mask]).view(-1, self.embedding_dim)


        parent_states = []
        sibling_states = []

        # Get the hidden and cell values of parents and previous siblings
        for i in range(self.num_rnn_layers_dec):
            parent_state_hidden = curr_h_p[i][mask]
            parent_state_cell = curr_c_p[i][mask]
            parent_states.append((parent_state_hidden, parent_state_cell))

            sibling_state_hidden = curr_h_s[i][mask]
            sibling_state_cell = curr_c_s[i][mask]
            sibling_states.append((sibling_state_hidden, sibling_state_cell))

        if 'RES' in node_type or node_type == 'ALL':
            # Compute hidden and cell values of current nodes
            new_parent_states = self.rnns_parent(emb_label, parent_states)
            new_sibling_states = self.rnns_sibling(emb_label, sibling_states)

        else:
            # Compute hidden and cell values of current nodes for previous siblings only (since we are not parents in the leafs)
            new_parent_states = [None for _ in range(self.num_rnn_layers_dec)]
            new_sibling_states = self.rnns_sibling(emb_label, sibling_states)


        # Update the hidden and cell values matrices
        state_mask = np.argwhere(node_mask == True).flatten()[mask]
        for idx, (p_state, s_state) in enumerate(zip(new_parent_states, new_sibling_states)):
            if 'RES' in node_type or node_type == 'ALL':
                h_p[idx][state_mask] = p_state[0]
                c_p[idx][state_mask] = p_state[1]

            h_s[idx][state_mask] = s_state[0]
            c_s[idx][state_mask] = s_state[1]


    def is_declared(self, current_name_sib_path, decl_name_sib_path):
        for cur, decl in zip(current_name_sib_path, decl_name_sib_path):
            if cur > decl:
                return True
            if cur < decl:
                return False


if __name__ == "__main__":
    from types import SimpleNamespace

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

    def make_encoder_batch(device):
        return {
            "tree_sizes": [3, 1],
            "node_order_bottomup": torch.tensor([1, 0, 0, 0], dtype=torch.long, device=device),
            "adjacency_list": torch.tensor([[0, 1], [0, 2]], dtype=torch.long, device=device),
            "edge_order_bottomup": torch.tensor([1, 1], dtype=torch.long, device=device),
            "vocabs": np.array(["ALL", "ALL", "ALL", "ALL"]),
            "features": torch.tensor([1, 2, 3, 1], dtype=torch.long, device=device),
            "features_combined": torch.tensor([1, 2, 3, 1], dtype=torch.long, device=device),
        }

    def make_decoder_batch(device):
        return {
            "tree_sizes": [2],
            "node_order_topdown": torch.tensor([0, 1], dtype=torch.long, device=device),
            "edge_order_topdown": torch.tensor([0], dtype=torch.long, device=device),
            "edge_order_topdown_sib": torch.tensor([0], dtype=torch.long, device=device),
            "features": torch.tensor([0, 1], dtype=torch.long, device=device),
            "features_combined": torch.tensor([0, 1], dtype=torch.long, device=device),
            "adjacency_list": torch.tensor([[0, 1]], dtype=torch.long, device=device),
            "adjacency_list_sib": torch.tensor([[0, 1]], dtype=torch.long, device=device),
            "vocabs": np.array(["RES", "RES"]),
            "tree_indices": torch.tensor([0, 0], dtype=torch.long, device=device),
        }

    class TinyVocabulary:
        def __init__(self):
            self.token_counts = {"RES": 2}
            self.token2index = {"RES": {"root": 0, "child": 1}}
            self.index2token = {"RES": {0: "root", 1: "child"}}

        def get_vocab_size(self, name):
            return self.token_counts[name]

    print("=" * 70)
    print("autoencoder_program_synthesis: Tree VAE/RNN benchmark")
    print("Automated Test Suite - 7 ablated functions")
    print("=" * 70)
    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    embedding_dim = 6
    hidden_dim = 5
    latent_dim = 4
    embedding_layers = nn.ModuleDict({"ALL": nn.Embedding(8, embedding_dim)})

    print("-" * 60)
    print("[Test 1/7] Sampling top-k and top-p filters")
    try:
        logits = torch.tensor([[0.1, 0.2, 0.3, 4.0, 5.0]], dtype=torch.float32)
        topk = Sampling._filter_top_k(logits.clone(), 2)
        check("top-k keeps expected count", torch.isfinite(topk).sum().item() == 2)
        check("top-k keeps largest logits", torch.isfinite(topk[0, 3]) and torch.isfinite(topk[0, 4]))
        topp = Sampling._filter_top_p(torch.tensor([[5.0, 4.0, 1.0, 0.5]], dtype=torch.float32), 0.75)
        check("top-p keeps at least one token", torch.isfinite(topp).sum().item() >= 1)
        check("top-p removes a tail token", torch.isneginf(topp).sum().item() >= 1)
    except Exception as exc:
        skip_checks(4, f"Sampling filters raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/7] TreeLSTM batched child-sum recurrence")
    try:
        tree_lstm = TreeLSTM(embedding_dim, hidden_dim).to(device)
        features = torch.randn(4, embedding_dim, device=device)
        node_order = torch.tensor([1, 0, 0, 0], dtype=torch.long, device=device)
        adjacency = torch.tensor([[0, 1], [0, 2]], dtype=torch.long, device=device)
        edge_order = torch.tensor([1, 1], dtype=torch.long, device=device)
        h0 = torch.zeros(4, hidden_dim, device=device)
        c0 = torch.zeros(4, hidden_dim, device=device)
        h, c = tree_lstm(features, node_order, adjacency, edge_order, h0, c0)
        check("TreeLSTM output not None", h is not None and c is not None)
        if h is not None and c is not None:
            check("TreeLSTM hidden shape", tuple(h.shape) == (4, hidden_dim), f"got {tuple(h.shape)}")
            check("TreeLSTM cell finite", torch.isfinite(c).all().item())
            check("TreeLSTM root updated", h[0].abs().sum().item() > 0)
            check("TreeLSTM leaf updated", h[1:].abs().sum().item() > 0)
        else:
            skip_checks(4, "TreeLSTM returned None")
    except Exception as exc:
        skip_checks(5, f"TreeLSTM raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/7] TreeLstmEncoderComplete forward pass")
    try:
        encoder = TreeLstmEncoderComplete(
            device=device,
            embedding_layers=embedding_layers,
            embedding_dim=embedding_dim,
            rnn_hidden_size=hidden_dim,
            latent_dim=latent_dim,
            use_cell_output_lstm=False,
            vae=True,
            num_rnn_layers_enc=1,
            recurrent_dropout=0.0,
            indiv_embed_layers=False,
        ).to(device)
        encoder.eval()
        z, kl_loss = encoder(make_encoder_batch(device))
        check("encoder output not None", z is not None and kl_loss is not None)
        if z is not None and kl_loss is not None:
            check("encoder latent shape", tuple(z.shape) == (2, latent_dim), f"got {tuple(z.shape)}")
            check("encoder latent finite", torch.isfinite(z).all().item())
            check("encoder KL scalar finite", kl_loss.dim() == 0 and torch.isfinite(kl_loss).item())
        else:
            skip_checks(3, "encoder returned None")
    except Exception as exc:
        skip_checks(4, f"encoder raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/7] TreeLstmEncoderComplete reparameterize semantics")
    try:
        encoder = TreeLstmEncoderComplete(device, embedding_layers, embedding_dim, hidden_dim, latent_dim, False, True, 1, 0.0, False)
        mu = torch.randn(3, latent_dim)
        log_var = torch.zeros(3, latent_dim)
        encoder.eval()
        z_eval = encoder.reparameterize(mu, log_var)
        check("reparameterize eval equals mean", torch.allclose(z_eval, mu))
        encoder.train()
        z_train = encoder.reparameterize(mu, log_var)
        check("reparameterize train shape", tuple(z_train.shape) == (3, latent_dim))
        check("reparameterize train finite", torch.isfinite(z_train).all().item())
    except Exception as exc:
        skip_checks(3, f"reparameterize raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/7] TreeLstmDecoderComplete training decode")
    try:
        vocab = TinyVocabulary()
        decoder = TreeLstmDecoderComplete(
            device=device,
            embedding_layers=embedding_layers,
            vocabulary=vocab,
            loss_weights={"RES": None},
            embedding_dim=embedding_dim,
            rnn_hidden_size=hidden_dim,
            latent_dim=latent_dim,
            use_cell_output_lstm=False,
            num_rnn_layers_dec=1,
            dropout=0.0,
            recurrent_dropout=0.0,
            indiv_embed_layers=False,
            max_name_tokens=4,
        ).to(device)
        z = torch.randn(1, latent_dim, device=device)
        loss, individual_losses, accuracies = decoder(z, make_decoder_batch(device))
        check("decoder loss not None", loss is not None)
        if loss is not None:
            check("decoder loss finite", torch.is_tensor(loss) and loss.dim() == 0 and torch.isfinite(loss).item())
            check("decoder individual loss keys", {"RES", "PARENT", "SIBLING", "IS_RES"}.issubset(individual_losses.keys()))
            check("decoder accuracies finite", all(np.isfinite(float(v)) for v in accuracies.values()))
        else:
            skip_checks(3, "decoder returned None")
    except Exception as exc:
        skip_checks(4, f"decoder raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 6/7] TreeLstmDecoderComplete update_rnn_state teacher forcing")
    try:
        vocab = TinyVocabulary()
        decoder = TreeLstmDecoderComplete(device, embedding_layers, vocab, {"RES": None}, embedding_dim, hidden_dim, latent_dim, False, 1, 0.0, 0.0, False, 4)
        h_p = [torch.zeros(2, hidden_dim)]
        c_p = [torch.zeros(2, hidden_dim)]
        h_s = [torch.zeros(2, hidden_dim)]
        c_s = [torch.zeros(2, hidden_dim)]
        curr_h_p = [torch.zeros(2, hidden_dim)]
        curr_c_p = [torch.zeros(2, hidden_dim)]
        curr_h_s = [torch.zeros(2, hidden_dim)]
        curr_c_s = [torch.zeros(2, hidden_dim)]
        features = torch.tensor([0, 1], dtype=torch.long)
        node_mask = torch.tensor([True, True])
        decoder.update_rnn_state(h_p, c_p, h_s, c_s, features, "ALL", curr_h_p, curr_c_p, curr_h_s, curr_c_s, np.array(["RES", "RES"]), node_mask)
        check("parent hidden updated", h_p[0].abs().sum().item() > 0)
        check("parent cell updated", c_p[0].abs().sum().item() > 0)
        check("sibling hidden updated", h_s[0].abs().sum().item() > 0)
        check("sibling cell updated", c_s[0].abs().sum().item() > 0)
    except Exception as exc:
        skip_checks(4, f"update_rnn_state raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 7/7] Tiny encoder-decoder integration")
    try:
        encoder = TreeLstmEncoderComplete(device, embedding_layers, embedding_dim, hidden_dim, latent_dim, False, True, 1, 0.0, False)
        decoder = TreeLstmDecoderComplete(device, embedding_layers, TinyVocabulary(), {"RES": None}, embedding_dim, hidden_dim, latent_dim, False, 1, 0.0, 0.0, False, 4)
        encoder.eval()
        z, kl_loss = encoder(make_encoder_batch(device))
        loss, _, _ = decoder(z[:1], make_decoder_batch(device))
        check("integration latent usable", z is not None and tuple(z.shape) == (2, latent_dim))
        check("integration losses finite", torch.isfinite(kl_loss).item() and torch.isfinite(loss).item())
    except Exception as exc:
        skip_checks(2, f"integration raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may not be implemented correctly.")
    print("=" * 70)
    if failed:
        raise SystemExit(1)
