"""
Standalone benchmark ground truth for QA-GNN core graph reasoning modules.

This file consolidates the model-side implementation needed to reproduce the
QAGNN decoder and its graph attention message passing, while intentionally
excluding pretrained language-model wrappers, dataloaders, training loops, and
dataset-specific preprocessing.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


def freeze_net(module):
    for p in module.parameters():
        p.requires_grad = False


def gelu(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


class MLP(nn.Module):
    """
    Multi-layer perceptron

    Parameters
    ----------
    num_layers: number of hidden layers
    """
    activation_classes = {'gelu': GELU, 'relu': nn.ReLU, 'tanh': nn.Tanh}

    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout, batch_norm=False,
                 init_last_layer_bias_to_zero=False, layer_norm=False, activation='gelu'):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.layer_norm = layer_norm

        assert not (self.batch_norm and self.layer_norm)

        self.layers = nn.Sequential()
        for i in range(self.num_layers + 1):
            n_in = self.input_size if i == 0 else self.hidden_size
            n_out = self.hidden_size if i < self.num_layers else self.output_size
            self.layers.add_module(f'{i}-Linear', nn.Linear(n_in, n_out))
            if i < self.num_layers:
                self.layers.add_module(f'{i}-Dropout', nn.Dropout(self.dropout))
                if self.batch_norm:
                    self.layers.add_module(f'{i}-BatchNorm1d', nn.BatchNorm1d(self.hidden_size))
                if self.layer_norm:
                    self.layers.add_module(f'{i}-LayerNorm', nn.LayerNorm(self.hidden_size))
                self.layers.add_module(f'{i}-{activation}', self.activation_classes[activation.lower()]())
        if init_last_layer_bias_to_zero:
            self.layers[-1].bias.data.fill_(0)

    def forward(self, input):
        return self.layers(input)


class MatrixVectorScaledDotProductAttention(nn.Module):

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, q, k, v, mask=None):
        """
        [TODO] Compute masked scaled attention from one query vector over a sequence.

        Input:
            q: (heads_times_batch, key_dim) - query vectors, one per flattened attention head and example.
            k: (heads_times_batch, sequence_length, key_dim) - key vectors for candidate nodes or tokens.
            v: (heads_times_batch, sequence_length, value_dim) - value vectors aligned with the keys.
            mask: (heads_times_batch, sequence_length) or None - True entries mark positions that must be ignored.

        Output:
            output: (heads_times_batch, value_dim) - pooled value representation.
            attention: (heads_times_batch, sequence_length) - normalized attention distribution.

"""
        pass



class MultiheadAttPoolLayer(nn.Module):

    def __init__(self, n_head, d_q_original, d_k_original, dropout=0.1):
        super().__init__()
        assert d_k_original % n_head == 0  # make sure the outpute dimension equals to d_k_origin
        self.n_head = n_head
        self.d_k = d_k_original // n_head
        self.d_v = d_k_original // n_head

        self.w_qs = nn.Linear(d_q_original, n_head * self.d_k)
        self.w_ks = nn.Linear(d_k_original, n_head * self.d_k)
        self.w_vs = nn.Linear(d_k_original, n_head * self.d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_q_original + self.d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_k_original + self.d_v)))

        self.attention = MatrixVectorScaledDotProductAttention(temperature=np.power(self.d_k, 0.5))
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, mask=None):
        """
        [TODO] Pool graph node states with a multi-head sentence-conditioned attention query.

        Input:
            q: (batch, query_dim) - sentence or context vectors used as pooling queries.
            k: (batch, num_nodes, node_dim) - node representations to be pooled.
            mask: (batch, num_nodes) or None - True entries mark graph nodes excluded from pooling.

        Output:
            output: (batch, node_dim) - concatenated pooled representation from all heads.
            attention: (num_heads * batch, num_nodes) - per-head node attention weights.

"""
        pass



class CustomizedEmbedding(nn.Module):
    def __init__(self, concept_num, concept_in_dim, concept_out_dim, use_contextualized=False,
                 pretrained_concept_emb=None, freeze_ent_emb=True, scale=1.0, init_range=0.02):
        super().__init__()
        self.scale = scale
        self.use_contextualized = use_contextualized
        if not use_contextualized:
            self.emb = nn.Embedding(concept_num, concept_in_dim)
            if pretrained_concept_emb is not None:
                self.emb.weight.data.copy_(pretrained_concept_emb)
            else:
                self.emb.weight.data.normal_(mean=0.0, std=init_range)
            if freeze_ent_emb:
                freeze_net(self.emb)

        if concept_in_dim != concept_out_dim:
            self.cpt_transform = nn.Linear(concept_in_dim, concept_out_dim)
            self.activation = GELU()

    def forward(self, index, contextualized_emb=None):
        """
        index: size (bz, a)
        contextualized_emb: size (bz, b, emb_size) (optional)
        """
        if contextualized_emb is not None:
            assert index.size(0) == contextualized_emb.size(0)
            if hasattr(self, 'cpt_transform'):
                contextualized_emb = self.activation(self.cpt_transform(contextualized_emb * self.scale))
            else:
                contextualized_emb = contextualized_emb * self.scale
            emb_dim = contextualized_emb.size(-1)
            return contextualized_emb.gather(1, index.unsqueeze(-1).expand(-1, -1, emb_dim))
        else:
            if hasattr(self, 'cpt_transform'):
                return self.activation(self.cpt_transform(self.emb(index) * self.scale))
            else:
                return self.emb(index) * self.scale


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """
    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        #For attention
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        #For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        """
        [TODO] Run edge-aware graph attention convolution for QA-GNN nodes.

        Input:
            x: (num_nodes, emb_dim) - current node hidden states.
            edge_index: (2, num_edges) - directed graph connectivity before self-loops are added.
            edge_type: (num_edges,) - relation type id for each original edge.
            node_type: (num_nodes,) - QA-GNN node type id for every node.
            node_feature_extra: (num_nodes, emb_dim) - encoded node-type and score features.
            return_attention_weights: bool - whether to expose augmented connectivity and attention weights.

        Output:
            If return_attention_weights is False: (num_nodes, emb_dim) updated node states.
            If return_attention_weights is True: updated node states plus augmented edges and per-head weights.

"""
        pass



    def message(self, edge_index, x_i, x_j, edge_attr): #i: tgt, j:src
        """
        [TODO] Compute edge-aware multi-head messages for each augmented graph edge.

        Input:
            edge_index: (2, augmented_edges) - directed source/target indices after self-loop insertion.
            x_i: (augmented_edges, 2 * emb_dim) - target-side node features for each edge.
            x_j: (augmented_edges, 2 * emb_dim) - source-side node features for each edge.
            edge_attr: (augmented_edges, emb_dim) - encoded relation and endpoint-type features.

        Output:
            messages: (augmented_edges, emb_dim) - flattened multi-head messages for aggregation.

"""
        pass



class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                    dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        self.basis_f = 'sin' #['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        self.edge_encoder = torch.nn.Sequential(torch.nn.Linear(n_etype +1 + n_ntype *2, hidden_size), torch.nn.BatchNorm1d(hidden_size), torch.nn.ReLU(), torch.nn.Linear(hidden_size, hidden_size))


        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])


        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout


    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        """
        [TODO] Apply the stacked QA-GNN graph attention layers.

        Input:
            _X: (batch * num_nodes, hidden_dim) - flattened node states.
            edge_index: (2, total_edges) - batched graph connectivity.
            edge_type: (total_edges,) - relation type ids for all batched edges.
            _node_type: (batch * num_nodes,) - flattened node type ids.
            _node_feature_extra: (batch * num_nodes, hidden_dim) - flattened type/score features.

        Output:
            updated: (batch * num_nodes, hidden_dim) - node states after all message-passing layers.

"""
        pass



    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        [TODO] Prepare node metadata features, run graph message passing, and fuse residual node states.

        Input:
            H: (batch, num_nodes, hidden_dim) - input node features.
            A: tuple(edge_index, edge_type) - batched graph connectivity and relation ids.
            node_type: (batch, num_nodes) - QA-GNN node type ids.
            node_score: (batch, num_nodes, 1) - normalized or raw scalar node relevance features.
            cache_output: bool - unused compatibility flag from the original interface.

        Output:
            output: (batch, num_nodes, hidden_dim) - updated node representations.

"""
        pass



class QAGNN(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02):
        super().__init__()
        self.init_range = init_range

        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb)
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()

        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                        input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim, dropout=p_gnn)

        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        if init_range > 0:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        [TODO] Run the full QA-GNN decoder from sentence vectors and concept graph inputs to logits.

        Input:
            sent_vecs: (batch, sent_dim) - language-model sentence representations.
            concept_ids: (batch, num_nodes) - concept identifiers, with the first position reserved for context.
            node_type_ids: (batch, num_nodes) - node type ids including context, question, answer, and other nodes.
            node_scores: (batch, num_nodes, 1) - scalar node relevance scores.
            adj_lengths: (batch,) - number of valid graph nodes per example.
            adj: tuple(edge_index, edge_type) - batched graph connectivity and relation ids.
            emb_data: optional contextualized embedding tensor for concept lookup.
            cache_output: bool - whether to store graph inputs and pool attention on the module.

        Output:
            logits: (batch, 1) - answer-choice score for each example.
            pool_attn: (num_heads * batch, num_nodes) - graph pooling attention over nodes.

"""
        pass


if __name__ == "__main__":
    from types import SimpleNamespace

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

    def batched_toy_graph(n_nodes):
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 5, 6, 7, 8, 5],
             [1, 2, 3, 1, 4, 6, 7, 8, 6, 9]],
            dtype=torch.long,
        )
        edge_type = torch.tensor([0, 1, 2, 1, 0, 0, 1, 2, 1, 0], dtype=torch.long)
        return edge_index, edge_type

    print("=" * 70)
    print("QA-GNN reproduction benchmark: core graph reasoning modules")
    print("Automated Test Suite - 7 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    print("-" * 60)
    print("[Test 1/5] MatrixVectorScaledDotProductAttention.forward")
    try:
        attention = MatrixVectorScaledDotProductAttention(temperature=2.0, attn_dropout=0.0).to(device)
        q = torch.randn(3, 4, device=device)
        k = torch.randn(3, 5, 4, device=device)
        v = torch.randn(3, 5, 6, device=device)
        mask = torch.tensor([[False, False, True, False, False],
                             [False, True, False, False, False],
                             [False, False, False, False, True]], device=device)
        output, weights = attention(q, k, v, mask=mask)
        check("matrix attention output not None", output is not None and weights is not None)
        if output is not None and weights is not None:
            check("matrix attention output shape", output.shape == (3, 6), f"got {tuple(output.shape)}")
            check("matrix attention weights shape", weights.shape == (3, 5), f"got {tuple(weights.shape)}")
            check("matrix attention output finite", torch.isfinite(output).all().item())
            check("matrix attention rows normalize", torch.allclose(weights.sum(dim=1), torch.ones(3, device=device), atol=1e-5))
            check("matrix attention mask respected", torch.all(weights[mask] < 1e-6).item())
        else:
            skip_checks(5, "matrix attention returned None")
    except Exception as exc:
        skip_checks(6, f"matrix attention raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/5] MultiheadAttPoolLayer.forward")
    try:
        pool = MultiheadAttPoolLayer(n_head=2, d_q_original=6, d_k_original=8, dropout=0.0).to(device)
        pool.eval()
        query = torch.randn(2, 6, device=device)
        key = torch.randn(2, 4, 8, device=device)
        pool_mask = torch.tensor([[False, True, False, False], [False, False, False, True]], device=device)
        pooled, pool_attn = pool(query, key, mask=pool_mask)
        check("multihead pool output not None", pooled is not None and pool_attn is not None)
        if pooled is not None and pool_attn is not None:
            check("multihead pool output shape", pooled.shape == (2, 8), f"got {tuple(pooled.shape)}")
            check("multihead pool attention shape", pool_attn.shape == (4, 4), f"got {tuple(pool_attn.shape)}")
            check("multihead pool output finite", torch.isfinite(pooled).all().item())
            check("multihead pool heads normalize", torch.allclose(pool_attn.sum(dim=1), torch.ones(4, device=device), atol=1e-5))
            check("multihead pool mask repeated per head", torch.all(pool_attn[pool_mask.repeat(2, 1)] < 1e-6).item())
        else:
            skip_checks(5, "multihead pooling returned None")
    except Exception as exc:
        skip_checks(6, f"multihead pooling raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/5] GATConvE.forward and GATConvE.message")
    try:
        args = SimpleNamespace()
        emb_dim, n_ntype, n_etype = 8, 4, 3
        edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_etype + 1 + n_ntype * 2, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        ).to(device)
        conv = GATConvE(args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=2).to(device)
        conv.eval()
        x = torch.randn(5, emb_dim, device=device, requires_grad=True)
        edge_index = torch.tensor([[0, 1, 2, 3, 0, 4], [1, 2, 3, 1, 4, 0]], dtype=torch.long, device=device)
        edge_type = torch.tensor([0, 1, 2, 1, 0, 2], dtype=torch.long, device=device)
        node_type = torch.tensor([3, 0, 1, 2, 2], dtype=torch.long, device=device)
        node_extra = torch.randn(5, emb_dim, device=device)
        result = conv(x, edge_index, edge_type, node_type, node_extra, return_attention_weights=True)
        check("GATConvE result not None", result is not None)
        if result is not None:
            out, (returned_edge_index, alpha) = result
            check("GATConvE output shape", out.shape == (5, emb_dim), f"got {tuple(out.shape)}")
            check("GATConvE augmented edge shape", returned_edge_index.shape == (2, edge_index.size(1) + x.size(0)), f"got {tuple(returned_edge_index.shape)}")
            check("GATConvE alpha shape", alpha.shape == (edge_index.size(1) + x.size(0), 2), f"got {tuple(alpha.shape)}")
            check("GATConvE output finite", torch.isfinite(out).all().item())
            per_source_ok = True
            for src in torch.unique(returned_edge_index[0]):
                src_mask = returned_edge_index[0] == src
                per_source_ok = per_source_ok and torch.allclose(alpha[src_mask].sum(dim=0), torch.ones(2, device=device), atol=1e-5)
            check("GATConvE attention normalizes by source", per_source_ok)
            out.sum().backward()
            check("GATConvE input gradient finite", x.grad is not None and torch.isfinite(x.grad).all().item())
            check("GATConvE key projection receives gradient", conv.linear_key.weight.grad is not None)
        else:
            skip_checks(7, "GATConvE returned None")
    except Exception as exc:
        skip_checks(8, f"GATConvE raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/5] QAGNN_Message_Passing.mp_helper and forward")
    try:
        args = SimpleNamespace()
        mp = QAGNN_Message_Passing(args, k=1, n_ntype=4, n_etype=3, input_size=8, hidden_size=8, output_size=8, dropout=0.0).to(device)
        mp.eval()
        H = torch.randn(2, 5, 8, device=device, requires_grad=True)
        node_type = torch.tensor([[3, 0, 1, 2, 2], [3, 1, 0, 2, 2]], dtype=torch.long, device=device)
        node_score = torch.randn(2, 5, 1, device=device)
        adj = tuple(t.to(device) for t in batched_toy_graph(n_nodes=5))
        output = mp(H, adj, node_type, node_score)
        check("message passing output not None", output is not None)
        if output is not None:
            check("message passing output shape", output.shape == (2, 5, 8), f"got {tuple(output.shape)}")
            check("message passing output finite", torch.isfinite(output).all().item())
            check("message passing transforms states", not torch.allclose(output, H))
            output.sum().backward()
            check("message passing input gradient finite", H.grad is not None and torch.isfinite(H.grad).all().item())
            check("message passing residual projections receive gradients", mp.Vh.weight.grad is not None and mp.Vx.weight.grad is not None)
        else:
            skip_checks(5, "message passing returned None")
    except Exception as exc:
        skip_checks(6, f"message passing raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/5] QAGNN.forward")
    try:
        args = SimpleNamespace()
        model = QAGNN(
            args=args,
            k=1,
            n_ntype=4,
            n_etype=3,
            sent_dim=10,
            n_concept=20,
            concept_dim=8,
            concept_in_dim=8,
            n_attention_head=2,
            fc_dim=16,
            n_fc_layer=1,
            p_emb=0.0,
            p_gnn=0.0,
            p_fc=0.0,
            freeze_ent_emb=False,
            init_range=0.02,
        ).to(device)
        model.eval()
        sent_vecs = torch.randn(2, 10, device=device, requires_grad=True)
        concept_ids = torch.tensor([[0, 1, 2, 3, 4], [0, 2, 3, 4, 5]], dtype=torch.long, device=device)
        node_type_ids = torch.tensor([[3, 0, 1, 2, 2], [3, 1, 0, 2, 2]], dtype=torch.long, device=device)
        node_scores = torch.tensor(
            [[[0.0], [0.4], [0.8], [1.2], [1.6]],
             [[0.0], [0.5], [0.9], [1.3], [1.7]]],
            dtype=torch.float32,
            device=device,
        )
        adj_lengths = torch.tensor([5, 4], dtype=torch.long, device=device)
        adj = tuple(t.to(device) for t in batched_toy_graph(n_nodes=5))
        result = model(sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, cache_output=True)
        check("QAGNN result not None", result is not None)
        if result is not None:
            logits, pool_attn = result
            check("QAGNN logits shape", logits.shape == (2, 1), f"got {tuple(logits.shape)}")
            check("QAGNN pool attention shape", pool_attn.shape == (4, 5), f"got {tuple(pool_attn.shape)}")
            check("QAGNN logits finite", torch.isfinite(logits).all().item())
            check("QAGNN attention finite", torch.isfinite(pool_attn).all().item())
            check("QAGNN attention rows normalize", torch.allclose(pool_attn.sum(dim=1), torch.ones(4, device=device), atol=1e-5))
            check("QAGNN masks context node", torch.all(pool_attn[:, 0] < 1e-6).item())
            check("QAGNN masks padded node", torch.all(pool_attn[[1, 3], 4] < 1e-6).item())
            check("QAGNN caches concept ids", torch.equal(model.concept_ids, concept_ids))
            check("QAGNN caches adjacency", model.adj[0] is adj[0] and model.adj[1] is adj[1])
            check("QAGNN caches pool attention", model.pool_attn is pool_attn)
            logits.sum().backward()
            check("QAGNN sentence gradient finite", sent_vecs.grad is not None and torch.isfinite(sent_vecs.grad).all().item())
            check("QAGNN classifier receives gradient", model.fc.layers[-1].weight.grad is not None)
        else:
            skip_checks(12, "QAGNN returned None")
    except Exception as exc:
        skip_checks(13, f"QAGNN raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
        raise SystemExit(1)

