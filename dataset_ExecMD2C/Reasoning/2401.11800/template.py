import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing


# --- [Original file: DocRE-CLiP/code/link prediction/utils.py] ---
def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


# --- [Original file: DocRE-CLiP/code/model/attention.py] ---
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        scores = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v)

        return output, scores


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class QueryContextAttention(nn.Module):

    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, d_k, bias=False)
        self.w_ks = nn.Linear(d_k, d_k, bias=False)
        self.w_vs = nn.Linear(d_v, d_v, bias=False)
        self.fc = nn.Linear(d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v = self.d_k, self.d_v
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).unsqueeze(dim=-1)
        k = self.w_ks(k)
        v = self.w_vs(v)

        temperature = self.d_k ** 0.5
        scores = torch.matmul(k / temperature, q).transpose(1, 2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(scores, dim=-1))
        output = torch.matmul(attn, v).unsqueeze(-2)

        return output


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


# --- [Original file: DocRE-CLiP/code/model/graph.py] ---
class GraphConvolutionLayer(nn.Module):
    def __init__(self,edges,input_size,hidden_size,graph_drop):
        super(GraphConvolutionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.edges = edges
        self.W_edge = nn.ModuleList([nn.Linear(hidden_size,hidden_size,bias=False) for i in (self.edges)])
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)
        self.loop_weight = nn.Parameter(torch.Tensor(input_size, hidden_size))
        nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

    def forward(self, nodes_embed,node_adj):
        """
        [TODO] Apply relation-aware graph convolution over document graph nodes.

        Input:
            nodes_embed: (batch, nodes, input_size) - node representations.
            node_adj: (batch, nodes, nodes) - integer relation labels, where
                zero means no edge and positive labels select relation types.

        Output:
            (batch, nodes, hidden_size) - updated node representations.

"""
        pass


class GraphMultiHeadAttention(nn.Module):
    def __init__(self,edges,input_size,hidden_size,nhead=4,graph_drop=0.0):
        super(GraphMultiHeadAttention, self).__init__()
        assert hidden_size%nhead == 0
        ho = int(hidden_size/nhead)
        self.head_graph = nn.ModuleList([GraphAttentionLayer(edges,input_size,ho,graph_drop) for _ in range(nhead)])
        self.nhead = nhead
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)

    def forward(self, nodes_embed,node_adj):

        x = []
        for cnt in range(0, self.nhead):
            x.append(self.head_graph[cnt](nodes_embed,node_adj))
    
        return torch.cat(x,dim=-1)


class GraphAttentionLayer(nn.Module):
    def __init__(self,edges,input_size,hidden_size,graph_drop):
        super(GraphAttentionLayer, self).__init__()
        self.W = nn.Parameter(torch.Tensor(size=(input_size, hidden_size)))
        nn.init.xavier_uniform_(self.W, gain=nn.init.calculate_gain('relu'))
        self.edges = edges
        self.W_edge = nn.ModuleList([nn.Linear(2*hidden_size,1,bias=False) for i in (self.edges)])
        for m in self.W_edge:
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
        
        
        self.bias = nn.Parameter(torch.Tensor(hidden_size))
        nn.init.zeros_(self.bias)
        self.self_loop = False
        self.loop_weight = nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size
        nn.init.xavier_uniform_(self.loop_weight.weight, gain=nn.init.calculate_gain('relu'))
        self.drop = torch.nn.Dropout(p=graph_drop, inplace=False)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, nodes_embed,node_adj):
        """
        [TODO] Compute relation-aware graph attention for document graph nodes.

        Input:
            nodes_embed: (batch, nodes, input_size) - node representations.
            node_adj: (batch, nodes, nodes) - integer relation labels, with
                zero entries masked out as absent edges.

        Output:
            (batch, nodes, hidden_size) - attention-updated node states.

"""
        pass


# --- [Original file: DocRE-CLiP/code/model/DRN_original.py] ---
class GraphReasonLayer(nn.Module):
    def __init__(self,edges,input_size,out_size,iters,graph_type="gat",graph_drop=0.0,graph_head=4):
        super(GraphReasonLayer, self).__init__()
        # self.W = nn.Parameter(nn.init.normal_(torch.empty(input_size, input_size)), requires_grad=True)
        # self.W_node = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        # self.W_sum = nn.ModuleList([nn.Linear(input_size,input_size) for i in range(iters)])
        self.iters = iters
        self.edges = edges
        self.graph_type = graph_type
        if graph_type == "gat":
            self.block = nn.ModuleList([GraphAttentionLayer(edges,input_size,input_size,graph_drop=graph_drop) for i in range(iters)])
        elif graph_type == "gcn":
            self.block = nn.ModuleList([GraphConvolutionLayer(edges,input_size,input_size,graph_drop) for i in range(iters)])
        else:
            raise("graph choose error")
        
    def forward(self, nodes_embed,node_adj):
        """
        [TODO] Run iterative graph reasoning and concatenate all reasoning states.

        Input:
            nodes_embed: (batch, nodes, input_size) - initial graph node states.
            node_adj: (batch, nodes, nodes) - relation-labeled adjacency matrix.

        Output:
            (batch, nodes, input_size * (iters + 1)) - initial states followed
            by each iteration's updated states.

"""
        pass


# --- [Original file: DocRE-CLiP/code/model/taskdecompose.py] ---
class Task_Decompose(nn.Module):
    def __init__(self,config):
        super(Task_Decompose, self).__init__()

        self.config = config
        # pattern recongnize
        if config.use_dis_embed:
            self.dis_embed = nn.Embedding(20, 20)
            self.dis_sent_embed = nn.Embedding(20, 20)
            self.dis2idx = torch.zeros(512).cuda().long()
            self.dis2idx[1] = 1
            self.dis2idx[2:] = 2
            self.dis2idx[4:] = 3
            self.dis2idx[8:] = 4
            self.dis2idx[16:] = 5
            self.dis2idx[32:] = 6
            self.dis2idx[64:] = 7
            self.dis2idx[128:] = 8
            self.dis2idx[256:] = 9
            self.dis_size = 20

    def patten_recon(self,ins_path,path_info,graph_feature,context_feature):
        """
        [TODO] Build feature tensors for pattern-recognition relation paths.

        Input:
            ins_path: (batch, pairs, meta_paths, 4) - selected node indices for
                each candidate path, with all-zero rows treated as inactive.
            path_info: (batch, nodes, info_dim) - node metadata including token
                and sentence positions.
            graph_feature: (batch, nodes, graph_dim) - graph node features.
            context_feature: optional (batch, tokens, context_dim) - contextual
                token features.

        Output:
            (batch, pairs, meta_paths, feature_dim) - sparse path features.

"""
        pass

    def coreference_reason(self,ins_path,path_info,graph_feature,context_feature):
        """
        [TODO] Build feature tensors for coreference reasoning paths.

        Input:
            ins_path: (batch, pairs, meta_paths, 4) - path node indices.
            path_info: (batch, nodes, info_dim) - metadata for graph nodes.
            graph_feature: (batch, nodes, graph_dim) - graph node features.
            context_feature: optional (batch, tokens, context_dim) - token-level
                context features.

        Output:
            (batch, pairs, meta_paths, feature_dim) - sparse coreference-path
            features.

"""
        pass

    def logical_reason(self,ins_path,path_info,graph_feature,context_feature):
        """
        [TODO] Build feature tensors for logical multi-hop reasoning paths.

        Input:
            ins_path: (batch, pairs, meta_paths, 4) - four-node logical paths.
            path_info: (batch, nodes, info_dim) - metadata for graph nodes.
            graph_feature: (batch, nodes, graph_dim) - graph node features.
            context_feature: optional (batch, tokens, context_dim) - contextual
                token features.

        Output:
            (batch, pairs, meta_paths, feature_dim) - sparse logical-path
            features.

"""
        pass

    def forward(self,relation_path,path_info,graph_feature,context_feature=None):
        """
        [TODO] Decompose relation paths into pattern, coreference, and logical features.

        Input:
            relation_path: (batch, pairs, 3 * path_per_type, 4) - all candidate
                paths grouped by reasoning family.
            path_info: (batch, nodes, info_dim) - graph node metadata.
            graph_feature: (batch, nodes, graph_dim) - graph node features.
            context_feature: optional (batch, tokens, context_dim) - token
                context features.

        Output:
            path_fea: (batch, pairs, 3 * path_per_type, feature_dim) - features
                for all decomposed paths.
            mask: (batch, pairs, 3 * path_per_type) - active-path mask.

"""
        pass


# --- [Original file: DocRE-CLiP/code/link prediction/models.py] ---
class RGCN(torch.nn.Module):
    def __init__(self, num_entities, num_relations, num_bases, dropout):
        super(RGCN, self).__init__()

        self.entity_embedding = nn.Embedding(num_entities, 100)
        self.relation_embedding = nn.Parameter(torch.Tensor(num_relations, 100))

        nn.init.xavier_uniform_(self.relation_embedding, gain=nn.init.calculate_gain('relu'))

        self.conv1 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)
        self.conv2 = RGCNConv(
            100, 100, num_relations * 2, num_bases=num_bases)

        self.dropout_ratio = dropout

    def forward(self, entity, edge_index, edge_type, edge_norm):
        """
        [TODO] Encode entities with two relational graph convolution layers.

        Input:
            entity: (num_nodes,) - entity ids to embed.
            edge_index: (2, num_edges) - graph connectivity.
            edge_type: (num_edges,) - relation id for each edge.
            edge_norm: (num_edges,) - normalization coefficient for each edge.

        Output:
            (num_nodes, 100) - link-prediction entity embeddings.

"""
        pass

    def distmult(self, embedding, triplets):
        """
        [TODO] Score knowledge-graph triples with the DistMult compatibility function.

        Input:
            embedding: (num_entities, 100) - encoded entity embeddings.
            triplets: (num_triplets, 3) - subject, relation, object ids.

        Output:
            (num_triplets,) - one scalar score per triple.

"""
        pass

    def score_loss(self, embedding, triplets, target):
        score = self.distmult(embedding, triplets)

        return F.binary_cross_entropy_with_logits(score, target)

    def reg_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.relation_embedding.pow(2))


class RGCNConv(MessagePassing):
    r"""The relational graph convolutional operator from the `"Modeling
    Relational Data with Graph Convolutional Networks"
    <https://arxiv.org/abs/1703.06103>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \mathbf{\Theta}_{\textrm{root}} \cdot
        \mathbf{x}_i + \sum_{r \in \mathcal{R}} \sum_{j \in \mathcal{N}_r(i)}
        \frac{1}{|\mathcal{N}_r(i)|} \mathbf{\Theta}_r \cdot \mathbf{x}_j,

    where :math:`\mathcal{R}` denotes the set of relations, *i.e.* edge types.
    Edge type needs to be a one-dimensional :obj:`torch.long` tensor which
    stores a relation identifier
    :math:`\in \{ 0, \ldots, |\mathcal{R}| - 1\}` for each edge.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relations (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        root_weight (bool, optional): If set to :obj:`False`, the layer will
            not add transformed root node features to the output.
            (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
        :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels, num_relations, num_bases,
                 root_weight=True, bias=True, **kwargs):
        super(RGCNConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases

        self.basis = nn.Parameter(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = nn.Parameter(torch.Tensor(num_relations, num_bases))

        if root_weight:
            self.root = nn.Parameter(torch.Tensor(in_channels, out_channels))
        else:
            self.register_parameter('root', None)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        size = self.num_bases * self.in_channels
        uniform(size, self.basis)
        uniform(size, self.att)
        uniform(size, self.root)
        uniform(size, self.bias)

    def forward(self, x, edge_index, edge_type, edge_norm=None, size=None):
        """"""
        return self.propagate(edge_index, size=size, x=x, edge_type=edge_type,
                              edge_norm=edge_norm)

    def message(self, x_j, edge_index_j, edge_type, edge_norm):
        """
        [TODO] Construct relation-specific RGCN messages using basis decomposition.

        Input:
            x_j: optional (num_edges, in_channels) - source node features.
            edge_index_j: (num_edges,) - source node ids for embedding lookup mode.
            edge_type: (num_edges,) - relation type for each edge.
            edge_norm: optional (num_edges,) - per-edge normalization weights.

        Output:
            (num_edges, out_channels) - transformed edge messages.

"""
        pass

    def update(self, aggr_out, x):
        """
        [TODO] Add root node contribution and optional bias after aggregation.

        Input:
            aggr_out: (num_nodes, out_channels) - aggregated neighbor messages.
            x: optional (num_nodes, in_channels) - current node features.

        Output:
            (num_nodes, out_channels) - updated node states.

"""
        pass

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)


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

    print("=" * 70)
    print("RecLink / DocRE-CLiP benchmark: graph reasoning and KG link prediction")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test group 1: relation-aware graph convolution
    try:
        gcn = GraphConvolutionLayer(["intra", "inter"], 6, 6, 0.0).to(device).eval()
        nodes = torch.randn(2, 4, 6, device=device)
        adj = torch.tensor(
            [[[0, 1, 2, 0], [1, 0, 0, 2], [2, 0, 0, 1], [0, 2, 1, 0]],
             [[0, 2, 0, 1], [2, 0, 1, 0], [0, 1, 0, 2], [1, 0, 2, 0]]],
            device=device
        )
        out = gcn(nodes, adj)
        check("GraphConvolution output not None", out is not None)
        if out is not None:
            check("GraphConvolution output shape", tuple(out.shape) == (2, 4, 6), f"got {tuple(out.shape)}")
            check("GraphConvolution output finite", torch.isfinite(out).all().item())
            check("GraphConvolution non-negative after relu", (out >= 0).all().item())
            out.sum().backward()
            check("GraphConvolution relation weights receive grad", gcn.W_edge[0].weight.grad is not None)
        else:
            skip_checks(4, "GraphConvolution returned None")
    except Exception as exc:
        skip_checks(5, f"GraphConvolution raised {type(exc).__name__}: {exc}")

    # Test group 2: graph attention and multi-iteration graph reasoning
    try:
        gat = GraphAttentionLayer(["intra", "inter"], 6, 6, 0.0).to(device).eval()
        nodes = torch.randn(2, 4, 6, device=device)
        adj = torch.tensor(
            [[[0, 1, 2, 0], [1, 0, 0, 2], [2, 0, 0, 1], [0, 2, 1, 0]],
             [[0, 2, 0, 1], [2, 0, 1, 0], [0, 1, 0, 2], [1, 0, 2, 0]]],
            device=device
        )
        att_out = gat(nodes, adj)
        check("GraphAttention output not None", att_out is not None)
        if att_out is not None:
            check("GraphAttention output shape", tuple(att_out.shape) == (2, 4, 6), f"got {tuple(att_out.shape)}")
            check("GraphAttention output finite", torch.isfinite(att_out).all().item())
            check("GraphAttention residual changes values", not torch.allclose(att_out, nodes))
        else:
            skip_checks(3, "GraphAttention returned None")

        reason = GraphReasonLayer(["intra", "inter"], 6, 6, 2, graph_type="gcn", graph_drop=0.0).to(device).eval()
        reason_out = reason(nodes, adj)
        check("GraphReason output not None", reason_out is not None)
        if reason_out is not None:
            check("GraphReason concatenated shape", tuple(reason_out.shape) == (2, 4, 18), f"got {tuple(reason_out.shape)}")
            check("GraphReason preserves initial slice", torch.allclose(reason_out[..., :6], nodes, atol=1e-5))
            check("GraphReason output finite", torch.isfinite(reason_out).all().item())
        else:
            skip_checks(3, "GraphReason returned None")
    except Exception as exc:
        skip_checks(8, f"GraphAttention/GraphReason raised {type(exc).__name__}: {exc}")

    # Test group 3: task decomposition over three reasoning-path families
    try:
        cfg = SimpleNamespace(use_dis_embed=False, path_per_type=2)
        task = Task_Decompose(cfg).to(device).eval()
        relation_path = torch.zeros(2, 3, 6, 4, dtype=torch.long, device=device)
        relation_path[:, :, 0, :] = torch.tensor([0, 1, 2, 3], device=device)
        relation_path[:, :, 2, :] = torch.tensor([1, 2, 3, 4], device=device)
        relation_path[:, :, 4, :] = torch.tensor([2, 3, 4, 5], device=device)
        path_info = torch.zeros(2, 6, 6, dtype=torch.long, device=device)
        path_info[:, :, 0] = torch.arange(6, device=device)
        path_info[:, :, 5] = torch.arange(6, device=device)
        graph_feature = torch.randn(2, 6, 8, device=device)
        context_feature = torch.randn(2, 6, 5, device=device)
        path_fea, path_mask = task(relation_path, path_info, graph_feature, context_feature)
        check("Task_Decompose output not None", path_fea is not None and path_mask is not None)
        if path_fea is not None and path_mask is not None:
            check("Task_Decompose feature shape", tuple(path_fea.shape) == (2, 3, 6, 26), f"got {tuple(path_fea.shape)}")
            check("Task_Decompose mask shape", tuple(path_mask.shape) == (2, 3, 6), f"got {tuple(path_mask.shape)}")
            check("Task_Decompose feature finite", torch.isfinite(path_fea).all().item())
            check("Task_Decompose one active path per family", path_mask.sum(dim=-1).eq(3).all().item())
            check("Pattern path uses endpoint pair", torch.allclose(path_fea[:, :, 0, :16], torch.cat((graph_feature[:, 0].unsqueeze(1).expand(-1, 3, -1), graph_feature[:, 2].unsqueeze(1).expand(-1, 3, -1)), dim=-1), atol=1e-5))
            check("Coreference path placed in second family", path_mask[:, :, 2].all().item())
            check("Logical path placed in third family", path_mask[:, :, 4].all().item())
        else:
            skip_checks(7, "Task_Decompose returned None")
    except Exception as exc:
        skip_checks(8, f"Task_Decompose raised {type(exc).__name__}: {exc}")

    # Test group 4: RGCNConv basis-decomposed messages and root update
    try:
        conv = RGCNConv(4, 5, num_relations=3, num_bases=2).to(device)
        x_j = torch.randn(6, 4, device=device)
        edge_index_j = torch.tensor([0, 1, 2, 3, 1, 0], device=device)
        edge_type = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
        edge_norm = torch.ones(6, device=device)
        msg = conv.message(x_j, edge_index_j, edge_type, edge_norm)
        check("RGCNConv message not None", msg is not None)
        if msg is not None:
            check("RGCNConv message shape", tuple(msg.shape) == (6, 5), f"got {tuple(msg.shape)}")
            check("RGCNConv message finite", torch.isfinite(msg).all().item())
            check("RGCNConv basis relation layout", conv.att.shape == (3, 2) and conv.basis.shape == (2, 4, 5))
            updated = conv.update(torch.zeros(4, 5, device=device), torch.randn(4, 4, device=device))
            check("RGCNConv update shape", tuple(updated.shape) == (4, 5), f"got {tuple(updated.shape)}")
        else:
            skip_checks(4, "RGCNConv.message returned None")
    except Exception as exc:
        skip_checks(5, f"RGCNConv raised {type(exc).__name__}: {exc}")

    # Test group 5: RGCN link-prediction encoder and DistMult scoring
    try:
        rgcn = RGCN(num_entities=6, num_relations=3, num_bases=2, dropout=0.0).to(device).eval()
        entity = torch.arange(6, device=device)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]], device=device)
        edge_type = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
        edge_norm = torch.ones(6, device=device)
        emb = rgcn(entity, edge_index, edge_type, edge_norm)
        check("RGCN forward not None", emb is not None)
        if emb is not None:
            check("RGCN embedding shape", tuple(emb.shape) == (6, 100), f"got {tuple(emb.shape)}")
            check("RGCN embedding finite", torch.isfinite(emb).all().item())
            triplets = torch.tensor([[0, 0, 1], [2, 1, 3], [4, 2, 5]], device=device)
            score = rgcn.distmult(emb, triplets)
            check("RGCN DistMult shape", tuple(score.shape) == (3,), f"got {tuple(score.shape)}")
            check("RGCN DistMult finite", torch.isfinite(score).all().item())
            target = torch.tensor([1.0, 0.0, 1.0], device=device)
            loss = rgcn.score_loss(emb, triplets, target)
            check("RGCN score loss scalar", loss.dim() == 0)
            check("RGCN regularization positive", rgcn.reg_loss(emb).item() > 0)
        else:
            skip_checks(6, "RGCN.forward returned None")
    except Exception as exc:
        skip_checks(7, f"RGCN raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The benchmark implementation is complete.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
