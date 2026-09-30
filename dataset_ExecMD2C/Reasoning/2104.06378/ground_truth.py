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
        q: tensor of shape (n*b, d_k)
        k: tensor of shape (n*b, l, d_k)
        v: tensor of shape (n*b, l, d_v)

        returns: tensor of shape (n*b, d_v), tensor of shape(n*b, l)
        """
        attn = (q.unsqueeze(1) * k).sum(2)  # (n*b, l)
        attn = attn / self.temperature
        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = (attn.unsqueeze(2) * v).sum(1)
        return output, attn


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
        q: tensor of shape (b, d_q_original)
        k: tensor of shape (b, l, d_k_original)
        mask: tensor of shape (b, l) (optional, default None)
        returns: tensor of shape (b, n*d_v)
        """
        n_head, d_k, d_v = self.n_head, self.d_k, self.d_v

        bs, _ = q.size()
        bs, len_k, _ = k.size()

        qs = self.w_qs(q).view(bs, n_head, d_k)  # (b, n, dk)
        ks = self.w_ks(k).view(bs, len_k, n_head, d_k)  # (b, l, n, dk)
        vs = self.w_vs(k).view(bs, len_k, n_head, d_v)  # (b, l, n, dv)

        qs = qs.permute(1, 0, 2).contiguous().view(n_head * bs, d_k)
        ks = ks.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_k)
        vs = vs.permute(2, 0, 1, 3).contiguous().view(n_head * bs, len_k, d_v)

        if mask is not None:
            mask = mask.repeat(n_head, 1)
        output, attn = self.attention(qs, ks, vs, mask=mask)

        output = output.view(n_head, bs, d_v)
        output = output.permute(1, 0, 2).contiguous().view(bs, n_head * d_v)  # (b, n*dv)
        output = self.dropout(output)
        return output, attn


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
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        #Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype +1) #[E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype +1).to(edge_vec.device)
        self_edge_vec[:,self.n_etype] = 1

        head_type = node_type[edge_index[0]] #[E,] #head=src
        tail_type = node_type[edge_index[1]] #[E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype) #[E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype) #[E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1) #[E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype) #[N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1) #[N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0) #[E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0) #[E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1)) #[E+N, emb_dim]

        #Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  #[2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings) #[N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr): #i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key   = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]


        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2) #[E, heads]
        src_node_index = edge_index[0] #[E,]
        alpha = softmax(scores, src_node_index) #[E, heads] #group by src side node
        self._alpha = alpha

        #adjust by outgoing degree of src
        E = edge_index.size(1)            #n_edges
        N = int(src_node_index.max()) + 1 #n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index] #[E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1) #[E, heads]

        out = msg * alpha.view(-1, self.head_count, 1) #[E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  #[E, emb_dim]


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
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training = self.training)
        return _X


    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        #Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T)) #[batch_size, n_node, dim/2]

        #Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device) #[1,1,dim/2]
            js = torch.pow(1.1, js) #[1,1,dim/2]
            B = torch.sin(js * node_score) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score)) #[batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B)) #[batch_size, n_node, dim/2]


        X = H
        edge_index, edge_type = A #edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous() #[`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous() #[`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous() #[`total_n_nodes`, dim]

        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1) #[batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output


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
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)


        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]


        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) #pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))
        logits = self.fc(concat)
        return logits, pool_attn

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

