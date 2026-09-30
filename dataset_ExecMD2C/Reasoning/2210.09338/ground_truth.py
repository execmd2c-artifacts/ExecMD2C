import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter


# --- [Original file: modeling/modeling_gnn.py] ---
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
        x: [N, emb_dim]
        edge_index: [2, E]
        edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39]
        node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        node_feature_extra [N, dim]
        """

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
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key   = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head) #[E, heads, _dim]

        if self.args.fp16 and self.training and self.args.upcast:
            with torch.cuda.amp.autocast(enabled=False):
                query = query.float() / math.sqrt(self.dim_per_head)
                scores = (query * key.float()).sum(dim=2) #[E, heads]
        else:
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



class Decoder(nn.Module):
    def __init__(self, args, num_rels, h_dim):
        super().__init__()
        self.args = args
        self.num_relations = num_rels
        self.embedding_dim = h_dim
        # nn.init.xavier_uniform_(self.w_relation,
        #                         gain=nn.init.calculate_gain('relu'))

        self.negative_adversarial_sampling = args.link_negative_adversarial_sampling
        self.adversarial_temperature = args.link_negative_adversarial_sampling_temperature
        self.reg_param = args.link_regularizer_weight


    def forward(self, embs, sample, mode='single'):
        """
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        """

        if mode == 'single':
            batch_size, negative_sample_size = sample[0].shape[0], 1

            head = embs[sample[0]].unsqueeze(1) #[n_triple, 1, dim]
            relation = self.w_relation[sample[1]].unsqueeze(1) #[n_triple, 1, dim]
            tail = embs[sample[2]].unsqueeze(1) #[n_triple, 1, dim]

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.shape

            head = embs[head_part] #[n_triple, n_neg, dim]
            relation = self.w_relation[tail_part[1]].unsqueeze(1) #[n_triple, 1, dim]
            tail = embs[tail_part[2]].unsqueeze(1) #[n_triple, 1, dim]

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.shape

            head = embs[head_part[0]].unsqueeze(1)
            relation = self.w_relation[head_part[1]].unsqueeze(1)

            tail = embs[tail_part]

        else:
            raise ValueError('mode %s not supported' % mode)

        score = self.score(head, relation, tail, mode) #[n_triple, 1 or n_neg]

        return score

    def score(self, h, r, t, mode):
        raise NotImplementedError

    def reg_loss(self):
        return torch.mean(self.w_relation.pow(2))
        # return torch.tensor(0)

    def loss(self, scores):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        positive_score, negative_score = scores
        if self.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * self.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1) #[n_triple,]

        positive_score = F.logsigmoid(positive_score).squeeze(dim=1) #[n_triple,]

        assert positive_score.dim() == 1
        if len(positive_score) == 0:
            positive_sample_loss = negative_sample_loss = 0.
        else:
            positive_sample_loss = - positive_score.mean() #scalar
            negative_sample_loss = - negative_score.mean() #scalar

        loss = (positive_sample_loss + negative_sample_loss) / 2 + self.reg_param * self.reg_loss()

        return loss, positive_sample_loss, negative_sample_loss


class TransEDecoder(Decoder):
    """TransE score function
    Paper link: https://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data
    """

    def __init__(self, args, num_rels, h_dim, dist_func='l2'):
        super().__init__(args, num_rels, h_dim)

        self.gamma = self.args.link_gamma
        if dist_func == 'l1':
            dist_ord = 1
        else:  # default use l2
            dist_ord = 2
        self.dist_ord = dist_ord

        print (f"Initializing w_relation for TransEDecoder... (gamma={self.gamma})", file=sys.stderr)
        self.epsilon = 2.0
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.embedding_range = (self.gamma + self.epsilon) / self.embedding_dim
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)


    def score(self, head, relation, tail, mode):
        """
        Input head/tail has stdev 1 for each element. Scale to stdev 1/sqrt(12) * (b-a) = a/sqrt(3).
        Reference: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/wikikg2/model.py
        """
        head = head * self.embedding_range / math.sqrt(3.0)
        tail = tail * self.embedding_range / math.sqrt(3.0)

        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma - torch.norm(score, p=self.dist_ord, dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={}, gamma={}, dist_ord={})'.format(self.__class__.__name__,
                                                                                       self.embedding_dim,
                                                                                       self.num_relations,
                                                                                       self.gamma,
                                                                                       self.dist_ord)


class DistMultDecoder(Decoder):
    """DistMult score function
        Paper link: https://arxiv.org/abs/1412.6575
    """
    def __init__(self, args, num_rels, h_dim):
        super().__init__(args, num_rels, h_dim)

        print ("Initializing w_relation for DistMultDecoder...", file=sys.stderr)
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim)))
        self.embedding_range = math.sqrt(1.0 / self.embedding_dim)
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)


    def score(self, head, relation, tail, mode):
        if mode == 'head-batch':
            if self.args.scaled_distmult:
                tail = tail / math.sqrt(self.embedding_dim)
            score = head * (relation * tail)
        else:
            if self.args.scaled_distmult:
                head = head / math.sqrt(self.embedding_dim)
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={})'.format(self.__class__.__name__,
                                                                self.embedding_dim,
                                                                self.num_relations)


class RotatEDecoder(Decoder):
    """RotatE score function
    Paper link: https://arxiv.org/pdf/1902.10197.pdf
    """

    def __init__(self, args, num_rels, h_dim):
        super().__init__(args, num_rels, h_dim)

        self.gamma = self.args.link_gamma

        print (f"Initializing w_relation for RotatEDecoder... (gamma={self.gamma})", file=sys.stderr)
        self.epsilon = 2.0
        self.register_parameter('w_relation', nn.Parameter(torch.Tensor(self.num_relations, self.embedding_dim //2)))
        self.embedding_range = (self.gamma + self.epsilon) / self.embedding_dim
        with torch.no_grad():
            self.w_relation.uniform_(-self.embedding_range, self.embedding_range)


    def score(self, head, relation, tail, mode):
        """
        Input head/tail has stdev 1 for each element. Scale to stdev 1/sqrt(12) * (b-a) = a/sqrt(3).
        Reference: https://github.com/snap-stanford/ogb/blob/master/examples/linkproppred/wikikg2/model.py
        """
        head = head * self.embedding_range / math.sqrt(3.0)
        tail = tail * self.embedding_range / math.sqrt(3.0)


        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        #Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation/(self.embedding_range/pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim = 0)
        score = score.norm(dim = 0)

        score = self.gamma - score.sum(dim = 2)
        return score

    def __repr__(self):
        return '{}(embedding_size={}, num_relations={}, gamma={}, dist_ord={})'.format(self.__class__.__name__,
                                                                                       self.embedding_dim,
                                                                                       self.num_relations,
                                                                                       self.gamma)


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
    print("DRAGON benchmark: KG-GNN attention and link-prediction decoders")
    print("=" * 70)

    device = torch.device("cpu")
    args = SimpleNamespace(
        fp16=False,
        upcast=False,
        link_negative_adversarial_sampling=False,
        link_negative_adversarial_sampling_temperature=1.0,
        link_regularizer_weight=0.01,
        link_gamma=12.0,
        scaled_distmult=False,
    )

    # Test group 1: one-hot utility used by typed graph edges/nodes
    try:
        labels = torch.tensor([0, 2, 1, 3], dtype=torch.long, device=device)
        one_hot = make_one_hot(labels, 4)
        check("make_one_hot output not None", one_hot is not None)
        if one_hot is not None:
            check("make_one_hot shape", tuple(one_hot.shape) == (4, 4), f"got {tuple(one_hot.shape)}")
            check("make_one_hot row sums", torch.allclose(one_hot.sum(dim=1), torch.ones(4, device=device)))
            check("make_one_hot selected entries", one_hot[1, 2].item() == 1.0 and one_hot[3, 3].item() == 1.0)
        else:
            skip_checks(3, "make_one_hot returned None")
    except Exception as exc:
        skip_checks(4, f"make_one_hot raised {type(exc).__name__}: {exc}")

    # Test group 2: GATConvE forward pass and attention semantics
    try:
        emb_dim = 8
        n_ntype = 4
        n_etype = 3
        edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_etype + 1 + n_ntype * 2, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        ).to(device)
        conv = GATConvE(args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=2).to(device).eval()
        x = torch.randn(5, emb_dim, device=device)
        edge_index = torch.tensor([[0, 0, 1, 2, 3, 4], [1, 2, 2, 3, 4, 0]], dtype=torch.long, device=device)
        edge_type = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long, device=device)
        node_type = torch.tensor([3, 0, 1, 2, 2], dtype=torch.long, device=device)
        node_extra = torch.randn(5, emb_dim, device=device)
        out, (returned_edge_index, alpha) = conv(x, edge_index, edge_type, node_type, node_extra, return_attention_weights=True)
        check("GATConvE output not None", out is not None)
        if out is not None:
            check("GATConvE output shape", tuple(out.shape) == (5, emb_dim), f"got {tuple(out.shape)}")
            check("GATConvE output finite", torch.isfinite(out).all().item())
            check("GATConvE adds self loops", returned_edge_index.shape[1] == edge_index.shape[1] + x.shape[0])
            check("GATConvE attention shape", tuple(alpha.shape) == (edge_index.shape[1] + x.shape[0], 2), f"got {tuple(alpha.shape)}")
            alpha_ok = True
            for node_id in returned_edge_index[0].unique():
                mask = returned_edge_index[0] == node_id
                alpha_ok = alpha_ok and torch.allclose(alpha[mask].sum(dim=0), torch.ones(2, device=device), atol=1e-5)
            check("GATConvE attention normalized per source", alpha_ok)
        else:
            skip_checks(5, "GATConvE returned None")
    except Exception as exc:
        skip_checks(6, f"GATConvE raised {type(exc).__name__}: {exc}")

    # Test group 3: decoder sampling modes and loss
    try:
        embs = torch.randn(7, 8, device=device, requires_grad=True)
        sample = (
            torch.tensor([0, 2, 4], dtype=torch.long, device=device),
            torch.tensor([0, 1, 2], dtype=torch.long, device=device),
            torch.tensor([1, 3, 5], dtype=torch.long, device=device),
        )
        neg_heads = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long, device=device)
        neg_tails = torch.tensor([[2, 3], [4, 5], [6, 0]], dtype=torch.long, device=device)
        decoder = DistMultDecoder(args, num_rels=3, h_dim=8).to(device)
        single_score = decoder(embs, sample, mode='single')
        head_score = decoder(embs, (sample, neg_heads), mode='head-batch')
        tail_score = decoder(embs, (sample, neg_tails), mode='tail-batch')
        check("Decoder single score not None", single_score is not None)
        if single_score is not None:
            check("Decoder single shape", tuple(single_score.shape) == (3, 1), f"got {tuple(single_score.shape)}")
            check("Decoder head-batch shape", tuple(head_score.shape) == (3, 2), f"got {tuple(head_score.shape)}")
            check("Decoder tail-batch shape", tuple(tail_score.shape) == (3, 2), f"got {tuple(tail_score.shape)}")
            check("Decoder scores finite", torch.isfinite(single_score).all().item() and torch.isfinite(head_score).all().item() and torch.isfinite(tail_score).all().item())
            loss, pos_loss, neg_loss = decoder.loss((single_score, torch.cat([head_score, tail_score], dim=1)))
            check("Decoder loss scalar", getattr(loss, "dim", lambda: -1)() == 0)
            check("Decoder loss finite", torch.isfinite(loss).item())
            loss.backward()
            check("Decoder relation gradient", decoder.w_relation.grad is not None and torch.isfinite(decoder.w_relation.grad).all().item())
        else:
            skip_checks(7, "Decoder returned None")
    except Exception as exc:
        skip_checks(8, f"Decoder raised {type(exc).__name__}: {exc}")

    # Test group 4: TransE, DistMult, and RotatE score functions
    try:
        head = torch.randn(3, 2, 8, device=device)
        tail = torch.randn(3, 2, 8, device=device)
        relation_8 = torch.randn(3, 1, 8, device=device)
        transe = TransEDecoder(args, num_rels=3, h_dim=8).to(device)
        transe_score = transe.score(head, relation_8, tail, mode='tail-batch')
        check("TransE score not None", transe_score is not None)
        if transe_score is not None:
            check("TransE score shape", tuple(transe_score.shape) == (3, 2), f"got {tuple(transe_score.shape)}")
            check("TransE score finite", torch.isfinite(transe_score).all().item())

        distmult = DistMultDecoder(args, num_rels=3, h_dim=8).to(device)
        dm_forward = distmult.score(head, relation_8, tail, mode='tail-batch')
        dm_reverse = distmult.score(tail, relation_8, head, mode='tail-batch')
        check("DistMult score shape", tuple(dm_forward.shape) == (3, 2), f"got {tuple(dm_forward.shape)}")
        check("DistMult symmetric score", torch.allclose(dm_forward, dm_reverse, atol=1e-5))
        check("DistMult score finite", torch.isfinite(dm_forward).all().item())

        rotate = RotatEDecoder(args, num_rels=3, h_dim=8).to(device)
        relation_4 = torch.randn(3, 1, 4, device=device)
        rotate_score = rotate.score(head, relation_4, tail, mode='tail-batch')
        check("RotatE score shape", tuple(rotate_score.shape) == (3, 2), f"got {tuple(rotate_score.shape)}")
        check("RotatE score finite", torch.isfinite(rotate_score).all().item())
        check("RotatE relation half dimension", rotate.w_relation.shape == (3, 4))
    except Exception as exc:
        skip_checks(10, f"score functions raised {type(exc).__name__}: {exc}")

    # Test group 5: direct GATConvE.message behavior
    try:
        emb_dim = 8
        n_edges = 7
        edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(3 + 1 + 4 * 2, emb_dim),
            torch.nn.BatchNorm1d(emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        ).to(device)
        conv = GATConvE(args, emb_dim, 4, 3, edge_encoder, head_count=2).to(device).eval()
        edge_index = torch.tensor([[0, 0, 1, 2, 2, 3, 4], [1, 2, 2, 3, 4, 4, 0]], dtype=torch.long, device=device)
        x_i = torch.randn(n_edges, 16, device=device)
        x_j = torch.randn(n_edges, 16, device=device)
        edge_attr = torch.randn(n_edges, 8, device=device)
        msg = conv.message(edge_index, x_i, x_j, edge_attr)
        check("GATConvE.message not None", msg is not None)
        if msg is not None:
            check("GATConvE.message shape", tuple(msg.shape) == (n_edges, 8), f"got {tuple(msg.shape)}")
            check("GATConvE.message finite", torch.isfinite(msg).all().item())
            check("GATConvE.message stores alpha", conv._alpha is not None and tuple(conv._alpha.shape) == (n_edges, 2))
            alpha_ok = True
            for node_id in edge_index[0].unique():
                mask = edge_index[0] == node_id
                alpha_ok = alpha_ok and torch.allclose(conv._alpha[mask].sum(dim=0), torch.ones(2, device=device), atol=1e-5)
            check("GATConvE.message alpha normalized", alpha_ok)
        else:
            skip_checks(4, "GATConvE.message returned None")
    except Exception as exc:
        skip_checks(5, f"GATConvE.message raised {type(exc).__name__}: {exc}")

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
