# ============================================================
# ground_truth.py - deepGCFX Graph-Level Core Model Components
# Source: Graphs/deepGCFX-main
#
# Contains only graph-level model architecture definitions and direct helpers.
# No dataset loading, training loop, evaluation, checkpoint, or CLI code.
# ============================================================

# --- Third-party imports ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear, Module, PReLU, ReLU, Sequential, Sigmoid, Tanh
from torch.nn import init
from torch.nn.utils.weight_norm import weight_norm
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.utils import add_self_loops, negative_sampling, remove_self_loops, to_dense_adj, to_dense_batch


dataset_num_features = 1


# --- [Original file: deepgcfx_graph/utils.py] ---
def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


# --- [Original file: deepgcfx_graph/summary_maker_1.py] ---
class Summary_Maker(nn.Module):
    def __init__(self, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super(Summary_Maker, self).__init__()
        self.num_slots = 1
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn( 1, dim))

        self.slots_logsigma = nn.Parameter(torch.zeros( 1, dim))
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)
        self.g_weight_calc = nn.Sequential(nn.Linear(dim*2, dim), nn.ReLU(), nn.Linear(dim, 1))

        #self.init_weight_calc = nn.Linear(in_features=dim, out_features=1, bias=True)

        hidden_dim = max(dim, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.mlp_nodes = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.norm_nodes  = nn.LayerNorm(dim)

    def forward(self, inputs, batch):

        uniq_ele, count = torch.unique(batch,  return_counts=True)
        b = uniq_ele.size(0)
        n, d = inputs.shape
        n_s = 1

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_logsigma.exp().expand(b, n_s, -1)

        slots = mu + sigma * torch.randn(mu.shape, device = device)

        '''node_wise_common_weight = torch.sigmoid(self.init_weight_calc(inputs))
        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x
        summary_related_node_info = node_wise_common_weight*inputs
        noisy_node_info = (1-node_wise_common_weight)*inputs

        slots = global_add_pool(summary_related_node_info,batch)'''

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        summary_related_node_info = None
        noisy_node_info = None

        for iter in range(self.iters):

            slots = slots.view(-1, slots.size(-1))
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

            node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

            compact_node = torch.sigmoid(k)
            mask = (compact_node >= node_wise_summary_weight).float()
            antimask = (compact_node < node_wise_summary_weight).float()

            summary_related_node_info = mask * v
            noisy_node_info = antimask * v

            updates = global_add_pool(summary_related_node_info, batch)

            slots = self.gru(
                updates.reshape(-1, d),
                slots_prev.reshape(-1, d)
            )

            slots = slots.reshape(b, -1, d)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

            #slots = (slots - slots.mean(1).unsqueeze(1)) / slots.std(1).unsqueeze(1)
            #slots = self.bn(slots)

        slots_a = slots.view(-1, slots.size(-1))
        slots = self.norm_slots(slots_a)
        q = self.to_q(slots)

        expanded_graph_latent_embeddings = torch.repeat_interleave(q, count, dim=0)

        node_wise_summary_weight = torch.sigmoid(self.g_weight_calc(torch.cat([k,expanded_graph_latent_embeddings],-1)))

        compact_node = torch.sigmoid(k)
        mask = (compact_node >= node_wise_summary_weight).float()

        antimask = (compact_node < node_wise_summary_weight).float()

        summary_related_node_info = mask * v
        noisy_node_info = antimask * v

        return slots, summary_related_node_info, noisy_node_info


# --- [Original file: deepgcfx_graph/gin_1.py] ---
class FCNet(Module):
    """Simple class for non-linear fully connect network with gated tangent as in paper
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        in_dim = dims[0]
        out_dim = dims[1]
        self.first_lin = weight_norm(Linear(in_dim, out_dim), dim=None)
        self.tanh = Tanh()
        self.second_lin = weight_norm(Linear(in_dim, out_dim), dim=None)
        self.sigmoid = Sigmoid()


    def forward(self, x):

        y_hat = self.tanh(self.first_lin(x))
        g = self.sigmoid(self.second_lin(x))
        y = y_hat * g

        return y


class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, summary_iter=2):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers
        self.eps = 1e-8
        self.dim = dim

        # self.nns = []
        #self.sum_q = Embedding(1, 128)
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.act = PReLU()

        for i in range(num_gc_layers):

            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)


            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)
        '''for i in range(num_gc_layers):

            if i == 0:
                conv = GCNConv(num_features, dim)
                bn = torch.nn.BatchNorm1d(dim)
            else:
                conv = GCNConv(dim, dim)
                bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)'''


        self.flatten = Linear(dim*num_gc_layers, dim)

        self.summary_maker = Summary_Maker(dim, iters=summary_iter)

        # node
        self.node_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_mu_bn = torch.nn.BatchNorm1d(dim)
        self.node_logvar_bn = torch.nn.BatchNorm1d(dim)

        self.graph_mu_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.graph_logvar_proj = Linear(in_features=dim, out_features=dim, bias=True)
        self.graph_mu_bn = torch.nn.BatchNorm1d(dim)
        self.graph_logvar_bn = torch.nn.BatchNorm1d(dim)

    def forward_diff(self, x, edge_index, batch):


        xs = []
        for i in range(self.num_gc_layers):

            a = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](a)
            xs.append(x)


        #ful_rep = self.flatten(torch.cat(xs, 1)) + x
        slots, summary_related_node_info, noisy_node_info = self.summary_maker(x, batch)

        #ful_rep = self.flatten(torch.cat(xs, 1))
        #ful_rep = x

        node_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(noisy_node_info))+self.eps)
        node_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(noisy_node_info))+self.eps)

        gnode_latent_space_mu = self.node_mu_bn(F.relu(self.node_mu_proj(summary_related_node_info))+self.eps)
        gnode_latent_space_logvar = self.node_logvar_bn(F.relu(self.node_logvar_proj(summary_related_node_info))+self.eps)

        graph_latent_space_mu = self.graph_mu_bn(F.relu(self.graph_mu_proj(slots))+self.eps)
        graph_latent_space_logvar = self.graph_logvar_bn(F.relu(self.graph_logvar_proj(slots))+self.eps)

        return (x, noisy_node_info, slots),node_latent_space_mu,  node_latent_space_logvar, gnode_latent_space_mu,  gnode_latent_space_logvar, graph_latent_space_mu, graph_latent_space_logvar


# --- [Original file: deepgcfx_graph/main.py] ---
class FF(nn.Module):
    def __init__(self, input_dim):
        super(FF, self).__init__()
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1)
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)
        # self.c0 = nn.Conv1d(input_dim, 512, kernel_size=1, stride=1, padding=0)
        # self.c1 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding=0)
        # self.c2 = nn.Conv1d(512, 1, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)



class deepGCFX(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers):
        super(deepGCFX, self).__init__()

        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers,summary_iter=2)
        self.fusioner = FCNet([hidden_dim*2, hidden_dim])
        self.eps = 1e-8

        '''self.fusioner = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
        )'''

        self.g_accum = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        '''self.g_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.n_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )'''

        self.g_proj = FF(hidden_dim)
        self.n_proj = FF(hidden_dim)

        self.bn_class = torch.nn.BatchNorm1d(hidden_dim)

        self.mse_loss = nn.MSELoss()

        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):


        n_nodes = x.size(0)

        # batch_size = data.num_graphs
        #our model
        internal_loss_items, node_mu, node_logvar,gnode_mu, gnode_logvar, graph_mu, graph_logvar = \
            self.encoder.forward_diff(x, edge_index, batch)
        #sum all layers of gnn for summary makes
        '''internal_loss_items, node_mu, node_logvar,gnode_mu, gnode_logvar, graph_mu, graph_logvar = \
            self.encoder.forward_tot4summary_diff(x, edge_index, batch)'''

        '''grouped_mu, grouped_logvar = accumulate_group_evidence(
            graph_mu.data, graph_mu.data, batch, True
        )'''

        node_kl_divergence_loss = (-0.5 / n_nodes) * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))

        graph_kl_divergence_loss = (-0.5 / num_graphs) * torch.mean(torch.sum(
            1 + 2 * graph_logvar - graph_mu.pow(2) - graph_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = node_kl_divergence_loss

        graph_kl_divergence_loss = 0.1*graph_kl_divergence_loss

        node_latent_embeddings_unnorm = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        gnode_latent_embeddings_unnorm = reparameterize(training=True, mu=gnode_mu, logvar=gnode_logvar)
        graph_latent_embeddings_unnorm = reparameterize(training=True, mu=graph_mu, logvar=graph_logvar)
        '''graph_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
        )'''
        node_latent_embeddings = node_latent_embeddings_unnorm
        graph_latent_embeddings = graph_latent_embeddings_unnorm

        _, count = torch.unique(batch,  return_counts=True)

        graph_embedding_expanded = torch.repeat_interleave(graph_latent_embeddings, count, dim=0)

        gl_nodes = self.fusioner(torch.cat([graph_embedding_expanded,node_latent_embeddings], -1))
        #g_nodes = self.fusioner(torch.cat([torch.zeros_like(node_latent_embeddings), graph_latent_embeddings], -1))

        full_node_feat_gl = gl_nodes


        edge_recon_loss_gl = self.recon_loss1(full_node_feat_gl, edge_index, batch)
        edge_recon_loss_g = self.recon_loss1(gnode_latent_embeddings_unnorm, edge_index, batch)

        '''contrastive_loss = self.node_graph_contrastive_loss(self.g_proj(graph_latent_embeddings), self.n_proj(gnode_latent_embeddings_unnorm),
                                                            self.n_proj(node_latent_embeddings_unnorm), batch)'''



        #loss = node_kl_divergence_loss + graph_kl_divergence_loss + edge_recon_loss_gl + contrastive_loss + edge_recon_loss_g
        loss = node_kl_divergence_loss + graph_kl_divergence_loss + edge_recon_loss_gl  + edge_recon_loss_g

        loss.backward()

        return  node_kl_divergence_loss.item(), graph_kl_divergence_loss.item(), 0, \
                edge_recon_loss_gl.item(),edge_recon_loss_g.item()

    def edge_recon(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def recon_loss1(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #org_adj = to_dense_adj(edge_index, batch)
        pos_weight = float(z.size(0) * z.size(0) - edge_index.size(0)) / edge_index.size(0)
        norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - edge_index.size(0)) * 2)



        recon_adj = self.edge_recon(z, edge_index)


        pos_loss = -torch.log(
            recon_adj + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) #random thingggg
        neg_loss = -torch.log(1 -
                              self.edge_recon(z, neg_edge_index) +
                              EPS).mean()

        return (pos_loss + neg_loss)


# ============================================================
# __main__: Automated test suite for 3 ablated functions
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

    print("=" * 70)
    print("deepGCFX Graph-Level Core Model Components")
    print("Automated Test Suite - 3 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long, device=device)
    x = torch.randn(6, 4, device=device)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 3, 4, 4, 5],
         [1, 0, 2, 1, 4, 3, 5, 4]],
        dtype=torch.long,
        device=device,
    )

    # ==============================================================
    # Test 1/3: Summary_Maker.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/3] Summary_Maker.forward")
    try:
        summary = Summary_Maker(dim=4, iters=2, hidden_dim=8).to(device)
        slots, summary_nodes, noisy_nodes = summary(x, batch)
        check("Summary_Maker output not None", slots is not None and summary_nodes is not None and noisy_nodes is not None)
        if slots is not None and summary_nodes is not None and noisy_nodes is not None:
            check("Summary_Maker slots shape", tuple(slots.shape) == (2, 4),
                  f"expected (2, 4), got {tuple(slots.shape)}")
            check("Summary_Maker node split shape", tuple(summary_nodes.shape) == (6, 4) and tuple(noisy_nodes.shape) == (6, 4),
                  f"expected both (6, 4), got {tuple(summary_nodes.shape)} and {tuple(noisy_nodes.shape)}")
            check("Summary_Maker outputs finite",
                  torch.isfinite(slots).all().item() and torch.isfinite(summary_nodes).all().item() and torch.isfinite(noisy_nodes).all().item())
            check("Summary_Maker disjoint node split",
                  torch.allclose(summary_nodes * noisy_nodes, torch.zeros_like(summary_nodes), atol=1e-6))
            check("Summary_Maker keeps some common signal", summary_nodes.abs().sum().item() > 0)
        else:
            skip_checks(5, "Summary_Maker.forward returned None")
    except Exception as exc:
        print(f"  [Summary_Maker.forward] ERROR: {exc}")
        skip_checks(6, "Summary_Maker.forward raised an exception")
    print()

    # ==============================================================
    # Test 2/3: Encoder.forward_diff
    # ==============================================================
    print("-" * 60)
    print("[Test 2/3] Encoder.forward_diff")
    try:
        encoder = Encoder(num_features=4, dim=8, num_gc_layers=2, summary_iter=2).to(device)
        encoder.train()
        result = encoder.forward_diff(x, edge_index, batch)
        check("Encoder.forward_diff output not None", result is not None)
        if result is not None:
            internal, node_mu, node_logvar, gnode_mu, gnode_logvar, graph_mu, graph_logvar = result
            check("Encoder.forward_diff tuple length", len(result) == 7)
            check("Encoder.forward_diff internal shapes",
                  tuple(internal[0].shape) == (6, 8) and tuple(internal[1].shape) == (6, 8) and tuple(internal[2].shape) == (2, 8))
            check("Encoder.forward_diff node latent shapes",
                  tuple(node_mu.shape) == (6, 8) and tuple(node_logvar.shape) == (6, 8)
                  and tuple(gnode_mu.shape) == (6, 8) and tuple(gnode_logvar.shape) == (6, 8))
            check("Encoder.forward_diff graph latent shapes",
                  tuple(graph_mu.shape) == (2, 8) and tuple(graph_logvar.shape) == (2, 8))
            check("Encoder.forward_diff outputs finite",
                  all(torch.isfinite(t).all().item() for t in [internal[0], internal[1], internal[2], node_mu, node_logvar, gnode_mu, gnode_logvar, graph_mu, graph_logvar]))
            check("Encoder.forward_diff separates latent branches",
                  not torch.allclose(node_mu, gnode_mu))
        else:
            skip_checks(6, "Encoder.forward_diff returned None")
    except Exception as exc:
        print(f"  [Encoder.forward_diff] ERROR: {exc}")
        skip_checks(7, "Encoder.forward_diff raised an exception")
    print()

    # ==============================================================
    # Test 3/3: deepGCFX.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/3] deepGCFX.forward")
    try:
        dataset_num_features = 4
        model = deepGCFX(hidden_dim=8, num_gc_layers=2).to(device)
        model.train()
        output = model(x.clone(), edge_index, batch, num_graphs=2)
        check("deepGCFX.forward output not None", output is not None)
        if output is not None:
            check("deepGCFX.forward tuple length", len(output) == 5)
            check("deepGCFX.forward scalar losses", all(isinstance(v, (int, float)) for v in output))
            check("deepGCFX.forward finite losses", all(np.isfinite(v) for v in output))
            check("deepGCFX.forward edge losses positive", output[3] > 0 and output[4] > 0)
        else:
            skip_checks(4, "deepGCFX.forward returned None")
    except Exception as exc:
        print(f"  [deepGCFX.forward] ERROR: {exc}")
        skip_checks(5, "deepGCFX.forward raised an exception")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
