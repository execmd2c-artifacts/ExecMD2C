# ============================================================
# ground_truth.py - ImGAGN Core Model Components
# Source:
#   Miscellaneous/ImGAGN-main/ImGAGN/layers.py
#   Miscellaneous/ImGAGN-main/ImGAGN/models.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter


# --- [Original file: ImGAGN/layers.py] ---
class GraphConvolution(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


# --- [Original file: ImGAGN/models.py] ---
class Attention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Attention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.ReLU(),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, generate_node, min_node):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.gc3 = GraphConvolution(nhid, 2)
        self.attention = Attention(nfeat*2, 1)
        self.generate_node = generate_node
        self.min_node = min_node
        self.dropout = dropout
        self.eps = 1e-10

    def forward(self, x, adj):

        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x1 = self.gc2(x, adj)
        x2 = self.gc3(x, adj)
        return F.log_softmax(x1, dim=1), F.log_softmax(x2, dim=1), F.softmax(x1, dim=1)[:,-1]

    def get_embedding(self,x , adj):
        x = F.relu(self.gc1(x, adj))
        x = torch.spmm(adj, x)
        return x


class Generator(nn.Module):
    def __init__(self,  dim):
        super(Generator, self).__init__( )

        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, dim)
        self.fc4 = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.fc4(x)
        x = (x+1)/2
        return x


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
    print("ImGAGN: Imbalanced Networks Embedding via Generative Adversarial Graph Networks")
    print("Automated reproduction benchmark - 3 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    def sparse_identity_with_edges():
        indices = torch.tensor([
            [0, 1, 2, 3, 0, 1, 2],
            [0, 1, 2, 3, 1, 2, 3],
        ], dtype=torch.long, device=device)
        values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)
        return torch.sparse_coo_tensor(indices, values, (4, 4)).coalesce()

    # ==============================================================
    # Test 1/3: GraphConvolution.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/3] GraphConvolution.forward - sparse graph feature propagation")
    try:
        layer = GraphConvolution(3, 2, bias=True).to(device)
        features = torch.randn(4, 3, device=device, requires_grad=True)
        adj = sparse_identity_with_edges()
        output = layer(features, adj)
        check("GraphConvolution output not None", output is not None)
        if output is not None:
            with torch.no_grad():
                expected = torch.spmm(adj, torch.mm(features, layer.weight)) + layer.bias
            check("GraphConvolution output shape", tuple(output.shape) == (4, 2), f"expected (4, 2), got {tuple(output.shape)}")
            check("GraphConvolution output finite", torch.isfinite(output).all().item())
            check("GraphConvolution performs weight projection then adjacency propagation", torch.allclose(output, expected, atol=1e-6))
            output.sum().backward()
            grad_ok = layer.weight.grad is not None and layer.weight.grad.abs().sum().item() > 0 and features.grad is not None and features.grad.abs().sum().item() > 0
            check("GraphConvolution keeps gradients to weights and input features", grad_ok)
        else:
            skip_checks(4, "GraphConvolution.forward returned None")
    except Exception as e:
        skip_checks(5, f"GraphConvolution.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/3: GCN.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/3] GCN.forward - class, discriminator, and minority-score heads")
    try:
        model = GCN(nfeat=5, nhid=4, nclass=3, dropout=0.0, generate_node=torch.tensor([2, 3]), min_node=torch.tensor([1])).to(device)
        model.eval()
        features = torch.randn(4, 5, device=device, requires_grad=True)
        adj = sparse_identity_with_edges()
        out_cls, out_disc, minority_score = model(features, adj)
        check("GCN forward outputs not None", out_cls is not None and out_disc is not None and minority_score is not None)
        if out_cls is not None and out_disc is not None and minority_score is not None:
            cls_probs = out_cls.exp()
            disc_probs = out_disc.exp()
            minority_from_cls = F.softmax(model.gc2(F.relu(model.gc1(features, adj)), adj), dim=1)[:, -1]
            check("GCN class head shape", tuple(out_cls.shape) == (4, 3), f"expected (4, 3), got {tuple(out_cls.shape)}")
            check("GCN discriminator head shape", tuple(out_disc.shape) == (4, 2), f"expected (4, 2), got {tuple(out_disc.shape)}")
            check("GCN minority score shape", tuple(minority_score.shape) == (4,), f"expected (4,), got {tuple(minority_score.shape)}")
            check("GCN log-softmax heads normalize", torch.allclose(cls_probs.sum(dim=1), torch.ones(4), atol=1e-5) and torch.allclose(disc_probs.sum(dim=1), torch.ones(4), atol=1e-5))
            check("GCN minority score is final class probability", torch.allclose(minority_score, minority_from_cls, atol=1e-6))
        else:
            skip_checks(5, "GCN.forward returned None output(s)")
    except Exception as e:
        skip_checks(6, f"GCN.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/3: Generator.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/3] Generator.forward - synthetic minority relation weights")
    try:
        generator = Generator(dim=6).to(device)
        z = torch.randn(3, 100, device=device, requires_grad=True)
        gen_weights = generator(z)
        check("Generator output not None", gen_weights is not None)
        if gen_weights is not None:
            range_ok = gen_weights.min().item() >= 0.0 and gen_weights.max().item() <= 1.0
            normalized_mix = torch.mm(F.softmax(gen_weights, dim=1), torch.randn(6, 5, device=device))
            check("Generator output shape", tuple(gen_weights.shape) == (3, 6), f"expected (3, 6), got {tuple(gen_weights.shape)}")
            check("Generator output finite", torch.isfinite(gen_weights).all().item())
            check("Generator maps noise to bounded relation weights", range_ok)
            check("Generator output can parameterize feature mixing", tuple(normalized_mix.shape) == (3, 5) and torch.isfinite(normalized_mix).all().item())
            gen_weights.sum().backward()
            grad_ok = generator.fc1.weight.grad is not None and generator.fc1.weight.grad.abs().sum().item() > 0 and z.grad is not None and z.grad.abs().sum().item() > 0
            check("Generator keeps gradients from relation weights to noise and parameters", grad_ok)
        else:
            skip_checks(5, "Generator.forward returned None")
    except Exception as e:
        skip_checks(6, f"Generator.forward raised {type(e).__name__}: {e}")
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
