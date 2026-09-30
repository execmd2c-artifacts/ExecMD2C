"""
ground_truth.py for Neural Improvement Heuristics core model components.

Source-consolidated from:
- networks/gnn_encoder.py
- networks/mlp_decoder.py
- networks/model.py

Only the neural policy architecture is included. Environment simulation,
parameter-update procedures, search procedures, command-line option parsing,
saved data files, and saved-weight I/O are intentionally excluded.
"""

import torch
from torch import nn
import torch.nn.functional as F


class GNNLayer(nn.Module):
    def __init__(self, **model_params):
        super(GNNLayer, self).__init__()
        hidden_dim = model_params['embedding_dim']
        self.aggregation = model_params['aggregation']

        self.U = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.V = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

        self.norm_h = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)
        self.norm_e = nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=False)


    def forward(self, h, e):
        batch_size, num_nodes, hidden_dim = h.shape
        h_in = h
        e_in = e

        # Linear transformations for node update
        Uh = self.U(h)  # B x V x H
        Vh = self.V(h).unsqueeze(1).expand(-1, num_nodes, -1, -1)  # B x V x V x H

        # Linear transformations for edge update and gating
        Ah = self.A(h)  # B x V x H
        Bh = self.B(h)  # B x V x H
        Ce = self.C(e)  # B x V x V x H

        # Update edge features and compute edge gates
        e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
        gates = torch.sigmoid(e)  # B x V x V x H

        # Update node features
        h = Uh + self.aggregate(Vh, gates)  # B x V x H

        # Normalize node features
        h = self.norm_h(h.view(batch_size * num_nodes, hidden_dim)).view(batch_size, num_nodes, hidden_dim)

        # Normalize edge features
        e = self.norm_e(e.view(batch_size * num_nodes * num_nodes, hidden_dim)).view(batch_size, num_nodes, num_nodes, hidden_dim)

        # Apply non-linearity
        h = F.relu(h)
        e = F.relu(e)

        # Make residual connection
        h = h_in + h
        e = e_in + e

        return h, e

    def aggregate(self, Vh, gates):
        # Perform feature-wise gating mechanism
        Vh = gates * Vh  # B x V x V x H

        if self.aggregation == "mean":
            return torch.sum(Vh, dim=2) / Vh.shape[1]
        elif self.aggregation == "max":
            return torch.max(Vh, dim=2)[0]
        else:
            return torch.sum(Vh, dim=2)


class GNN_Encoder(nn.Module):
    def __init__(self, **model_params):
        super(GNN_Encoder, self).__init__()
        self.init_node_embed = nn.Linear(model_params['node_dim'], model_params['embedding_dim'], bias=True)
        self.init_edge_embed = nn.Linear(model_params['edge_dim'], model_params['embedding_dim'], bias=True)
        self.layers = nn.ModuleList([
            GNNLayer(**model_params)
            for _ in range(model_params['n_encode_layers'])
        ])

    def forward(self, nodes, edges):
        # Embed node and edge features
        x = self.init_node_embed(nodes)
        e = self.init_edge_embed(edges)
        for layer in self.layers:
            x, e = layer(x, e)
        return x, e


class MLP_Decoder(nn.Module):
    def __init__(self):
        super(MLP_Decoder, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, e, mask):
        e = F.relu(self.fc1(e))
        e = F.relu(self.fc2(e))
        e = self.fc3(e)

        e = torch.reshape(e.clone(), (e.shape[0], e.shape[1] * e.shape[2]))
        e[:, mask[0, :]] = -1e10

        log_p = F.log_softmax(e, dim=-1)

        return log_p


class Model(nn.Module):
    def __init__(self, **model_params):
        super(Model, self).__init__()
        self.encoder = GNN_Encoder(**model_params)
        self.decoder = MLP_Decoder()

    def forward(self, batch, mask, problem, solutions=None):
        nodes, edges = batch
        batch_size, n, _ = edges.shape

        if problem == 'prp':
            edges = torch.stack([torch.triu(edges, 1), torch.triu(edges.permute(0, 2, 1), 1)])
            edges = edges.permute(1, 2, 3, 0)
        elif problem == 'tsp':
            if not hasattr(self, 'distance_indices'):
                distance_indices = torch.zeros((n, n), dtype=torch.bool)
                for i in range(n - 1):
                    distance_indices[i, i + 1] = True
                    distance_indices[i + 1, i] = True
                distance_indices[-1, 0] = True
                distance_indices[0, -1] = True
                self.distance_indices = distance_indices
            edges = torch.stack((self.distance_indices.repeat(edges.shape[0], 1, 1),
                                ~self.distance_indices.repeat(edges.shape[0], 1, 1), edges), -1)
        elif problem == 'gpp':
            assert solutions is not None
            edge_features = []
            for b in range(batch_size):
                e = torch.zeros(n, n, 2)
                sol = solutions[b, :, :]
                idx_mat = torch.zeros(n, n, dtype=torch.bool)
                for i in range(n // 2):
                    for j in range(n // 2):
                        idx_mat[sol[0, i], sol[1, j]] = True
                        idx_mat[sol[1, j], sol[0, i]] = True

                e[:, :, 0] = edges[b, :, :] * idx_mat
                e[:, :, 1] = edges[b, :, :] * ~idx_mat
                edge_features.append(e)

            edges = torch.stack(edge_features)

        # ENCODER
        node_embedding, edge_embedding = self.encoder(nodes, edges)
        edge_embedding = edge_embedding.view(edge_embedding.shape[0], -1, edge_embedding.shape[-1])

        # DECODER
        log_p = self.decoder(edge_embedding, mask)

        return log_p


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

    def params(node_dim=1, edge_dim=2, aggregation="sum", layers=2):
        return {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "embedding_dim": 128,
            "n_encode_layers": layers,
            "aggregation": aggregation,
        }

    print("Running Neural Improvement Heuristics core model benchmark checks...")

    try:
        layer = GNNLayer(**params(node_dim=1, edge_dim=2, aggregation="sum", layers=1))
        h = torch.randn(2, 4, 128, requires_grad=True)
        e = torch.randn(2, 4, 4, 128, requires_grad=True)
        out_h, out_e = layer(h, e)
        check("gnn layer node not None", out_h is not None)
        check("gnn layer edge not None", out_e is not None)
        if out_h is not None and out_e is not None:
            check("gnn layer node shape", tuple(out_h.shape) == (2, 4, 128), f"got {tuple(out_h.shape)}")
            check("gnn layer edge shape", tuple(out_e.shape) == (2, 4, 4, 128), f"got {tuple(out_e.shape)}")
            check("gnn layer finite", torch.isfinite(out_h).all().item() and torch.isfinite(out_e).all().item())
            (out_h.sum() + out_e.sum()).backward()
            check("gnn layer gradient path", h.grad is not None and e.grad is not None)
        else:
            skip_checks(4, "GNNLayer.forward returned None")
    except Exception as exc:
        skip_checks(6, f"GNNLayer.forward checks raised {type(exc).__name__}: {exc}")

    try:
        Vh = torch.arange(1 * 3 * 3 * 2, dtype=torch.float32).view(1, 3, 3, 2)
        gates = torch.ones_like(Vh)
        layer_sum = GNNLayer(**params(aggregation="sum", layers=1))
        layer_mean = GNNLayer(**params(aggregation="mean", layers=1))
        layer_max = GNNLayer(**params(aggregation="max", layers=1))
        sum_out = layer_sum.aggregate(Vh, gates)
        mean_out = layer_mean.aggregate(Vh, gates)
        max_out = layer_max.aggregate(Vh, gates)
        check("aggregate sum shape", tuple(sum_out.shape) == (1, 3, 2), f"got {tuple(sum_out.shape)}")
        check("aggregate sum values", torch.equal(sum_out, Vh.sum(dim=2)))
        check("aggregate mean values", torch.equal(mean_out, Vh.sum(dim=2) / Vh.shape[1]))
        check("aggregate max values", torch.equal(max_out, Vh.max(dim=2)[0]))
    except Exception as exc:
        skip_checks(4, f"GNNLayer.aggregate checks raised {type(exc).__name__}: {exc}")

    try:
        encoder = GNN_Encoder(**params(node_dim=1, edge_dim=2, aggregation="sum", layers=2))
        nodes = torch.randn(2, 4, 1)
        edges = torch.randn(2, 4, 4, 2)
        node_emb, edge_emb = encoder(nodes, edges)
        check("encoder node shape", tuple(node_emb.shape) == (2, 4, 128), f"got {tuple(node_emb.shape)}")
        check("encoder edge shape", tuple(edge_emb.shape) == (2, 4, 4, 128), f"got {tuple(edge_emb.shape)}")
        check("encoder finite", torch.isfinite(node_emb).all().item() and torch.isfinite(edge_emb).all().item())
        check("encoder layer count", len(encoder.layers) == 2)
    except Exception as exc:
        skip_checks(4, f"GNN_Encoder checks raised {type(exc).__name__}: {exc}")

    try:
        decoder = MLP_Decoder()
        edge_emb = torch.randn(2, 16, 128)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, 0] = True
        mask[0, 5] = True
        log_p = decoder(edge_emb, mask)
        check("decoder output not None", log_p is not None)
        if log_p is not None:
            check("decoder output shape", tuple(log_p.shape) == (2, 16), f"got {tuple(log_p.shape)}")
            check("decoder output finite", torch.isfinite(log_p).all().item())
            check("decoder prob normalization", torch.allclose(log_p.exp().sum(dim=1), torch.ones(2), atol=1e-5))
            check("decoder masked probability", torch.all(log_p.exp()[:, mask[0]] < 1e-6).item())
        else:
            skip_checks(4, "MLP_Decoder.forward returned None")
    except Exception as exc:
        skip_checks(5, f"MLP_Decoder checks raised {type(exc).__name__}: {exc}")

    try:
        model = Model(**params(node_dim=1, edge_dim=2, aggregation="sum", layers=1))
        nodes = torch.randn(2, 4, 1)
        edges = torch.randn(2, 4, 4)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        mask[0, 0] = True
        log_p = model((nodes, edges), mask, "prp")
        check("model prp output shape", tuple(log_p.shape) == (2, 16), f"got {tuple(log_p.shape)}")
        check("model prp finite", torch.isfinite(log_p).all().item())
        check("model prp probabilities", torch.allclose(log_p.exp().sum(dim=1), torch.ones(2), atol=1e-5))
    except Exception as exc:
        skip_checks(3, f"Model PRP checks raised {type(exc).__name__}: {exc}")

    try:
        tsp_model = Model(**params(node_dim=2, edge_dim=3, aggregation="mean", layers=1))
        nodes = torch.rand(2, 4, 2)
        edges = torch.rand(2, 4, 4)
        mask = torch.zeros(1, 16, dtype=torch.bool)
        log_p = tsp_model((nodes, edges), mask, "tsp")
        check("model tsp output shape", tuple(log_p.shape) == (2, 16), f"got {tuple(log_p.shape)}")
        check("model tsp distance cache", hasattr(tsp_model, "distance_indices") and tuple(tsp_model.distance_indices.shape) == (4, 4))
        check("model tsp finite", torch.isfinite(log_p).all().item())

        gpp_model = Model(**params(node_dim=1, edge_dim=2, aggregation="max", layers=1))
        nodes = torch.rand(2, 4, 1)
        edges = torch.rand(2, 4, 4)
        solutions = torch.tensor([[[0, 1], [2, 3]], [[0, 2], [1, 3]]], dtype=torch.long)
        log_p = gpp_model((nodes, edges), mask, "gpp", solutions=solutions)
        check("model gpp output shape", tuple(log_p.shape) == (2, 16), f"got {tuple(log_p.shape)}")
        check("model gpp finite", torch.isfinite(log_p).all().item())
    except Exception as exc:
        skip_checks(5, f"Model TSP/GPP checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
