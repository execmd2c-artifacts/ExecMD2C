# ============================================================
# ground_truth.py - DualMatch Core Model Components
# Source: Knowledge_Base/DualMatch-main/models.py
#
# Contains ONLY the model architecture definitions.
# No training, inference, dataset, or pipeline code.
# ============================================================

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter


# Minimal model-relevant config folded in from config.global_args for one-file use.
global_args = SimpleNamespace(dual_no_time=False)


# --- [Original file: models.py] ---
class Embedding_init(nn.Module):
    @staticmethod
    def init_emb(row, col):
        w = torch.empty(row, col)
        torch.nn.init.normal_(w)
        w = torch.nn.functional.normalize(w)
        entities_emb = nn.Parameter(w)
        return entities_emb


class OverAll(nn.Module):
    def __init__(self, node_size, node_hidden,
                 rel_size, rel_hidden,
                 time_size,
                 triple_size,
                 rel_matrix,
                 ent_matrix,
                 time_matrix,
                 dropout_rate=0, depth=2, dropout_time=0.5,
                 device='cpu'
                 ):
        super(OverAll, self).__init__()
        self.dropout_rate = dropout_rate
        self.dropout_time = dropout_time

        # new adding
        # rel_or_time in GraphAttention.forward

        self.e_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)
        self.r_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
                                        dim=node_hidden)
        # self.t_encoder = GraphAttention(node_size, rel_size, triple_size, time_size, depth=depth, device=device,
        #                                 dim=node_hidden)

        self.ent_adj = self.get_spares_matrix_by_index(ent_matrix, (node_size, node_size))
        self.rel_adj = self.get_spares_matrix_by_index(rel_matrix, (node_size, rel_size))
        self.time_adj = self.get_spares_matrix_by_index(time_matrix, (node_size, time_size))

        self.ent_emb = self.init_emb(node_size, node_hidden)
        self.rel_emb = self.init_emb(rel_size, node_hidden)
        self.time_emb = self.init_emb(time_size, node_hidden)
        self.try_emb = self.init_emb(1, node_hidden)
        self.device = device
        self.ent_adj, self.rel_adj, self.time_adj = \
            map(lambda x: x.to(device), [self.ent_adj, self.rel_adj, self.time_adj])

    # get prepared
    @staticmethod
    def get_spares_matrix_by_index(index, size):
        index = torch.LongTensor(index)
        adj = torch.sparse.FloatTensor(torch.transpose(index, 0, 1),
                                       torch.ones_like(index[:, 0], dtype=torch.float), size)
        # dim ??
        return torch.sparse.softmax(adj, dim=1)

    @staticmethod
    def init_emb(*size):
        entities_emb = nn.Parameter(torch.randn(size))
        torch.nn.init.xavier_normal_(entities_emb)
        return entities_emb

    def forward(self, inputs):
        # inputs = [adj_matrix, r_index, r_val, rel_matrix, ent_matrix, train_pairs]
        ent_feature = torch.matmul(self.ent_adj, self.ent_emb)
        rel_feature = torch.matmul(self.rel_adj, self.rel_emb)
        time_feature = torch.matmul(self.time_adj, self.time_emb)
        # ent_feature = torch.cat([ent_feature, rel_feature, time_feature], dim=1)

        # note that time_feature and rel_feature is has the same shape of ent_feature
        # the dim = node_hidden, the shape[0] = # of entities
        # They are obtained by gather the linked rel/time of an entity

        adj_input = inputs[0]
        r_index = inputs[1]
        r_val = inputs[2]
        t_index = inputs[3]

        opt = [self.rel_emb, adj_input, r_index, r_val]
        opt2 = [self.time_emb, adj_input, t_index, r_val]
        # attention opt_1 or 2
        out_feature_ent = self.e_encoder([ent_feature] + opt)
        out_feature_rel = self.r_encoder([rel_feature] + opt)
        out_feature_time = self.e_encoder([time_feature] + opt2, 1)
        # out_feature_ent2 = self.e_encoder([ent_feature] + opt)
        # out_feature_rel2 = self.r_encoder([rel_feature] + opt)
        # out_feature_time2 = self.t_encoder([time_feature] + opt2, 1)
        # out_feature_time = self.e_encoder([time_feature] + opt)
        # out_feature_time = F.dropout(out_feature_time, p=self.dropout_time, training=self.training)

        if global_args.dual_no_time:
            out_feature_overall = out_feature_rel
        else:
            out_feature_overall = (out_feature_rel + out_feature_time) / 2
        out_feature = torch.cat((out_feature_ent, out_feature_overall), dim=-1)
        out_feature = F.dropout(out_feature, p=self.dropout_rate, training=self.training)
        return out_feature


class GraphAttention(nn.Module):
    def __init__(self, node_size, rel_size, triple_size, time_size,
                 activation=torch.tanh, use_bias=True,
                 attn_heads=1, dim=100,
                 depth=1, device='cpu'):
        super(GraphAttention, self).__init__()
        self.node_size = node_size
        self.activation = activation
        self.rel_size = rel_size

        self.time_size = time_size

        self.triple_size = triple_size
        self.use_bias = use_bias
        self.attn_heads = attn_heads
        self.attn_heads_reduction = 'concat'
        self.depth = depth
        self.device = device
        self.attn_kernels = []

        node_F = dim
        rel_F = dim
        self.ent_F = node_F
        ent_F = self.ent_F

        # gate kernel Eq 9 M
        self.gate_kernel = OverAll.init_emb(ent_F * (self.depth + 1), ent_F * (self.depth + 1))
        self.proxy = OverAll.init_emb(64, node_F * (self.depth + 1))
        if self.use_bias:
            self.bias = OverAll.init_emb(1, ent_F * (self.depth + 1))
        for d in range(self.depth):
            self.attn_kernels.append([])
            for h in range(self.attn_heads):
                attn_kernel = OverAll.init_emb(node_F, 1)
                self.attn_kernels[d].append(attn_kernel.to(device))

    def forward(self, inputs, rel_or_time=0):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj_index = inputs[2]  # adj
        index = torch.tensor(adj_index, dtype=torch.int64)
        index = index.to(self.device)
        # adj = torch.sparse.FloatTensor(torch.LongTensor(index),
        #                                torch.FloatTensor(torch.ones_like(index[:,0])),
        #                                (self.node_size, self.node_size))
        sparse_indices = inputs[3]  # relation index  i.e. r_index
        sparse_val = inputs[4]  # relation value  i.e. r_val

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            features_list = []
            for head in range(self.attn_heads):
                attention_kernel = self.attn_kernels[l][head]
                ####
                col = self.rel_size if rel_or_time == 0 else self.time_size
                rels_sum = torch.sparse.FloatTensor(
                    torch.transpose(torch.LongTensor(sparse_indices), 0, 1),
                    torch.FloatTensor(sparse_val),
                    (self.triple_size, col)
                )  # relation matrix
                rels_sum = rels_sum.to(self.device)
                rels_sum = torch.matmul(rels_sum, rel_emb)
                neighs = features[index[:, 1]]
                # selfs = features[index[:, 0]]
                rels_sum = F.normalize(rels_sum, p=2, dim=1)
                neighs = neighs - 2 * torch.sum(neighs * rels_sum, 1, keepdim=True) * rels_sum

                # Eq.3
                att1 = torch.squeeze(torch.matmul(rels_sum, attention_kernel), dim=-1)
                att = torch.sparse.FloatTensor(torch.transpose(index, 0, 1), att1, (self.node_size, self.node_size))
                # ??? dim ??
                att = torch.sparse.softmax(att, dim=1)
                # ?
                # print(att1)
                # print(att.data)
                new_features = torch_scatter.scatter_add(
                    torch.transpose(neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), 0, 1),
                    index[:, 0])
                new_features = torch.transpose(new_features, 0, 1)
                features_list.append(new_features)

            if self.attn_heads_reduction == 'concat':
                features = torch.cat(features_list)

            features = self.activation(features)
            outputs.append(features)

        outputs = torch.cat(outputs, dim=1)
        proxy_att = torch.matmul(F.normalize(outputs, dim=-1),
                                 torch.transpose(F.normalize(self.proxy, dim=-1), 0, 1))
        proxy_att = F.softmax(proxy_att, dim=-1)  # eq.3
        proxy_feature = outputs - torch.matmul(proxy_att, self.proxy)

        if self.use_bias:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel) + self.bias)
        else:
            gate_rate = F.sigmoid(torch.matmul(proxy_feature, self.gate_kernel))
        outputs = gate_rate * outputs + (1 - gate_rate) * proxy_feature
        return outputs


# ============================================================
# __main__: Automated test suite for 2 ablated functions
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
    print("DualMatch: Core Model Component Tests")
    print("Automated Test Suite - 2 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    node_size = 4
    rel_size = 3
    time_size = 2
    triple_size = 6
    dim = 5
    depth = 2

    adj_index = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [0, 2],
        [1, 3],
    ]
    r_index = [
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 0],
        [4, 1],
        [5, 2],
    ]
    t_index = [
        [0, 0],
        [1, 1],
        [2, 0],
        [3, 1],
        [4, 0],
        [5, 1],
    ]
    r_val = [1.0] * triple_size
    ent_matrix = [[i, i] for i in range(node_size)]
    rel_matrix = [[0, 0], [1, 1], [2, 2], [3, 0]]
    time_matrix = [[0, 0], [1, 1], [2, 0], [3, 1]]

    # ==========================================================
    # Test 1/2: GraphAttention.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/2] GraphAttention.forward - relation/time graph propagation")
    graph_attention_checks = 10
    try:
        graph_attention = GraphAttention(
            node_size=node_size,
            rel_size=rel_size,
            triple_size=triple_size,
            time_size=time_size,
            dim=dim,
            depth=depth,
            device=device,
        )
        features = torch.randn(node_size, dim, device=device)
        rel_emb = torch.randn(rel_size, dim, device=device, requires_grad=True)
        time_emb = torch.randn(time_size, dim, device=device, requires_grad=True)

        rel_output = graph_attention([features, rel_emb, adj_index, r_index, r_val], 0)
        time_output = graph_attention([features, time_emb, adj_index, t_index, r_val], 1)

        check("GraphAttention relation output not None", rel_output is not None)
        check("GraphAttention time output not None", time_output is not None)
        if rel_output is not None and time_output is not None:
            expected_graph_shape = (node_size, dim * (depth + 1))
            check("GraphAttention relation output shape", tuple(rel_output.shape) == expected_graph_shape,
                  f"expected {expected_graph_shape}, got {tuple(rel_output.shape)}")
            check("GraphAttention time output shape", tuple(time_output.shape) == expected_graph_shape,
                  f"expected {expected_graph_shape}, got {tuple(time_output.shape)}")
            check("GraphAttention relation output finite", torch.isfinite(rel_output).all().item())
            check("GraphAttention time output finite", torch.isfinite(time_output).all().item())
            check("GraphAttention rel/time branch propagation",
                  not torch.allclose(rel_output, time_output, atol=1e-5),
                  "relation and time branches should react to different sparse feature spaces")
            loss = rel_output.sum() + time_output.sum()
            loss.backward()
            check("GraphAttention gate kernel receives gradient",
                  graph_attention.gate_kernel.grad is not None
                  and torch.isfinite(graph_attention.gate_kernel.grad).all().item()
                  and graph_attention.gate_kernel.grad.abs().sum().item() > 0)
            check("GraphAttention relation embedding receives gradient",
                  rel_emb.grad is not None
                  and torch.isfinite(rel_emb.grad).all().item()
                  and rel_emb.grad.abs().sum().item() > 0)
            check("GraphAttention time embedding receives gradient",
                  time_emb.grad is not None
                  and torch.isfinite(time_emb.grad).all().item()
                  and time_emb.grad.abs().sum().item() > 0)
        else:
            skip_checks(graph_attention_checks - 2, "GraphAttention returned None")
    except Exception as exc:
        skip_checks(graph_attention_checks, f"GraphAttention.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/2: OverAll.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 2/2] OverAll.forward - dual entity/relation/time assembly")
    overall_checks = 8
    original_dual_no_time = global_args.dual_no_time
    try:
        model = OverAll(
            node_size=node_size,
            node_hidden=dim,
            rel_size=rel_size,
            rel_hidden=dim,
            time_size=time_size,
            triple_size=triple_size,
            rel_matrix=rel_matrix,
            ent_matrix=ent_matrix,
            time_matrix=time_matrix,
            dropout_rate=0,
            depth=depth,
            device=device,
        )
        model.eval()
        inputs = [adj_index, r_index, r_val, t_index]

        global_args.dual_no_time = False
        dual_output = model(inputs)
        global_args.dual_no_time = True
        rel_only_output = model(inputs)
        global_args.dual_no_time = original_dual_no_time

        check("OverAll dual output not None", dual_output is not None)
        check("OverAll relation-only output not None", rel_only_output is not None)
        if dual_output is not None and rel_only_output is not None:
            expected_overall_shape = (node_size, 2 * dim * (depth + 1))
            half_width = dim * (depth + 1)
            check("OverAll dual output shape", tuple(dual_output.shape) == expected_overall_shape,
                  f"expected {expected_overall_shape}, got {tuple(dual_output.shape)}")
            check("OverAll relation-only output shape", tuple(rel_only_output.shape) == expected_overall_shape,
                  f"expected {expected_overall_shape}, got {tuple(rel_only_output.shape)}")
            check("OverAll dual output finite", torch.isfinite(dual_output).all().item())
            check("OverAll relation-only output finite", torch.isfinite(rel_only_output).all().item())
            check("OverAll entity half stable across dual branch",
                  torch.allclose(dual_output[:, :half_width], rel_only_output[:, :half_width], atol=1e-5),
                  "entity encoder path should not depend on the time toggle")
            check("OverAll time toggle changes fused half",
                  not torch.allclose(dual_output[:, half_width:], rel_only_output[:, half_width:], atol=1e-5),
                  "fused relation/time half should change when time is enabled")
        else:
            skip_checks(overall_checks - 2, "OverAll returned None")
    except Exception as exc:
        global_args.dual_no_time = original_dual_no_time
        skip_checks(overall_checks, f"OverAll.forward raised {type(exc).__name__}: {exc}")
    finally:
        global_args.dual_no_time = original_dual_no_time
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
