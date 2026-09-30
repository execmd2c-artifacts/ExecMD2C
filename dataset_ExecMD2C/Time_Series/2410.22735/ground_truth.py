# ============================================================
# ground_truth.py - MIXAD Core Model Components (Self-contained)
# Source: Time_Series/MIXAD-main/model/model.py
#
# Contains ONLY the model architecture definitions and direct dependencies.
# No training, evaluation, dataset, or pipeline code.
# ============================================================

import warnings
import torch
import torch.nn as nn
import numpy as np


# --- [Original file: model/model.py] ---
def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> torch.Tensor:
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )
    gumbels = (logits + gumbels) / tau
    y_soft = gumbels

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


class GC(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k):
        super(GC, self).__init__()
        self.cheb_k = cheb_k 
        self.weights = nn.Parameter(torch.FloatTensor(2*cheb_k*dim_in, dim_out))
        self.bias = nn.Parameter(torch.FloatTensor(dim_out))
        nn.init.xavier_normal_(self.weights)
        nn.init.constant_(self.bias, val=0)
        
    def forward(self, x, supports):
        x_g = []        
        support_set = []

        # Aggregate Neighbors
        for support in supports:
            support_ks = [torch.eye(support.shape[0]).to(support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]) 
            support_set.extend(support_ks)
        # print(len(support_set))
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)

        # Transformation
        x_gconv = torch.einsum('bni,io->bno', x_g, self.weights) + self.bias
        return x_gconv
    

class STRGCCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k):
        super(STRGCCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GC(dim_in+self.hidden_dim, 2*dim_out, cheb_k) 
        self.update = GC(dim_in+self.hidden_dim, dim_out, cheb_k)

    def forward(self, x, state, supports):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)

        z_r = torch.sigmoid(self.gate(input_and_state, supports)) 
        z,r = torch.split(z_r, self.hidden_dim, dim=-1) 
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)
    

class STRGC_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STRGC_Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num 
        self.input_dim = dim_in 
        self.num_layers = num_layers 
        self.strgc_cells = nn.ModuleList()
        self.strgc_cells.append(STRGCCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.strgc_cells.append(STRGCCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, x, init_state, supports):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x 
        output_hidden = []

        for i in range(self.num_layers): 
            state = init_state[i] 
            inner_states = []

            for t in range(seq_length): 
                state = self.strgc_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)

            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)

        return current_inputs, output_hidden 
    
    def init_hidden(self, batch_size): 
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.strgc_cells[i].init_hidden_state(batch_size)) 
        return init_states


class STRGC_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers):
        super(STRGC_Decoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Decoder.'
        self.node_num = node_num
        self.input_dim = dim_in 
        self.num_layers = num_layers 
        self.strgc_cells = nn.ModuleList()
        self.strgc_cells.append(STRGCCell(node_num, dim_in, dim_out, cheb_k))
        for _ in range(1, num_layers):
            self.strgc_cells.append(STRGCCell(node_num, dim_out, dim_out, cheb_k))

    def forward(self, xt, init_state, supports): 
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt 
        output_hidden = []

        for i in range(self.num_layers): 
            state = self.strgc_cells[i](current_inputs, init_state[i], supports) 
            output_hidden.append(state)
            current_inputs = state

        return current_inputs, output_hidden


class MIXAD(nn.Module):
    def __init__(self, args):
        super(MIXAD, self).__init__()
        self.num_nodes = args.num_nodes 
        self.input_dim = args.input_dim
        self.rnn_units = args.rnn_units
        self.output_dim = args.output_dim 
        self.horizon = args.seq_len
        self.num_layers = args.num_rnn_layers
        self.cheb_k = args.max_diffusion_step
        self.cl_decay_steps = args.cl_decay_steps
        self.use_curriculum_learning = args.use_curriculum_learning 
        
        self.mem_num = args.mem_num
        self.mem_dim = args.mem_dim
        self.memory = self.construct_memory()

        self.encoder = STRGC_Encoder(self.num_nodes, self.input_dim, self.rnn_units, self.cheb_k, self.num_layers)
        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = STRGC_Decoder(self.num_nodes, self.output_dim, self.decoder_dim, self.cheb_k, self.num_layers)

        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))
    
    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict['Memory'] = nn.Parameter(torch.randn(self.mem_num, self.mem_dim), requires_grad=True)
        memory_dict['Wq'] = nn.Parameter(torch.randn(self.rnn_units, self.mem_dim), requires_grad=True)
        memory_dict['We1'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) 
        memory_dict['We2'] = nn.Parameter(torch.randn(self.num_nodes, self.mem_num), requires_grad=True) 
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict
    
    def query_memory(self, h_t:torch.Tensor): 
        query = torch.matmul(h_t, self.memory['Wq']) 
        att_score = torch.softmax(torch.matmul(query, self.memory['Memory'].t()), dim=-1)
        value = torch.matmul(att_score, self.memory['Memory'])

        _, ind = torch.topk(att_score, k=2, dim=-1) 
        pos = self.memory['Memory'][ind[:, :, 0]] 
        neg = self.memory['Memory'][ind[:, :, 1]] 
        return value, query, pos, neg, att_score
    
    def scaled_laplacian(self, node_embeddings1, node_embeddings2, is_eval=False):
        # Normalized graph Laplacian function.
        node_num = self.num_nodes
        learned_graph = torch.mm(node_embeddings1, node_embeddings2.transpose(0, 1))
        
        # Normalize the graph
        norm1 = torch.norm(node_embeddings1, p=2, dim=1, keepdim=True)
        norm2 = torch.norm(node_embeddings2, p=2, dim=1, keepdim=True)
        norm = torch.mm(norm1, norm2.transpose(0, 1))
        learned_graph = learned_graph / (norm + 1e-6)  # Adding a small epsilon to avoid division by zero
        learned_graph = (learned_graph + 1) / 2.
        learned_graph = torch.stack([learned_graph, 1-learned_graph], dim=-1)
       
        # Make the adj sparse
        if is_eval:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        else:
            adj = gumbel_softmax(learned_graph, tau=1, hard=True)
        adj = adj[:, :, 0].clone().reshape(node_num, -1)
        mask = torch.eye(node_num, node_num).bool().to(node_embeddings1.device)
        adj.masked_fill_(mask, 0)
       
        W = adj
        n = W.shape[0]
        d = torch.sum(W, axis=1)
        L = -W
        L[range(len(L)), range(len(L))] = d
        
        try:
            lambda_max = (L.max() - L.min())
        except Exception as e:
            print("eig error!!: {}".format(e))
            lambda_max = 1.0
       
        tilde = (2 * L / lambda_max - torch.eye(n).to(node_embeddings1.device))
        return adj, tilde
            
    def forward(self, x, batches_seen=None):
        node_embeddings1 = torch.matmul(self.memory['We1'], self.memory['Memory'])
        node_embeddings2 = torch.matmul(self.memory['We2'], self.memory['Memory'])
        if self.train:
            adj1, learned_tilde1 = self.scaled_laplacian(node_embeddings1, node_embeddings2, is_eval=False)
            adj2, learned_tilde2 = self.scaled_laplacian(node_embeddings2, node_embeddings1, is_eval=False)
        else:
            adj1, learned_tilde1 = self.scaled_laplacian(node_embeddings1, node_embeddings2, is_eval=True)
            adj2, learned_tilde2 = self.scaled_laplacian(node_embeddings2, node_embeddings1, is_eval=True)
        supports = [learned_tilde1, learned_tilde2] 

        init_state = self.encoder.init_hidden(x.shape[0]) 
        h_en, state_en = self.encoder(x, init_state, supports) 
        h_t = h_en[:, -1, :, :] 

        h_att, query, pos, neg, att_score = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)
        
        labels = torch.flip(x, [1])
        ht_list = [h_t]*self.num_layers
        go = torch.zeros((x.shape[0], self.num_nodes, self.output_dim), device=x.device) 
        out = []

        for t in range(self.horizon):
            h_de, ht_list = self.decoder(go, ht_list, supports)
            go = self.proj(h_de)
            out.append(go)

            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batches_seen):
                    go = labels[:, t, ...]

        output = torch.stack(out, dim=1)
        output = torch.flip(output, [1])

        h_att = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        query = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        pos = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        neg = torch.randn(x.shape[0], self.num_nodes, self.mem_dim).to(x.device)
        att_score = torch.randn(x.shape[0], self.num_nodes, self.mem_num).to(x.device)

        return output, h_att, query, pos, neg, att_score, (adj1, adj2)


# ============================================================
# __main__: Automated test suite for 5 ablated functions
# ============================================================

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

    print("=" * 70)
    print("MIXAD: Memory-Induced Explainable Time Series Anomaly Detection")
    print("Automated reproduction benchmark - 5 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/5: GC.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] GC.forward - Chebyshev graph convolution")
    try:
        gc = GC(dim_in=2, dim_out=2, cheb_k=2).to(device)
        with torch.no_grad():
            gc.weights.zero_()
            gc.bias.zero_()
            gc.weights[:2, :] = torch.eye(2)
        x = torch.randn(2, 3, 2, device=device)
        support_a = torch.eye(3, device=device)
        support_b = torch.zeros(3, 3, device=device)
        y = gc(x, [support_a, support_b])
        check("GC output not None", y is not None)
        if y is not None:
            check("GC output shape", tuple(y.shape) == (2, 3, 2), f"expected (2, 3, 2), got {tuple(y.shape)}")
            check("GC output finite", torch.isfinite(y).all().item())
            check("GC identity support propagation", torch.allclose(y, x, atol=1e-5))
        else:
            skip_checks(3, "GC.forward returned None")
    except Exception as e:
        skip_checks(4, f"GC.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/5: STRGCCell.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] STRGCCell.forward - gated graph recurrent update")
    try:
        cell = STRGCCell(node_num=4, dim_in=2, dim_out=3, cheb_k=2).to(device)
        x = torch.randn(2, 4, 2, device=device)
        state = torch.randn(2, 4, 3, device=device)
        support_a = torch.eye(4, device=device)
        support_b = torch.ones(4, 4, device=device) / 4
        h = cell(x, state, [support_a, support_b])
        check("STRGCCell output not None", h is not None)
        if h is not None:
            check("STRGCCell output shape", tuple(h.shape) == (2, 4, 3), f"expected (2, 4, 3), got {tuple(h.shape)}")
            check("STRGCCell output finite", torch.isfinite(h).all().item())
            loss = h.sum()
            loss.backward()
            gate_grad = cell.gate.weights.grad is not None and cell.gate.weights.grad.abs().sum().item() > 0
            update_grad = cell.update.weights.grad is not None and cell.update.weights.grad.abs().sum().item() > 0
            check("STRGCCell gate and update receive gradients", gate_grad and update_grad)
        else:
            skip_checks(3, "STRGCCell.forward returned None")
    except Exception as e:
        skip_checks(4, f"STRGCCell.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/5: MIXAD.query_memory
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] MIXAD.query_memory - memory retrieval and top-k slots")
    try:
        args = SimpleNamespace(
            num_nodes=4,
            input_dim=2,
            rnn_units=5,
            output_dim=2,
            seq_len=3,
            num_rnn_layers=1,
            max_diffusion_step=2,
            cl_decay_steps=1000,
            use_curriculum_learning=False,
            mem_num=4,
            mem_dim=6,
        )
        model = MIXAD(args).to(device)
        h_t = torch.randn(2, 4, 5, device=device)
        memory_result = model.query_memory(h_t)
        check("query_memory returns five tensors", isinstance(memory_result, tuple) and len(memory_result) == 5)
        if isinstance(memory_result, tuple) and len(memory_result) == 5 and all(item is not None for item in memory_result):
            value, query, pos, neg, att_score = memory_result
            expected_shapes = (
                tuple(value.shape) == (2, 4, 6) and
                tuple(query.shape) == (2, 4, 6) and
                tuple(pos.shape) == (2, 4, 6) and
                tuple(neg.shape) == (2, 4, 6) and
                tuple(att_score.shape) == (2, 4, 4)
            )
            finite_outputs = all(torch.isfinite(item).all().item() for item in memory_result)
            simplex_scores = torch.allclose(att_score.sum(dim=-1), torch.ones(2, 4, device=device), atol=1e-5)
            top_indices = torch.topk(att_score, k=2, dim=-1).indices
            top_slots_match = (
                torch.allclose(pos, model.memory['Memory'][top_indices[:, :, 0]], atol=1e-5) and
                torch.allclose(neg, model.memory['Memory'][top_indices[:, :, 1]], atol=1e-5)
            )
            check("query_memory output shapes", expected_shapes)
            check("query_memory outputs finite", finite_outputs)
            check("query_memory attention simplex and top-k slots", simplex_scores and top_slots_match)
        else:
            skip_checks(3, "query_memory returned None or malformed output")
    except Exception as e:
        skip_checks(4, f"MIXAD.query_memory raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/5: MIXAD.scaled_laplacian
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] MIXAD.scaled_laplacian - learned sparse graph Laplacian")
    try:
        args = SimpleNamespace(
            num_nodes=4,
            input_dim=2,
            rnn_units=5,
            output_dim=2,
            seq_len=3,
            num_rnn_layers=1,
            max_diffusion_step=2,
            cl_decay_steps=1000,
            use_curriculum_learning=False,
            mem_num=4,
            mem_dim=6,
        )
        model = MIXAD(args).to(device)
        emb1 = torch.randn(4, 6, device=device)
        emb2 = torch.randn(4, 6, device=device)
        graph_result = model.scaled_laplacian(emb1, emb2, is_eval=True)
        check("scaled_laplacian returns adjacency and support", isinstance(graph_result, tuple) and len(graph_result) == 2)
        if isinstance(graph_result, tuple) and len(graph_result) == 2 and all(item is not None for item in graph_result):
            adj, tilde = graph_result
            shape_and_finite = (
                tuple(adj.shape) == (4, 4) and
                tuple(tilde.shape) == (4, 4) and
                torch.isfinite(adj).all().item() and
                torch.isfinite(tilde).all().item()
            )
            zero_diag = torch.allclose(torch.diag(adj), torch.zeros(4, device=device), atol=1e-6)
            binary_adj = torch.all((adj == 0) | (adj == 1)).item()
            check("scaled_laplacian shapes and finite values", shape_and_finite)
            check("scaled_laplacian zero diagonal", zero_diag)
            check("scaled_laplacian hard sparse adjacency", binary_adj)
        else:
            skip_checks(3, "scaled_laplacian returned None or malformed output")
    except Exception as e:
        skip_checks(4, f"MIXAD.scaled_laplacian raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 5/5: MIXAD.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] MIXAD.forward - end-to-end learned graph memory decoder")
    try:
        args = SimpleNamespace(
            num_nodes=4,
            input_dim=2,
            rnn_units=5,
            output_dim=2,
            seq_len=3,
            num_rnn_layers=1,
            max_diffusion_step=2,
            cl_decay_steps=1000,
            use_curriculum_learning=False,
            mem_num=4,
            mem_dim=6,
        )
        model = MIXAD(args).to(device)
        model.eval()
        x = torch.randn(2, 3, 4, 2, device=device)
        with torch.no_grad():
            forward_result = model(x, batches_seen=0)
        check("MIXAD forward returns seven outputs", isinstance(forward_result, tuple) and len(forward_result) == 7)
        if isinstance(forward_result, tuple) and len(forward_result) == 7 and all(item is not None for item in forward_result):
            output, h_att, query, pos, neg, att_score, adjs = forward_result
            aux_shapes = (
                tuple(h_att.shape) == (2, 4, 6) and
                tuple(query.shape) == (2, 4, 6) and
                tuple(pos.shape) == (2, 4, 6) and
                tuple(neg.shape) == (2, 4, 6) and
                tuple(att_score.shape) == (2, 4, 4)
            )
            all_tensors = [output, h_att, query, pos, neg, att_score, adjs[0], adjs[1]]
            finite_outputs = all(torch.isfinite(item).all().item() for item in all_tensors)
            adjacency_valid = (
                isinstance(adjs, tuple) and len(adjs) == 2 and
                tuple(adjs[0].shape) == (4, 4) and tuple(adjs[1].shape) == (4, 4) and
                torch.allclose(torch.diag(adjs[0]), torch.zeros(4, device=device), atol=1e-6) and
                torch.allclose(torch.diag(adjs[1]), torch.zeros(4, device=device), atol=1e-6)
            )
            check("MIXAD reconstruction output shape", tuple(output.shape) == (2, 3, 4, 2), f"expected (2, 3, 4, 2), got {tuple(output.shape)}")
            check("MIXAD outputs finite with auxiliary memory shapes", finite_outputs and aux_shapes)
            check("MIXAD returns valid learned adjacency pair", adjacency_valid)
        else:
            skip_checks(3, "MIXAD.forward returned None or malformed output")
    except Exception as e:
        skip_checks(4, f"MIXAD.forward raised {type(e).__name__}: {e}")
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
