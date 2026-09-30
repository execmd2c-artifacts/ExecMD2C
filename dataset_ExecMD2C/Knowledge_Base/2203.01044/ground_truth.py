"""
Self-contained SelfKG benchmark answer key.

This file extracts the core model components from model/layers_LaBSE_neighbor.py:
the self-supervised neighbor-aware entity embedder, its MoCo-style momentum
update and contrastive objective, and the masked multi-head graph attention
layer used to aggregate neighboring LaBSE entity representations.

Training loops, dataset loaders, FAISS evaluation, CLI wrappers, checkpoint I/O,
and preprocessing scripts are intentionally excluded.
"""

import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np


VOCAB_SIZE = 100000
LaBSE_DIM = 768
NEIGHBOR_SIZE = 20
MULTI_HEAD_DIM = 1
MAX_LEN = 88


def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class NCESoftmaxLoss(nn.Module):

    def __init__(self, device):
        super(NCESoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.squeeze()
        label = torch.zeros([batch_size]).to(self.device).long()
        loss = self.criterion(x, label)
        return loss


class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args

        self.device = torch.device(self.args.device)

        self.attn = BatchMultiHeadGraphAttention(self.device, self.args)
        
        self.attn_mlp = nn.Sequential(
            nn.Linear(LaBSE_DIM * 2, LaBSE_DIM),
        )

        # loss
        self.criterion = NCESoftmaxLoss(self.device)

        # batch queue
        self.batch_queue = []

    def contrastive_loss(self, pos_1, pos_2, neg_value):
        bsz = pos_1.shape[0]
        l_pos = torch.bmm(pos_1.view(bsz, 1, -1), pos_2.view(bsz, -1, 1))
        l_pos = l_pos.view(bsz, 1)
        l_neg = torch.mm(pos_1.view(bsz, -1), neg_value.t())
        logits = torch.cat((l_pos, l_neg), dim=1)
        logits = logits.squeeze().contiguous()
        return self.criterion(logits / self.args.t)

    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.args.momentum
            key_param.data += (1 - self.args.momentum) * query_param.data
        self.eval()

    def forward(self, batch):
        batch = batch.to(self.device)
        batch_in = batch[:, :, :LaBSE_DIM]
        adj = batch[:, :, LaBSE_DIM:]

        center = batch_in[:, 0].to(self.device)
        center_neigh = batch_in.to(self.device)

        for i in range(0, self.args.gat_num):
            center_neigh = self.attn(center_neigh, adj.bool()).squeeze(1)
        
        center_neigh = center_neigh[:, 0]

        if self.args.center_norm:
            center = F.normalize(center, p=2, dim=1)
        if self.args.neighbor_norm:
            center_neigh = F.normalize(center_neigh, p=2, dim=1)
        if self.args.combine:
            out_hat = torch.cat((center, center_neigh), dim=1)
            out_hat = self.attn_mlp(out_hat)
            if self.args.emb_norm:
                out_hat = F.normalize(out_hat, p=2, dim=1)
        else:
            out_hat = center_neigh

        return out_hat


class BatchMultiHeadGraphAttention(nn.Module):
    def __init__(self, device, args, n_head=MULTI_HEAD_DIM, f_in=LaBSE_DIM, f_out=LaBSE_DIM, bias=True):
        super(BatchMultiHeadGraphAttention, self).__init__()
        self.device = device
        self.n_head = n_head
        self.w = Parameter(torch.Tensor(n_head, f_in, f_out))
        self.a_src = Parameter(torch.Tensor(n_head, f_out, 1))
        self.a_dst = Parameter(torch.Tensor(n_head, f_out, 1))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(args.dropout)
        if bias:
            self.bias = Parameter(torch.Tensor(f_out))
            nn.init.constant_(self.bias, 0)
        else:
            self.register_parameter('bias', None)

        nn.init.xavier_uniform_(self.w)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        bs, n = h.size()[:2]  # h is of size bs x n x f_in
        h_prime = torch.matmul(h.unsqueeze(1), self.w)  # bs x n_head x n x f_out
        attn_src = torch.matmul(torch.tanh(h_prime), self.a_src)  # bs x n_head x n x 1
        attn_dst = torch.matmul(torch.tanh(h_prime), self.a_dst)  # bs x n_head x n x 1
        attn = attn_src.expand(-1, -1, -1, n) + attn_dst.expand(-1, -1, -1, n).permute(0, 1, 3, 2)  # bs x n_head x n x n

        attn = self.leaky_relu(attn)
        mask = ~(adj.unsqueeze(1) | torch.eye(adj.shape[-1]).bool().to(self.device))  # bs x 1 x n x n
        attn.data.masked_fill_(mask, float("-inf"))
        attn = self.softmax(attn)  # bs x n_head x n x n
        attn = self.dropout(attn)
        # logging.info("attn: ", attn)
        # logging.info("attn.shape: ", attn.shape)
        output = torch.matmul(attn, h_prime)  # bs x n_head x n x f_out
        if self.bias is not None:
            return output + self.bias
        else:
            return output


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

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

    def make_args(**overrides):
        values = {
            "device": "cpu",
            "dropout": 0.0,
            "t": 0.08,
            "momentum": 0.25,
            "gat_num": 1,
            "center_norm": False,
            "neighbor_norm": True,
            "emb_norm": True,
            "combine": True,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    print("=" * 70)
    print("SelfKG benchmark: neighbor-aware self-supervised entity encoder")
    print("=" * 70)

    # ==============================================================
    # Test 1/4: BatchMultiHeadGraphAttention.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] BatchMultiHeadGraphAttention.forward")
    try:
        args = make_args(dropout=0.0)
        attn_layer = BatchMultiHeadGraphAttention(
            torch.device("cpu"), args, n_head=1, f_in=3, f_out=3, bias=True
        )
        with torch.no_grad():
            attn_layer.w.zero_()
            attn_layer.w[0].copy_(torch.eye(3))
            attn_layer.a_src.zero_()
            attn_layer.a_dst.zero_()
            attn_layer.bias.zero_()
        h = torch.randn(2, 4, 3, requires_grad=True)
        adj = torch.zeros(2, 4, 4, dtype=torch.bool)
        output = attn_layer(h, adj)
        check("graph attention output not None", output is not None)
        if output is None:
            skip_checks(5, "graph attention output missing")
        else:
            check("graph attention output shape", tuple(output.shape) == (2, 1, 4, 3), f"got {tuple(output.shape)}")
            check("graph attention output finite", torch.isfinite(output).all().item())
            expected = h.unsqueeze(1)
            check("self-loop mask preserves isolated nodes", torch.allclose(output, expected, atol=1e-6))
            output.sum().backward()
            check("graph attention input gradient", h.grad is not None and h.grad.abs().sum().item() > 0)
            check("graph attention parameter gradient", attn_layer.w.grad is not None and attn_layer.w.grad.abs().sum().item() > 0)
    except Exception as exc:
        skip_checks(6, f"BatchMultiHeadGraphAttention.forward raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 2/4: MyEmbedder.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] MyEmbedder.forward")
    try:
        args = make_args()
        embedder = MyEmbedder(args, VOCAB_SIZE)
        embedder.eval()
        batch_size = 2
        node_count = 4
        features = torch.randn(batch_size, node_count, LaBSE_DIM)
        adj = torch.zeros(batch_size, node_count, node_count)
        adj[:, 0, 1] = 1
        adj[:, 1, 0] = 1
        batch = torch.cat([features, adj], dim=2)
        encoded = embedder(batch)
        check("embedder output not None", encoded is not None)
        if encoded is None:
            skip_checks(4, "embedder output missing")
        else:
            check("embedder output shape", tuple(encoded.shape) == (2, LaBSE_DIM), f"got {tuple(encoded.shape)}")
            check("embedder output finite", torch.isfinite(encoded).all().item())
            norms = encoded.norm(p=2, dim=1)
            check("embedder output normalized", torch.allclose(norms, torch.ones_like(norms), atol=1e-5))
            encoded.sum().backward()
            grad_ok = embedder.attn_mlp[0].weight.grad is not None and embedder.attn_mlp[0].weight.grad.abs().sum().item() > 0
            check("embedder projection gradient", grad_ok)
    except Exception as exc:
        skip_checks(5, f"MyEmbedder.forward raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 3/4: MyEmbedder.contrastive_loss
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] MyEmbedder.contrastive_loss")
    try:
        args = make_args(t=0.2)
        embedder = MyEmbedder(args, VOCAB_SIZE)
        pos_1 = F.normalize(torch.randn(3, LaBSE_DIM), p=2, dim=1).requires_grad_()
        pos_2 = pos_1.detach().clone()
        neg_value = F.normalize(torch.randn(5, LaBSE_DIM), p=2, dim=1)
        loss = embedder.contrastive_loss(pos_1, pos_2, neg_value)
        check("contrastive loss not None", loss is not None)
        if loss is None:
            skip_checks(4, "contrastive loss missing")
        else:
            check("contrastive loss scalar", tuple(loss.shape) == (), f"got {tuple(loss.shape)}")
            check("contrastive loss finite", torch.isfinite(loss).item())
            loss.backward()
            check("contrastive loss gradient", pos_1.grad is not None and pos_1.grad.abs().sum().item() > 0)
            shuffled_pos = pos_2.flip(0)
            worse_loss = embedder.contrastive_loss(pos_1.detach(), shuffled_pos, neg_value)
            check("positive pair ordering matters", loss.item() <= worse_loss.item() + 1e-6)
    except Exception as exc:
        skip_checks(5, f"MyEmbedder.contrastive_loss raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 4/4: MyEmbedder.update
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] MyEmbedder.update")
    try:
        args = make_args(momentum=0.25)
        key_model = MyEmbedder(args, VOCAB_SIZE)
        query_model = MyEmbedder(args, VOCAB_SIZE)
        with torch.no_grad():
            for param in key_model.parameters():
                param.fill_(2.0)
            for param in query_model.parameters():
                param.fill_(10.0)
        key_model.train()
        key_model.update(query_model)
        first_param = next(key_model.parameters())
        expected_value = torch.full_like(first_param, 8.0)
        check("momentum update returns model", key_model is not None)
        check("momentum update weighted average", torch.allclose(first_param, expected_value))
        check("momentum update switches eval mode", not key_model.training)
        all_params_ok = all(torch.allclose(param, torch.full_like(param, 8.0)) for param in key_model.parameters())
        check("momentum update covers all params", all_params_ok)
    except Exception as exc:
        skip_checks(4, f"MyEmbedder.update raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
