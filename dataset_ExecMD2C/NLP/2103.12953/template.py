from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel


# --- [Original file: learner/cluster_utils.py] ---
eps = 1e-8  

class KLDiv(nn.Module):    
    def forward(self, predict, target):
        assert predict.ndimension()==2,'Input dimension must be 2'
        target = target.detach()
        p1 = predict + eps
        t1 = target + eps
        logI = p1.log()
        logT = t1.log()
        TlogTdI = target * (logT - logI)
        kld = TlogTdI.sum(1)
        return kld

class KCL(nn.Module):
    def __init__(self):
        super(KCL,self).__init__()
        self.kld = KLDiv()

    def forward(self, prob1, prob2):
        kld = self.kld(prob1, prob2)
        return kld.mean()
    
def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    TODO: Build SCCL's sharpened clustering target distribution.

    Input:
        batch: Tensor of shape (batch_size, num_clusters), containing row-normalized
            soft cluster assignments for a minibatch.

    Output:
        Tensor of shape (batch_size, num_clusters), row-normalized after sharpening.

"""
    pass


# --- [Original file: learner/contrastive_utils.py] ---
class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        """
        TODO: Compute SCCL's paired instance contrastive loss.

        Input:
            features_1: Tensor of shape (batch_size, feature_dim), usually L2-normalized
                projected embeddings from one view.
            features_2: Tensor of shape (batch_size, feature_dim), matching positive views
                in the same row order.

        Output:
            Dictionary with:
            - "loss": scalar Tensor contrastive loss.
            - "pos_mean", "neg_mean": detached CPU/numpy summaries.
            - "pos": detached vector of positive scores with shape (2 * batch_size,).
            - "neg": detached matrix of negative scores with shape
              (2 * batch_size, 2 * batch_size - 2).

"""
        pass


# --- [Original file: models/Transformers.py] ---
class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, cluster_centers=None, alpha=1.0):
        super(SCCLBert, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        self.alpha = alpha
        
        # Instance-CL head
        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128))
        
        # Clustering head
        initial_cluster_centers = torch.tensor(
            cluster_centers, dtype=torch.float, requires_grad=True)
        self.cluster_centers = Parameter(initial_cluster_centers)
      
    
    def forward(self, input_ids, attention_mask, task_type="virtual"):
        """
        TODO: Route SCCLBert inputs through evaluate, virtual-augmentation, or explicit-augmentation paths.

        Input:
            input_ids:
                - evaluate: LongTensor of shape (batch_size, seq_len)
                - virtual: LongTensor of shape (batch_size, 2, seq_len)
                - explicit: LongTensor of shape (batch_size, 3, seq_len)
            attention_mask: same leading shape as input_ids.
            task_type: one of "evaluate", "virtual", or "explicit".

        Output:
            - evaluate: Tensor of shape (batch_size, hidden_size).
            - virtual: tuple of two tensors, each (batch_size, hidden_size).
            - explicit: tuple of three tensors, each (batch_size, hidden_size).

"""
        pass
      
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output
    

    def get_cluster_prob(self, embeddings):
        """
        TODO: Compute SCCL soft cluster assignments from embeddings and trainable centers.

        Input:
            embeddings: Tensor of shape (batch_size, hidden_size).

        Output:
            Tensor of shape (batch_size, num_clusters), where each row sums to one.

"""
        pass

    def local_consistency(self, embd0, embd1, embd2, criterion):
        """
        TODO: Compute local distributional consistency across original and augmented embeddings.

        Input:
            embd0: Tensor of shape (batch_size, hidden_size), original-view embeddings.
            embd1: Tensor of shape (batch_size, hidden_size), first augmented-view embeddings.
            embd2: Tensor of shape (batch_size, hidden_size), second augmented-view embeddings.
            criterion: callable that compares two cluster-probability tensors.

        Output:
            Scalar Tensor equal to the summed consistency penalty for both augmented views.

"""
        pass
    
    def contrast_logits(self, embd1, embd2=None):
        """
        TODO: Project SCCL embeddings for instance-level contrastive learning.

        Input:
            embd1: Tensor of shape (batch_size, hidden_size).
            embd2: optional Tensor of shape (batch_size, hidden_size).

        Output:
            If embd2 is provided, return a pair of tensors of shape (batch_size, 128).
            Otherwise return one tensor of shape (batch_size, 128).

"""
        pass


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
    print("SCCL benchmark: clustering supported by contrastive learning")
    print("=" * 70)

    class FakeBert(nn.Module):
        def __init__(self, hidden_size=4):
            super().__init__()
            self.config = type("Config", (), {"hidden_size": hidden_size})()

        def forward(self, input_ids, attention_mask):
            del attention_mask
            base = input_ids.float().unsqueeze(-1)
            hidden = torch.cat([base, base + 1.0, base * 2.0, torch.ones_like(base)], dim=-1)
            return (hidden,)

    cluster_centers = np.array(
        [
            [1.0, 2.0, 2.0, 1.0],
            [5.0, 6.0, 10.0, 1.0],
            [9.0, 10.0, 18.0, 1.0],
        ],
        dtype=np.float32,
    )
    model = SCCLBert(FakeBert(hidden_size=4), tokenizer=None, cluster_centers=cluster_centers, alpha=1.0)

    print("-" * 70)
    print("[Test 1/5] SCCLBert routing and masked mean embeddings")
    try:
        ids = torch.tensor([[1, 2, 99], [5, 6, 7]])
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
        eval_out = model(ids, mask, task_type="evaluate")
        check("evaluate output not None", eval_out is not None)
        if eval_out is not None:
            expected_first = torch.tensor([1.5, 2.5, 3.0, 1.0])
            check("evaluate output shape", eval_out.shape == (2, 4), str(tuple(eval_out.shape)))
            check("masked mean ignores padded tokens", torch.allclose(eval_out[0], expected_first, atol=1e-6), str(eval_out[0]))
        else:
            skip_checks(2, "evaluate output is None")
        pair_ids = torch.stack([ids, ids + 10], dim=1)
        pair_mask = torch.stack([mask, mask], dim=1)
        virtual_out = model(pair_ids, pair_mask, task_type="virtual")
        explicit_out = model(torch.stack([ids, ids + 10, ids + 20], dim=1), torch.stack([mask, mask, mask], dim=1), task_type="explicit")
        check("virtual returns two views", isinstance(virtual_out, tuple) and len(virtual_out) == 2)
        check("explicit returns three views", isinstance(explicit_out, tuple) and len(explicit_out) == 3)
    except Exception as exc:
        skip_checks(5, f"SCCLBert routing raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 2/5] SCCLBert cluster probabilities and target distribution")
    try:
        emb = torch.tensor([[1.0, 2.0, 2.0, 1.0], [8.5, 9.5, 17.0, 1.0]])
        prob = model.get_cluster_prob(emb)
        target = target_distribution(torch.tensor([[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32))
        check("cluster probability output not None", prob is not None)
        if prob is not None:
            check("cluster probability shape", prob.shape == (2, 3), str(tuple(prob.shape)))
            check("cluster probabilities sum to one", torch.allclose(prob.sum(dim=1), torch.ones(2), atol=1e-6), str(prob.sum(dim=1)))
            check("nearest cluster receives highest probability", torch.equal(prob.argmax(dim=1), torch.tensor([0, 2])), str(prob))
        else:
            skip_checks(3, "cluster probability is None")
        check("target distribution output not None", target is not None)
        if target is not None:
            check("target distribution shape", target.shape == (2, 3), str(tuple(target.shape)))
            check("target distribution preserves row-normalization", torch.allclose(target.sum(dim=1), torch.ones(2), atol=1e-6), str(target.sum(dim=1)))
            check("target distribution sharpens confident assignment", target[0, 0] > 0.8, str(target[0]))
        else:
            skip_checks(3, "target distribution is None")
    except Exception as exc:
        skip_checks(6, f"cluster probability test raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 3/5] SCCLBert contrast projection")
    try:
        emb1 = torch.randn(4, 4)
        emb2 = emb1 + 0.05 * torch.randn(4, 4)
        feat1, feat2 = model.contrast_logits(emb1, emb2)
        single = model.contrast_logits(emb1)
        check("contrast pair output not None", feat1 is not None and feat2 is not None)
        if feat1 is not None and feat2 is not None:
            check("contrast feature shapes", feat1.shape == (4, 128) and feat2.shape == (4, 128), f"{tuple(feat1.shape)}, {tuple(feat2.shape)}")
            check("contrast features are finite", torch.isfinite(feat1).all().item() and torch.isfinite(feat2).all().item())
            check("contrast features are unit-normalized", torch.allclose(feat1.norm(dim=1), torch.ones(4), atol=1e-5), str(feat1.norm(dim=1)))
        else:
            skip_checks(3, "contrast pair output is None")
        check("single-view contrast returns one tensor", isinstance(single, torch.Tensor) and single.shape == (4, 128))
    except Exception as exc:
        skip_checks(5, f"contrast projection raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 4/5] PairConLoss positive/negative masking")
    try:
        loss_fn = PairConLoss(temperature=0.5)
        f1 = F.normalize(torch.randn(5, 8), dim=1).requires_grad_(True)
        f2 = F.normalize(f1.detach() + 0.01 * torch.randn(5, 8), dim=1).requires_grad_(True)
        losses = loss_fn(f1, f2)
        check("PairConLoss returns dictionary", isinstance(losses, dict))
        if isinstance(losses, dict):
            check("PairConLoss loss key exists", "loss" in losses and losses["loss"] is not None)
            if "loss" in losses and losses["loss"] is not None:
                check("PairConLoss scalar finite loss", losses["loss"].dim() == 0 and torch.isfinite(losses["loss"]).item())
                check("PairConLoss positive vector length doubles batch", np.asarray(losses["pos"]).shape == (10,), str(np.asarray(losses["pos"]).shape))
                check("PairConLoss negatives exclude paired diagonal only", np.asarray(losses["neg"]).shape == (10, 8), str(np.asarray(losses["neg"]).shape))
                losses["loss"].backward()
                check("PairConLoss propagates gradients", f1.grad is not None and f2.grad is not None and torch.isfinite(f1.grad).all().item())
            else:
                skip_checks(4, "PairConLoss loss is missing")
        else:
            skip_checks(5, "PairConLoss did not return a dictionary")
    except Exception as exc:
        skip_checks(6, f"PairConLoss raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 5/5] KL clustering consistency")
    try:
        kcl = KCL()
        prob_a = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.4, 0.5]], dtype=torch.float32)
        prob_b = torch.tensor([[0.6, 0.25, 0.15], [0.2, 0.3, 0.5]], dtype=torch.float32)
        same = kcl(prob_a, prob_a)
        diff = kcl(prob_a, prob_b)
        emb0 = torch.tensor([[1.0, 2.0, 2.0, 1.0], [9.0, 10.0, 18.0, 1.0]])
        emb1 = emb0 + 0.1
        emb2 = emb0 - 0.1
        local = model.local_consistency(emb0, emb1, emb2, kcl)
        check("KCL same-distribution output not None", same is not None)
        check("KCL identical distributions near zero", torch.allclose(same, torch.tensor(0.0), atol=1e-6), str(same))
        check("KCL different distributions positive", diff.item() > 0.0, str(diff))
        check("local consistency output not None", local is not None)
        if local is not None:
            check("local consistency returns scalar", local.dim() == 0, str(local.shape))
            check("local consistency finite and non-negative", torch.isfinite(local).item() and local.item() >= 0.0, str(local))
        else:
            skip_checks(2, "local consistency is None")
    except Exception as exc:
        skip_checks(5, f"KL consistency raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
