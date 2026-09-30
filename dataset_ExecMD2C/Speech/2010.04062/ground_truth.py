# ============================================================
# ground_truth.py - SimTA Core Model Components
# Source:
#   Speech/SimTA-main/models/simta.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: models/simta.py] ---
class SimTALayer(nn.Module):

    def __init__(self, d_in=1, d_model=256, lamb=1, beta=0):
        super().__init__()
        self.linear = nn.Linear(d_in, d_model)
        self.lamb = nn.Parameter(torch.tensor(lamb, dtype=torch.float))
        self.beta = nn.Parameter(torch.tensor(beta, dtype=torch.float))

    @staticmethod
    def _get_attn_matrix(tau):
        """
        tau: A time interval vector of dimension batch_size * T;
        return: A time-related attention matrix of dimension batch_size * T * T.
        """
        with torch.no_grad():
            d_t = tau.size(1)

            # Repeat tau d_t times along the second dimension, set all elemen-
            # ts on and below the main diagonal to 0, and compute the cumulat-
            # ive sum along the last dimension. Now the element on the i-th r-
            # ow and j-th column of the matrix equals to the duration between
            # tau_i and tau_j.
            tau = torch.unsqueeze(tau, dim=1).repeat(1, d_t, 1)
            t_attn = torch.cumsum(torch.triu(tau, diagonal=1),
                dim=-1).transpose(1, 2)

        return t_attn

    @staticmethod
    def _apply_ninf_mask(attn):
        NINF = -9e8
        d_t = attn.size(-1)
        ninf_mask = torch.triu(torch.ones(d_t, d_t)).bool().logical_not()\
            .transpose(0, 1).to(attn.device)
        attn.masked_fill_(ninf_mask, NINF)

    def forward(self, x, tau):
        """
        x: A feature vector of dimension B * T * d_in;
        tau: A time interval vector of dimension B * T;
        return: B * T * d_model.
        """
        x = F.elu(self.linear(x))
        t_attn = self._get_attn_matrix(tau)
        t_attn = -F.relu(self.lamb) * t_attn + self.beta
        self._apply_ninf_mask(t_attn)

        return torch.bmm(F.softmax(t_attn, dim=-1), x)


class SimTABlock(nn.Module):

    def __init__(self, d_in=1, d_model=256, lamb=1, beta=0, num_layers=4):
        super().__init__()

        for i in range(num_layers):
            if i == 0:
                self.add_module("simta_{}".format(i),
                    SimTALayer(d_in, d_model, lamb, beta))
            else:
                self.add_module("simta_{}".format(i),
                    SimTALayer(d_model, d_model, lamb, beta))

    def forward(self, x, tau):
        for _, module in self.named_children():
            x = module(x, tau)

        return x


class SimTANet(nn.Module):

    def __init__(self, d_in=1, d_model=256, d_out=3,
            lamb=1, beta=0, num_layers=4):
        super().__init__()
        
        self.linear = nn.Linear(d_in, d_model)
        self.simta = SimTABlock(d_model, d_model, lamb, beta, num_layers)
        self.regressor = nn.Linear(d_model, d_out)
    
    def forward(self, x, tau):
        if len(x.size()) == 2:
            x = torch.unsqueeze(x, dim=-1)
        x = self.linear(x)
        # x = self.simta(x, tau).mean(dim=1)
        x = self.simta(x, tau)[:, -1, :]
        y = self.regressor(x)

        return y


# ============================================================
# __main__: Automated test suite for 4 ablated functions
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
    print("SimTA: Simple Temporal Attention")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: SimTALayer._get_attn_matrix
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] SimTALayer._get_attn_matrix - cumulative interval attention distances")
    try:
        tau = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device)
        attn = SimTALayer._get_attn_matrix(tau)
        expected = torch.tensor([[[0.0, 0.0, 0.0, 0.0],
                                  [2.0, 0.0, 0.0, 0.0],
                                  [5.0, 3.0, 0.0, 0.0],
                                  [9.0, 7.0, 4.0, 0.0]]], device=device)
        check("_get_attn_matrix output not None", attn is not None)
        if attn is not None:
            check("_get_attn_matrix output shape", tuple(attn.shape) == (1, 4, 4), f"expected (1, 4, 4), got {tuple(attn.shape)}")
            check("_get_attn_matrix values match cumulative intervals", torch.allclose(attn, expected))
            check("_get_attn_matrix diagonal zeros", torch.allclose(torch.diagonal(attn, dim1=1, dim2=2), torch.zeros(1, 4, device=device)))
            check("_get_attn_matrix detached from gradients", not attn.requires_grad)
        else:
            skip_checks(4, "SimTALayer._get_attn_matrix returned None")
    except Exception as e:
        skip_checks(5, f"SimTALayer._get_attn_matrix raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: SimTALayer._apply_ninf_mask
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] SimTALayer._apply_ninf_mask - causal future-position masking")
    try:
        attn = torch.zeros(2, 4, 4, device=device)
        result = SimTALayer._apply_ninf_mask(attn)
        check("_apply_ninf_mask returns None in-place", result is None)
        check("_apply_ninf_mask preserves shape", tuple(attn.shape) == (2, 4, 4))
        history_ok = torch.all(attn[:, torch.tril(torch.ones(4, 4, device=device), diagonal=0).bool()] == 0)
        check("_apply_ninf_mask keeps current/history positions unmasked", history_ok)
        future_mask = torch.triu(torch.ones(4, 4, device=device), diagonal=1).bool()
        check("_apply_ninf_mask writes large negative future mask", torch.all(attn[:, future_mask] < -1e8))
        check("_apply_ninf_mask finite sentinel values", torch.isfinite(attn).all().item())
    except Exception as e:
        skip_checks(5, f"SimTALayer._apply_ninf_mask raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: SimTALayer.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] SimTALayer.forward - time-decayed attention aggregation")
    try:
        layer = SimTALayer(d_in=2, d_model=4, lamb=1.0, beta=0.5).to(device)
        x = torch.randn(3, 5, 2, device=device, requires_grad=True)
        tau = torch.rand(3, 5, device=device) + 0.1
        y = layer(x, tau)
        check("SimTALayer output not None", y is not None)
        if y is not None:
            check("SimTALayer output shape", tuple(y.shape) == (3, 5, 4), f"expected (3, 5, 4), got {tuple(y.shape)}")
            check("SimTALayer output finite", torch.isfinite(y).all().item())
            with torch.no_grad():
                logits = -F.relu(layer.lamb) * layer._get_attn_matrix(tau) + layer.beta
                layer._apply_ninf_mask(logits)
                probs = F.softmax(logits, dim=-1)
            first_step_self_only = torch.allclose(probs[:, 0, 0], torch.ones(3, device=device), atol=1e-6)
            check("SimTALayer first timestep attends only to itself", first_step_self_only)
            (y ** 2).sum().backward()
            grad_ok = layer.linear.weight.grad is not None and layer.lamb.grad is not None and layer.beta.grad is not None
            check("SimTALayer routes gradient to projection and time parameters", grad_ok)
        else:
            skip_checks(4, "SimTALayer.forward returned None")
    except Exception as e:
        skip_checks(5, f"SimTALayer.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: SimTANet.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] SimTANet.forward - end-to-end temporal attention regression")
    try:
        model = SimTANet(d_in=1, d_model=8, d_out=3, lamb=1.0, beta=0.0, num_layers=2).to(device)
        x = torch.randn(4, 6, device=device, requires_grad=True)
        tau = torch.rand(4, 6, device=device) + 0.1
        y = model(x, tau)
        check("SimTANet output not None", y is not None)
        if y is not None:
            check("SimTANet output shape", tuple(y.shape) == (4, 3), f"expected (4, 3), got {tuple(y.shape)}")
            check("SimTANet output finite", torch.isfinite(y).all().item())
            check("SimTANet uses final timestep feature before regression", model.regressor.in_features == 8)
            (y ** 2).sum().backward()
            check("SimTANet routes gradient to initial projection", model.linear.weight.grad is not None and model.linear.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "SimTANet.forward returned None")
    except Exception as e:
        skip_checks(5, f"SimTANet.forward raised {type(e).__name__}: {e}")
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
