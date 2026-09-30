# ============================================================
# ground_truth.py - Fortran2Cpp Core Model Components
# Source: Computer_Code/Fortran2Cpp-main
#
# Contains only the LoRA model adaptation layer and direct helper used by
# the fine-tuning path. No training loops, datasets, evaluation pipelines,
# web demo code, checkpoint I/O, or remote model loading.
# ============================================================

import math
import torch
from torch import nn
import torch.nn.functional as F


# --- [Original file: training/utils/module/lora.py] ---
class LinearLayer_LoRA(nn.Module):
    # an simple implementation of LoRA
    # for now only support Linear Layer
    def __init__(self,
                 weight,
                 lora_dim=0,
                 lora_scaling=1,
                 lora_droppout=0,
                 bias=None):
        super(LinearLayer_LoRA, self).__init__()
        self.weight = weight
        self.bias = bias

        if lora_dim <= 0:
            raise ValueError(
                "You are training to use LoRA, whose reduced dim should be larger than 1"
            )

        try:
            # for zero stage 3
            rows, columns = weight.ds_shape
        except:
            rows, columns = weight.shape
        self.lora_right_weight = nn.Parameter(torch.zeros(
            columns,
            lora_dim))  # apply transpose so in forward we do not need to
        self.lora_left_weight = nn.Parameter(torch.zeros(lora_dim, rows))
        self.lora_scaling = lora_scaling / lora_dim

        if lora_droppout > 0:
            self.lora_dropout = nn.Dropout(lora_droppout)
        else:
            self.lora_dropout = nn.Identity()

        self.reset_parameters()
        # disable the original weight gradient
        self.weight.requires_grad = False
        # fuse LoRA to the original weight
        self.fuse_lora = False

    def eval(self):
        self.lora_dropout.eval()

    #   self.fuse_lora_weight()

    def train(self, mode=True):
        self.lora_dropout.train(mode)
        # self.unfuse_lora_weight()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_right_weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_left_weight)

    def fuse_lora_weight(self):
        if not self.fuse_lora:
            self.weight.data += self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = True

    def unfuse_lora_weight(self):
        if self.fuse_lora:
            self.weight.data -= self.lora_scaling * torch.matmul(
                self.lora_left_weight.t(), self.lora_right_weight.t())
        self.fuse_lora = False

    def forward(self, input):
        if self.fuse_lora:
            return F.linear(input, self.weight, self.bias)
        else:
            return F.linear(
                input, self.weight,
                self.bias) + (self.lora_dropout(input) @ self.lora_right_weight
                              @ self.lora_left_weight) * self.lora_scaling


def only_optimize_lora_parameters(model):
    # turn off the gradient of all the parameters except the LoRA parameters
    for name, param in model.named_parameters():
        if "lora_right_weight" in name or "lora_left_weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    return model


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

    def make_lora_layer():
        weight = nn.Parameter(torch.arange(20, dtype=torch.float32).view(4, 5) / 10.0)
        bias = nn.Parameter(torch.linspace(-0.2, 0.2, 4))
        return LinearLayer_LoRA(weight, lora_dim=2, lora_scaling=2, lora_droppout=0.0, bias=bias)

    print("=" * 70)
    print("Fortran2Cpp: LoRA model-adaptation benchmark")
    print("Automated Test Suite - 5 ablated functions")
    print("=" * 70)
    print()

    print("-" * 60)
    print("[Test 1/5] LinearLayer_LoRA initialization")
    try:
        layer = make_lora_layer()
        check("LoRA right shape", tuple(layer.lora_right_weight.shape) == (5, 2), f"got {tuple(layer.lora_right_weight.shape)}")
        check("LoRA left shape", tuple(layer.lora_left_weight.shape) == (2, 4), f"got {tuple(layer.lora_left_weight.shape)}")
        check("LoRA left starts zero", torch.allclose(layer.lora_left_weight, torch.zeros_like(layer.lora_left_weight)))
        check("LoRA right finite", torch.isfinite(layer.lora_right_weight).all().item())
        check("base weight frozen", layer.weight.requires_grad is False)
    except Exception as exc:
        skip_checks(5, f"initialization raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/5] LinearLayer_LoRA forward initial equivalence")
    try:
        layer = make_lora_layer()
        x = torch.randn(3, 5)
        y = layer(x)
        expected = F.linear(x, layer.weight, layer.bias)
        check("initial forward not None", y is not None)
        if y is not None:
            check("initial forward shape", tuple(y.shape) == (3, 4), f"got {tuple(y.shape)}")
            check("initial forward finite", torch.isfinite(y).all().item())
            check("initial forward equals base linear", torch.allclose(y, expected, atol=1e-6))
        else:
            skip_checks(3, "forward returned None")
    except Exception as exc:
        skip_checks(4, f"initial forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/5] LinearLayer_LoRA low-rank delta and gradients")
    try:
        layer = make_lora_layer()
        with torch.no_grad():
            layer.lora_right_weight.copy_(torch.tensor([
                [0.2, -0.1],
                [0.0, 0.3],
                [-0.4, 0.5],
                [0.1, 0.2],
                [0.3, -0.2],
            ]))
            layer.lora_left_weight.copy_(torch.tensor([
                [0.4, -0.2, 0.1, 0.0],
                [-0.3, 0.2, 0.5, -0.1],
            ]))
        x = torch.randn(3, 5)
        y = layer(x)
        expected = F.linear(x, layer.weight, layer.bias) + (x @ layer.lora_right_weight @ layer.lora_left_weight) * layer.lora_scaling
        check("delta forward not None", y is not None)
        if y is not None:
            check("delta forward matches manual", torch.allclose(y, expected, atol=1e-6))
            check("delta changes base output", not torch.allclose(y, F.linear(x, layer.weight, layer.bias), atol=1e-6))
            loss = y.sum()
            loss.backward()
            check("LoRA gradients exist", layer.lora_left_weight.grad is not None and layer.lora_right_weight.grad is not None)
            check("base weight remains frozen", layer.weight.grad is None)
        else:
            skip_checks(4, "forward returned None")
    except Exception as exc:
        skip_checks(5, f"delta/gradient path raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/5] LoRA fuse and unfuse semantics")
    try:
        layer = make_lora_layer()
        with torch.no_grad():
            layer.lora_right_weight.copy_(torch.randn_like(layer.lora_right_weight) * 0.2)
            layer.lora_left_weight.copy_(torch.randn_like(layer.lora_left_weight) * 0.2)
        x = torch.randn(2, 5)
        original_weight = layer.weight.detach().clone()
        unfused = layer(x)
        layer.fuse_lora_weight()
        fused = layer(x)
        check("fuse flag true", layer.fuse_lora is True)
        check("fused forward matches unfused", torch.allclose(fused, unfused, atol=1e-5))
        check("fuse changes base weight", not torch.allclose(layer.weight, original_weight, atol=1e-6))
        layer.unfuse_lora_weight()
        restored = layer(x)
        check("fuse flag false", layer.fuse_lora is False)
        check("unfuse restores behavior and weight", torch.allclose(restored, unfused, atol=1e-5) and torch.allclose(layer.weight, original_weight, atol=1e-6))
    except Exception as exc:
        skip_checks(5, f"fuse/unfuse raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/5] only_optimize_lora_parameters")
    try:
        model = nn.Sequential(
            nn.Linear(5, 4),
            LinearLayer_LoRA(nn.Parameter(torch.randn(4, 5)), lora_dim=2, lora_scaling=1, lora_droppout=0.0, bias=nn.Parameter(torch.zeros(4))),
        )
        returned = only_optimize_lora_parameters(model)
        params = dict(model.named_parameters())
        lora_flags = {name: p.requires_grad for name, p in params.items() if "lora_" in name}
        non_lora_flags = {name: p.requires_grad for name, p in params.items() if "lora_" not in name}
        check("returns same model", returned is model)
        check("all LoRA params trainable", len(lora_flags) == 2 and all(lora_flags.values()))
        check("all non-LoRA params frozen", len(non_lora_flags) > 0 and not any(non_lora_flags.values()))
        check("parameter partition complete", len(lora_flags) + len(non_lora_flags) == len(params))
    except Exception as exc:
        skip_checks(4, f"only_optimize_lora_parameters raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may not be implemented correctly.")
    print("=" * 70)
    if failed:
        raise SystemExit(1)
