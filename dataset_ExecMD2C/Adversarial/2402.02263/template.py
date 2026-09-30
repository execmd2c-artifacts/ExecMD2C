# ============================================================
# ground_truth.py - MixedNUTS Core Model Components
# Source: Adversarial/MixedNUTS-main
#
# Contains ONLY the nonlinearly mixed classifier, output maps,
# and their direct tensor helper dependency.
# ============================================================

from typing import Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- [Original file: utils/misc_utils.py] ---

def outer_prod(tensor1, tensor2):
    """
    [TODO] Compute an outer product that preserves every axis of both input tensors.

    Input:
        tensor1: shape (d1_0, ..., d1_m) - mixing weights or any leading tensor.
        tensor2: shape (d2_0, ..., d2_n) - logits/probabilities or any trailing tensor.

    Output: shape (d1_0, ..., d1_m, d2_0, ..., d2_n) - elementwise products over all axis pairs.

"""
    pass


# --- [Original file: models/output_maps.py] ---

def normalize_topk(logits, k=None, center_only=False):
    """
    [TODO] Normalize logits using either all classes or the top-k class scores per sample.

    Input:
        logits: (batch, num_classes) - raw classifier logits.
        k: int or None - number of largest logits to use for statistics; None uses all classes.
        center_only: bool - when true, subtract the selected mean without variance scaling.

    Output: (batch, num_classes) - centered or standardized logits.

"""
    pass


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, return_probs=False):
        return logits.softmax(dim=1) if return_probs else logits


class HardMaxMap(nn.Module):
    def __init__(self, ln_k=250):
        super().__init__()
        self.approx = LNPowerScaleMap(scale=1e6, power=1, ln_k=ln_k, center_only=False)

    def forward_exact(self, logits, return_probs=False):
        probs = F.one_hot(logits.argmax(dim=1), num_classes=logits.shape[1]).float()
        mapped_output = probs if return_probs else (probs - 1e-12) * np.inf
        return mapped_output.cpu() if 'mps' in str(logits.device) else mapped_output

    def forward(self, logits, return_probs=False):
        if logits.grad_fn is not None:
            return self.approx(logits, return_probs)
        else:
            return self.forward_exact(logits, return_probs)


class ScaleMap(nn.Module):
    def __init__(self, scale=1):
        super().__init__()
        self.scale = nn.parameter.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, logits, return_probs=False):
        scaled_logits = logits * self.scale
        mapped_output = scaled_logits.softmax(dim=1) if return_probs else scaled_logits
        return mapped_output.cpu() if 'mps' in str(logits.device) else mapped_output


class LayerNormMap(nn.Module):
    def __init__(self, ln_k=250, center_only=False):
        super().__init__()
        self.ln_k, self.center_only = ln_k, center_only

    def forward(self, logits, return_probs=False):
        orig_device = logits.device
        if 'cuda' not in str(orig_device):
            logits = logits.cpu()

        normed_logits = normalize_topk(logits.double(), k=self.ln_k, center_only=self.center_only)
        mapped_output = normed_logits.softmax(dim=1) if return_probs else normed_logits
        if 'mps' not in str(orig_device):
            return mapped_output.to(orig_device)
        else:
            return mapped_output


class LNPowerScaleMap(nn.Module):
    def __init__(self, scale=1, power=1, ln_k=250, center_only=False):
        super().__init__()
        self.ln_k, self.center_only = ln_k, center_only
        self.scale = nn.parameter.Parameter(torch.tensor(scale), requires_grad=False)
        self.power = nn.parameter.Parameter(torch.tensor(power), requires_grad=False)

    def forward(self, logits, return_probs=False):
        """
        [TODO] Apply MixedNUTS layer-normalization, signed power, and scale mapping to logits.

        Input:
            logits: (batch, num_classes) - raw logits from a base classifier.
            return_probs: bool - when true, return class probabilities instead of transformed logits.

        Output: (batch, num_classes) - transformed logits or row-normalized probabilities.

"""
        pass


class LNClampPowerScaleMap(nn.Module):
    def __init__(
        self, scale=.4, power=1, clamp_bias=-1, clamp_fn=nn.GELU(), ln_k=250, center_only=False
    ):
        super().__init__()
        self.ln_k, self.center_only = ln_k, center_only
        self.scale = nn.parameter.Parameter(torch.tensor(scale), requires_grad=False)
        self.power = nn.parameter.Parameter(torch.tensor(power), requires_grad=False)
        self.clamp_bias = nn.parameter.Parameter(torch.tensor(clamp_bias), requires_grad=False)
        self.clamp_fn = clamp_fn

    def forward(self, logits, return_probs=False):
        """
        [TODO] Apply the clamped nonlinear logit transformation used for MixedNUTS robust logits.

        Input:
            logits: (batch, num_classes) - raw logits from a base classifier.
            return_probs: bool - when true, return class probabilities instead of transformed logits.

        Output: (batch, num_classes) - transformed logits or row-normalized probabilities.

"""
        pass


class MappedModel(nn.Module):
    def __init__(self, model, map):
        super().__init__()
        self.model = model
        self.map = map

    def forward(self, image, return_probs=False):
        output = self.map(self.model(image), return_probs)
        return output.float().to(image.device) if 'mps' in str(image.device) else output


# --- [Original file: models/nonlin_mixed_classifier.py] ---

class NonLinMixedClassifier(nn.Module):

    def __init__(self, std_model: nn.Module, rob_model: nn.Module, forward_settings: dict):
        """ Initialize the MixedNUTS classifier.

        Args:
            std_model (nn.Module): The standard base classifier (can be non-robust).
            rob_model (nn.Module): The robust base classifier.
            forward_settings (dict): A dictionary containing the following forward settings:
                - use_nonlin_for_grad (bool):
                    If True, use the robust base model logit nonlinearity for gradient.
                    If False, (partially) bypass the nonlinearity for
                    better gradient flow and more effective attack.
                - std_map (nn.Module):
                    The mapping function for the standard base model logits.
                - rob_map (nn.Module):
                    The mapping function for the robust base model logits.
                - alpha (float or Tensor):
                    The mixing weight between the two base classifiers.
                    If a float, the MixedNUTS output has shape (n, num_classes).
                    If a Tensor with shape (m1, m2, ..., md), the MixedNUTS output has shape
                    (n, num_classes, m1, ..., md). I.e., an outer product is performed.
                - alpha_diffable (float or Tensor):
                    The mixing weight between the two base classifiers for differentiable output.
                    The shape-dependent behavior is the same as alpha.
        """
        super().__init__()
        self.std_model, self.rob_model = std_model, rob_model

        # Freeze models and set to eval mode
        for model, name in zip([self.std_model, self.rob_model], ["STD", "ROB"]):
            model.eval()
            for param in model.parameters():
                assert param.requires_grad == False
            print(f"The {name} classifier has "
                  f"{sum(p.numel() for p in model.parameters())} parameters. "
                  f"{sum(p.numel() for p in model.parameters() if p.requires_grad)} "
                  "parameters are trainable.")

        # alpha value (mixing balance)
        self.set_alpha_value(forward_settings["alpha"], forward_settings["alpha_diffable"])
        self.use_nonlin_for_grad = forward_settings.get("use_nonlin_for_grad", False)
        print(f"{'Using' if self.use_nonlin_for_grad else 'Bypassing'} "
              "robust base model nonlinear transformation for gradient calculations.")

        # Base model logit transformations
        self.std_map, self.rob_map = forward_settings['std_map'], forward_settings['rob_map']

        # Enable autocast if specified
        self.enable_autocast = forward_settings.get("enable_autocast", False)
        print(f"{'Enabling' if self.enable_autocast else 'Disabling'} autocast.")

    def set_alpha_value(
        self, alpha: Union[float, int, Tensor, List],
        alpha_diffable: Union[float, int, Tensor, List], verbose=True
    ):
        """ Set the alpha value for the mixed classifier. """
        # Set alpha
        self._alpha = nn.parameter.Parameter(torch.tensor(alpha).float(), requires_grad=False)
        assert self._alpha.min() >= 0 and self._alpha.max() <= 1, \
            "The range of alpha should be [0, 1]."
        if verbose:
            print(f"Using alpha={alpha}.")
        if torch.numel(self._alpha) == 1 and self._alpha.item() == 0:
            print("Using the STD network only.")
        elif torch.numel(self._alpha) == 1 and self._alpha.item() == 1:
            print("Using the ROB network only.")

        # Set alpha_diffable
        self._alpha_diffable = nn.parameter.Parameter(
            torch.tensor(alpha_diffable).float(), requires_grad=False
        )
        assert self._alpha_diffable.min() >= 0 and self._alpha_diffable.max() <= 1, \
            "The range of alpha_diffable should be [0, 1]."
        if verbose and not torch.allclose(self._alpha_diffable, self._alpha, rtol=1e-8):
            print(f"Using alpha_diffable={alpha_diffable}.")
        if torch.numel(self._alpha_diffable) == 1 and self._alpha_diffable.item() == 0:
            print("Using the STD network only for differentiable output.")
        elif torch.numel(self._alpha_diffable) == 1 and self._alpha_diffable.item() == 1:
            print("Using the ROB network only for differentiable output.")

        assert self._alpha.shape == self._alpha_diffable.shape, \
            "alpha and alpha_diffable must have the same shape."

    def forward(self, images: Tensor, return_probs: bool = False, return_all: bool = False):
        """
        [TODO] Run the MixedNUTS classifier with nonlinear base-logit maps and alpha mixing.

        Input:
            images: (batch, channels, height, width) - input image batch for both frozen base classifiers.
            return_probs: bool - when true, return mixed probabilities; otherwise return their log values.
            return_all: bool - when true, also return the differentiable attack-oriented branch and flattened alpha values.

        Output:
            If alpha is scalar and return_all is false: (batch, num_classes).
            If alpha has shape (m1, ..., md) and return_all is false: (m1, ..., md, batch, num_classes).
            If return_all is true: a tuple of two mixed tensors with the above shape and alpha reshaped to one dimension.

"""
        pass


# ============================================================
# __main__: Automated test suite for 5 ablated functions
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

    class FrozenLinearModel(nn.Module):
        def __init__(self, in_features=12, num_classes=4, offset=0.0):
            super().__init__()
            self.linear = nn.Linear(in_features, num_classes)
            with torch.no_grad():
                weight = torch.arange(num_classes * in_features, dtype=torch.float32).view(num_classes, in_features)
                self.linear.weight.copy_((weight % 7 - 3.0) / 10.0)
                self.linear.bias.copy_(torch.linspace(-0.2, 0.2, num_classes) + offset)
            for param in self.parameters():
                param.requires_grad = False

        def forward(self, images):
            return self.linear(images.flatten(1))

    def make_mixed(alpha=0.35, alpha_diffable=0.2, use_nonlin_for_grad=False):
        settings = {
            "alpha": alpha,
            "alpha_diffable": alpha_diffable,
            "std_map": IdentityMap(),
            "rob_map": LNClampPowerScaleMap(scale=.7, power=1.2, clamp_bias=-.25, ln_k=3),
            "use_nonlin_for_grad": use_nonlin_for_grad,
            "enable_autocast": False,
        }
        model = NonLinMixedClassifier(
            FrozenLinearModel(offset=0.0),
            FrozenLinearModel(offset=0.4),
            settings,
        )
        model.eval()
        return model

    print("=" * 70)
    print("MixedNUTS Benchmark: Nonlinearly Mixed Classifier")
    print("Automated Test Suite - 5 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/5: outer_prod
    # ==========================================================
    print("-" * 60)
    print("[Test 1/5] outer_prod - alpha grid and probability tensor mixing")
    try:
        a = torch.tensor([0.2, 0.8], device=device)
        b = torch.arange(12, dtype=torch.float32, device=device).view(3, 4)
        y = outer_prod(a, b)
        check("outer_prod output not None", y is not None)
        if y is not None:
            check("outer_prod output shape", y.shape == (2, 3, 4), f"expected (2, 3, 4), got {tuple(y.shape)}")
            check("outer_prod output finite", torch.isfinite(y).all().item())
            check("outer_prod first slice", torch.allclose(y[0], a[0] * b))
            check("outer_prod second slice", torch.allclose(y[1], a[1] * b))
        else:
            skip_checks(4, "outer_prod returned None")
    except Exception as exc:
        skip_checks(5, f"outer_prod raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/5: normalize_topk
    # ==========================================================
    print("-" * 60)
    print("[Test 2/5] normalize_topk - centered and top-k normalized logits")
    try:
        logits = torch.tensor([[1.0, 2.0, 5.0, -1.0], [4.0, 3.0, 2.0, 1.0]], device=device)
        centered = normalize_topk(logits, k=2, center_only=True)
        normed = normalize_topk(logits, k=2, center_only=False)
        check("normalize_topk centered output not None", centered is not None)
        check("normalize_topk normalized output not None", normed is not None)
        if centered is not None and normed is not None:
            check("normalize_topk centered shape", centered.shape == (2, 4), f"expected (2, 4), got {tuple(centered.shape)}")
            check("normalize_topk normalized shape", normed.shape == (2, 4), f"expected (2, 4), got {tuple(normed.shape)}")
            check("normalize_topk finite", torch.isfinite(centered).all().item() and torch.isfinite(normed).all().item())
            top2_centered = centered.topk(k=2, dim=1).values.mean(dim=1)
            check("normalize_topk top-k centered", torch.allclose(top2_centered, torch.zeros_like(top2_centered), atol=1e-6))
            top2_normed = normed.topk(k=2, dim=1).values
            check("normalize_topk top-k unit variance", torch.allclose(top2_normed.var(dim=1), torch.ones(2, device=device), atol=1e-5))
        else:
            skip_checks(5, "normalize_topk returned None")
    except Exception as exc:
        skip_checks(7, f"normalize_topk raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/5: LNPowerScaleMap.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 3/5] LNPowerScaleMap.forward - logit normalization, power, and scaling")
    try:
        logits = torch.randn(3, 5, device=device, requires_grad=True)
        mapper = LNPowerScaleMap(scale=1.5, power=1.2, ln_k=3).to(device)
        mapped_logits = mapper(logits, return_probs=False)
        mapped_probs = mapper(logits, return_probs=True)
        check("LNPowerScaleMap logits not None", mapped_logits is not None)
        check("LNPowerScaleMap probs not None", mapped_probs is not None)
        if mapped_logits is not None and mapped_probs is not None:
            check("LNPowerScaleMap logits shape", mapped_logits.shape == (3, 5), f"expected (3, 5), got {tuple(mapped_logits.shape)}")
            check("LNPowerScaleMap probs shape", mapped_probs.shape == (3, 5), f"expected (3, 5), got {tuple(mapped_probs.shape)}")
            check("LNPowerScaleMap outputs finite", torch.isfinite(mapped_logits).all().item() and torch.isfinite(mapped_probs).all().item())
            check("LNPowerScaleMap probability rows sum to one", torch.allclose(mapped_probs.sum(dim=1), torch.ones_like(mapped_probs.sum(dim=1)), atol=1e-5))
            mapped_logits.sum().backward()
            check("LNPowerScaleMap input gradient", logits.grad is not None and torch.isfinite(logits.grad).all().item())
        else:
            skip_checks(5, "LNPowerScaleMap.forward returned None")
    except Exception as exc:
        skip_checks(7, f"LNPowerScaleMap.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/5: LNClampPowerScaleMap.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 4/5] LNClampPowerScaleMap.forward - clamped nonlinear logit transformation")
    try:
        logits = torch.randn(3, 5, device=device, requires_grad=True)
        mapper = LNClampPowerScaleMap(scale=.6, power=1.3, clamp_bias=-.2, clamp_fn=nn.GELU(), ln_k=3).to(device)
        mapped_logits = mapper(logits, return_probs=False)
        mapped_probs = mapper(logits, return_probs=True)
        check("LNClampPowerScaleMap logits not None", mapped_logits is not None)
        check("LNClampPowerScaleMap probs not None", mapped_probs is not None)
        if mapped_logits is not None and mapped_probs is not None:
            check("LNClampPowerScaleMap logits shape", mapped_logits.shape == (3, 5), f"expected (3, 5), got {tuple(mapped_logits.shape)}")
            check("LNClampPowerScaleMap probs shape", mapped_probs.shape == (3, 5), f"expected (3, 5), got {tuple(mapped_probs.shape)}")
            check("LNClampPowerScaleMap outputs finite", torch.isfinite(mapped_logits).all().item() and torch.isfinite(mapped_probs).all().item())
            check("LNClampPowerScaleMap probability rows sum to one", torch.allclose(mapped_probs.sum(dim=1), torch.ones_like(mapped_probs.sum(dim=1)), atol=1e-5))
            mapped_logits.sum().backward()
            check("LNClampPowerScaleMap input gradient", logits.grad is not None and torch.isfinite(logits.grad).all().item())
        else:
            skip_checks(5, "LNClampPowerScaleMap.forward returned None")
    except Exception as exc:
        skip_checks(7, f"LNClampPowerScaleMap.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/5: NonLinMixedClassifier.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 5/5] NonLinMixedClassifier.forward - probability mixing and differentiable branch")
    try:
        images = torch.randn(2, 3, 2, 2, device=device, requires_grad=True)
        model = make_mixed(alpha=torch.tensor([0.25, 0.75]), alpha_diffable=torch.tensor([0.1, 0.6]), use_nonlin_for_grad=False)
        logits = model(images, return_probs=False, return_all=False)
        probs, probs_diffable, alpha_values = model(images, return_probs=True, return_all=True)
        logits_all, logits_diffable, alpha_values_logits = model(images, return_probs=False, return_all=True)
        check("NonLinMixedClassifier logits not None", logits is not None)
        check("NonLinMixedClassifier return_all probs not None", probs is not None and probs_diffable is not None and alpha_values is not None)
        check("NonLinMixedClassifier return_all logits not None", logits_all is not None and logits_diffable is not None and alpha_values_logits is not None)
        if logits is not None and probs is not None and probs_diffable is not None and alpha_values is not None and logits_all is not None and logits_diffable is not None:
            check("NonLinMixedClassifier grid logits shape", logits.shape == (2, 2, 4), f"expected (2, 2, 4), got {tuple(logits.shape)}")
            check("NonLinMixedClassifier grid probs shape", probs.shape == (2, 2, 4), f"expected (2, 2, 4), got {tuple(probs.shape)}")
            check("NonLinMixedClassifier diffable shape", probs_diffable.shape == (2, 2, 4), f"expected (2, 2, 4), got {tuple(probs_diffable.shape)}")
            check("NonLinMixedClassifier alpha flatten shape", alpha_values.shape == (2,), f"expected (2,), got {tuple(alpha_values.shape)}")
            check("NonLinMixedClassifier finite outputs", torch.isfinite(logits).all().item() and torch.isfinite(probs).all().item() and torch.isfinite(probs_diffable).all().item())
            check("NonLinMixedClassifier probability rows sum to one", torch.allclose(probs.sum(dim=2), torch.ones(2, 2, device=device), atol=1e-5))
            check("NonLinMixedClassifier logits/probs consistency", torch.allclose(logits.exp(), probs, atol=1e-5))
            check("NonLinMixedClassifier return_all logits consistency", torch.allclose(logits_all, logits, atol=1e-6))
            check("NonLinMixedClassifier alpha values", torch.allclose(alpha_values, torch.tensor([0.25, 0.75], device=device)))
            loss = probs_diffable[..., 0].sum()
            loss.backward()
            check("NonLinMixedClassifier image gradient", images.grad is not None and torch.isfinite(images.grad).all().item() and images.grad.abs().sum().item() > 0)
        else:
            skip_checks(10, "NonLinMixedClassifier.forward returned None for one or more outputs")
    except Exception as exc:
        skip_checks(13, f"NonLinMixedClassifier.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
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
