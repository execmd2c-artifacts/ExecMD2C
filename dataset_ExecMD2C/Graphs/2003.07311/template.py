# ============================================================
# ground_truth.py - clDice Core Loss Components (Self-contained)
# Source: Graphs/clDice-master
#
# Contains only the PyTorch soft-skeletonization and clDice loss definitions.
# No training loops, datasets, evaluation scripts, or CLI wrappers.
# ============================================================

# --- Third-party imports ---
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: cldice_loss/pytorch/soft_skeleton.py] ---
class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=40):

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):
        """
        [TODO] Compute differentiable morphological erosion for 2D or 3D probability maps.

        Input:
            img: (batch, channels, height, width) or (batch, channels, depth, height, width) - soft segmentation probabilities.

        Output: same shape as img - eroded probabilities.

"""
        pass

    def soft_dilate(self, img):
        """
        [TODO] Compute differentiable morphological dilation for 2D or 3D probability maps.

        Input:
            img: (batch, channels, height, width) or (batch, channels, depth, height, width) - soft segmentation probabilities.

        Output: same shape as img - dilated probabilities.

"""
        pass

    def soft_open(self, img):
        
        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):
        """
        [TODO] Iteratively extract a differentiable soft skeleton.

        Input:
            img: (batch, channels, height, width) or (batch, channels, depth, height, width) - soft segmentation probabilities.

        Output: same shape as img - nonnegative soft skeleton probabilities.

"""
        pass

    def forward(self, img):

        return self.soft_skel(img)


# --- [Original file: cldice_loss/pytorch/cldice.py] ---
class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1., exclude_background=False):
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        """
        [TODO] Compute the differentiable clDice topology loss.

        Input:
            y_true: (batch, channels, height, width) - target segmentation probabilities or masks.
            y_pred: (batch, channels, height, width) - predicted segmentation probabilities.

        Output: scalar tensor - soft clDice loss.

"""
        pass


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]

    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]

    Returns:
        [float32]: [loss value]
    """
    smooth = 1
    intersection = torch.sum((y_true * y_pred))
    coeff = (2. *  intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1., exclude_background=False):
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.soft_skeletonize = SoftSkeletonize(num_iter=10)
        self.exclude_background = exclude_background

    def forward(self, y_true, y_pred):
        """
        [TODO] Compute the alpha-weighted combination of Dice loss and soft clDice loss.

        Input:
            y_true: (batch, channels, height, width) - target segmentation probabilities or masks.
            y_pred: (batch, channels, height, width) - predicted segmentation probabilities.

        Output: scalar tensor - combined segmentation and topology loss.

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

    print("=" * 70)
    print("clDice Core Loss Components")
    print("Automated Test Suite - 5 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    skeletonizer = SoftSkeletonize(num_iter=2).to(device)

    # ==============================================================
    # Test 1/5: SoftSkeletonize.soft_erode
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] SoftSkeletonize.soft_erode")
    try:
        img2d = torch.rand(2, 1, 6, 6, device=device)
        img3d = torch.rand(1, 1, 4, 5, 6, device=device)
        eroded2d = skeletonizer.soft_erode(img2d)
        eroded3d = skeletonizer.soft_erode(img3d)
        check("soft_erode output not None", eroded2d is not None and eroded3d is not None)
        if eroded2d is not None and eroded3d is not None:
            check("soft_erode 2D shape", tuple(eroded2d.shape) == (2, 1, 6, 6),
                  f"expected (2, 1, 6, 6), got {tuple(eroded2d.shape)}")
            check("soft_erode 3D shape", tuple(eroded3d.shape) == (1, 1, 4, 5, 6),
                  f"expected (1, 1, 4, 5, 6), got {tuple(eroded3d.shape)}")
            check("soft_erode finite", torch.isfinite(eroded2d).all().item() and torch.isfinite(eroded3d).all().item())
            check("soft_erode does not increase activations",
                  bool((eroded2d <= img2d + 1e-6).all().item() and (eroded3d <= img3d + 1e-6).all().item()))
        else:
            skip_checks(4, "soft_erode returned None")
    except Exception as exc:
        print(f"  [soft_erode] ERROR: {exc}")
        skip_checks(5, "soft_erode raised an exception")
    print()

    # ==============================================================
    # Test 2/5: SoftSkeletonize.soft_dilate
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] SoftSkeletonize.soft_dilate")
    try:
        dot2d = torch.zeros(1, 1, 5, 5, device=device)
        dot2d[:, :, 2, 2] = 1.0
        dot3d = torch.zeros(1, 1, 3, 5, 5, device=device)
        dot3d[:, :, 1, 2, 2] = 1.0
        dilated2d = skeletonizer.soft_dilate(dot2d)
        dilated3d = skeletonizer.soft_dilate(dot3d)
        check("soft_dilate output not None", dilated2d is not None and dilated3d is not None)
        if dilated2d is not None and dilated3d is not None:
            check("soft_dilate 2D shape", tuple(dilated2d.shape) == (1, 1, 5, 5),
                  f"expected (1, 1, 5, 5), got {tuple(dilated2d.shape)}")
            check("soft_dilate 3D shape", tuple(dilated3d.shape) == (1, 1, 3, 5, 5),
                  f"expected (1, 1, 3, 5, 5), got {tuple(dilated3d.shape)}")
            check("soft_dilate finite", torch.isfinite(dilated2d).all().item() and torch.isfinite(dilated3d).all().item())
            check("soft_dilate expands isolated activations",
                  bool(dilated2d.sum().item() > dot2d.sum().item() and dilated3d.sum().item() > dot3d.sum().item()))
        else:
            skip_checks(4, "soft_dilate returned None")
    except Exception as exc:
        print(f"  [soft_dilate] ERROR: {exc}")
        skip_checks(5, "soft_dilate raised an exception")
    print()

    # ==============================================================
    # Test 3/5: SoftSkeletonize.soft_skel
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] SoftSkeletonize.soft_skel")
    try:
        vessel = torch.zeros(1, 1, 7, 7, device=device)
        vessel[:, :, 3, 1:6] = 1.0
        vessel.requires_grad_(True)
        empty = torch.zeros(1, 1, 7, 7, device=device)
        skel = skeletonizer.soft_skel(vessel)
        empty_skel = skeletonizer.soft_skel(empty)
        check("soft_skel output not None", skel is not None and empty_skel is not None)
        if skel is not None and empty_skel is not None:
            check("soft_skel shape", tuple(skel.shape) == (1, 1, 7, 7),
                  f"expected (1, 1, 7, 7), got {tuple(skel.shape)}")
            check("soft_skel finite", torch.isfinite(skel).all().item() and torch.isfinite(empty_skel).all().item())
            check("soft_skel nonnegative", bool((skel >= -1e-6).all().item()))
            check("soft_skel empty image remains zero", torch.allclose(empty_skel, torch.zeros_like(empty_skel), atol=1e-6))
            skel.sum().backward()
            check("soft_skel gradient reaches input", vessel.grad is not None and torch.isfinite(vessel.grad).all().item())
        else:
            skip_checks(5, "soft_skel returned None")
    except Exception as exc:
        print(f"  [soft_skel] ERROR: {exc}")
        skip_checks(6, "soft_skel raised an exception")
    print()

    # ==============================================================
    # Test 4/5: soft_cldice.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] soft_cldice.forward")
    try:
        y_true = torch.zeros(1, 2, 7, 7, device=device)
        y_true[:, 0, :, :] = 1.0
        y_true[:, 1, 3, 1:6] = 1.0
        y_pred = y_true.clone()
        y_pred[:, 0, :, :] = 0.0
        y_pred.requires_grad_(True)
        module = soft_cldice(exclude_background=True).to(device)
        loss = module(y_true, y_pred)
        check("soft_cldice output not None", loss is not None)
        if loss is not None:
            check("soft_cldice scalar output", loss.dim() == 0)
            check("soft_cldice output finite", torch.isfinite(loss).item())
            check("soft_cldice perfect foreground near zero", loss.item() < 1e-5,
                  f"expected near zero, got {loss.item():.6f}")
            loss.backward()
            check("soft_cldice gradient reaches prediction",
                  y_pred.grad is not None and torch.isfinite(y_pred.grad).all().item())
        else:
            skip_checks(4, "soft_cldice.forward returned None")
    except Exception as exc:
        print(f"  [soft_cldice.forward] ERROR: {exc}")
        skip_checks(5, "soft_cldice.forward raised an exception")
    print()

    # ==============================================================
    # Test 5/5: soft_dice_cldice.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] soft_dice_cldice.forward")
    try:
        y_true = torch.zeros(1, 1, 7, 7, device=device)
        y_true[:, :, 3, 1:6] = 1.0
        y_pred = torch.full((1, 1, 7, 7), 0.1, device=device)
        y_pred[:, :, 3, 1:6] = 0.9
        y_pred.requires_grad_(True)
        module = soft_dice_cldice(alpha=0.5).to(device)
        combined = module(y_true, y_pred)
        dice_only_module = soft_dice_cldice(alpha=0.0).to(device)
        dice_only = dice_only_module(y_true, y_pred)
        expected_dice = soft_dice(y_true, y_pred)
        check("soft_dice_cldice output not None", combined is not None)
        if combined is not None:
            check("soft_dice_cldice scalar output", combined.dim() == 0)
            check("soft_dice_cldice output finite", torch.isfinite(combined).item())
            check("soft_dice_cldice alpha zero recovers dice",
                  torch.allclose(dice_only, expected_dice, atol=1e-6),
                  f"expected {expected_dice.item():.6f}, got {dice_only.item():.6f}")
            combined.backward()
            check("soft_dice_cldice gradient reaches prediction",
                  y_pred.grad is not None and torch.isfinite(y_pred.grad).all().item())
        else:
            skip_checks(4, "soft_dice_cldice.forward returned None")
    except Exception as exc:
        print(f"  [soft_dice_cldice.forward] ERROR: {exc}")
        skip_checks(5, "soft_dice_cldice.forward raised an exception")
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
