# ============================================================
# ground_truth.py - Localizable-Rotation Core Components
# Source: Adversarial/Localizable-Rotation-main
#
# Contains model architecture code and the LoRot-I local patch
# sampler needed for the reproduction benchmark.
# ============================================================

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter


# --- [Original file: Imbalanced/models/resnet_cifar.py] ---

__all__ = ['ResNet_s', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class NormedLinear(nn.Module):

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        """
        [TODO] Compute the normalized linear classifier scores used by the long-tailed CIFAR model.

        Input:
            x: (batch, in_features) - dense feature vectors from the penultimate network layer.

        Output: (batch, out_features) - class or rotation logits produced from normalized features and weights.

"""
        pass


class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet_s(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10, use_norm=False, num_trans=16):
        super(ResNet_s, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        if use_norm:
            self.linear = NormedLinear(64, num_classes)
        else:
            self.linear = nn.Linear(64, num_classes)
        if use_norm:
            self.linear2 = NormedLinear(64, num_trans)
        else:
            self.linear2 = nn.Linear(64, num_trans)
        print('num_trans : {}'.format(num_trans))
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x, rot=False, both=False):
        """
        [TODO] Run the CIFAR ResNet backbone and dispatch features to supervised and LoRot heads.

        Input:
            x: (batch, 3, height, width) - CIFAR-style image tensor, typically (batch, 3, 32, 32).
            rot: bool - when true, return only the localizable-rotation prediction head.
            both: bool - when true, return both the supervised classifier head and rotation head.

        Output:
            If both is true: ((batch, num_classes), (batch, num_trans)).
            If rot is true: (batch, num_trans).
            Otherwise: (batch, num_classes).

"""
        pass


def resnet20():
    return ResNet_s(BasicBlock, [3, 3, 3])


def resnet32(num_classes=10, use_norm=False, num_trans=16):
    return ResNet_s(BasicBlock, [5, 5, 5], num_classes=num_classes, use_norm=use_norm, num_trans=num_trans)


def resnet44():
    return ResNet_s(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet_s(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet_s(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet_s(BasicBlock, [200, 200, 200])


# --- [Original file: Imbalanced/cifar_train_lorot-I.py] ---

def rand_bbox(size, lam):
    """
    [TODO] Sample the square local patch used by the LoRot-I pretext task.

    Input:
        size: tuple-like (batch, channels, width, height) - tensor shape of an image batch.
        lam: unused compatibility argument from the original training script.

    Output: (bbx1, bby1, bbx2, bby2) - integer crop coordinates for one square patch.

"""
    pass


# ============================================================
# __main__: Automated test suite for 3 ablated functions
# ============================================================

if __name__ == "__main__":
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
    print("Localizable-Rotation Benchmark: Core Model Components")
    print("Automated Test Suite - 3 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/4: NormedLinear.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/4] NormedLinear.forward - normalized linear classifier")
    try:
        layer = NormedLinear(6, 5).to(device)
        x = torch.randn(4, 6, device=device, requires_grad=True)
        y = layer(x)
        check("NormedLinear output not None", y is not None)
        if y is not None:
            check("NormedLinear output shape", y.shape == (4, 5), f"expected (4, 5), got {tuple(y.shape)}")
            check("NormedLinear output finite", torch.isfinite(y).all().item())
            y_scaled = layer(x * 3.0)
            check("NormedLinear input scale invariant", torch.allclose(y, y_scaled, atol=1e-5))
            y.sum().backward()
            check("NormedLinear weight gradient", layer.weight.grad is not None and torch.isfinite(layer.weight.grad).all().item())
        else:
            skip_checks(4, "NormedLinear.forward returned None")
    except Exception as exc:
        skip_checks(5, f"NormedLinear.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/4: rand_bbox
    # ==========================================================
    print("-" * 60)
    print("[Test 2/4] rand_bbox - LoRot-I square patch sampler")
    try:
        bbox = rand_bbox((8, 3, 32, 32), 0)
        check("rand_bbox returns four coordinates", isinstance(bbox, tuple) and len(bbox) == 4)
        if isinstance(bbox, tuple) and len(bbox) == 4:
            bbx1, bby1, bbx2, bby2 = bbox
            width = bbx2 - bbx1
            height = bby2 - bby1
            check("rand_bbox ordered bounds", bbx1 < bbx2 and bby1 < bby2)
            check("rand_bbox square patch", width == height)
            check("rand_bbox side length range", 2 <= width <= 16, f"got side length {width}")
            check("rand_bbox inside image", 0 <= bbx1 and 0 <= bby1 and bbx2 <= 32 and bby2 <= 32)
        else:
            skip_checks(4, "rand_bbox did not return a coordinate tuple")
    except Exception as exc:
        skip_checks(5, f"rand_bbox raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/4: ResNet_s.forward branch outputs
    # ==========================================================
    print("-" * 60)
    print("[Test 3/4] ResNet_s.forward - classifier, rotation, and dual-head branches")
    try:
        model = resnet32(num_classes=10, use_norm=False, num_trans=16).to(device)
        model.eval()
        x = torch.randn(2, 3, 32, 32, device=device)
        with torch.no_grad():
            cls = model(x)
            rot = model(x, rot=True)
            both = model(x, both=True)
        check("ResNet_s classifier output not None", cls is not None)
        check("ResNet_s rotation output not None", rot is not None)
        check("ResNet_s both output not None", both is not None)
        if cls is not None and rot is not None and both is not None:
            check("ResNet_s classifier shape", cls.shape == (2, 10), f"expected (2, 10), got {tuple(cls.shape)}")
            check("ResNet_s rotation shape", rot.shape == (2, 16), f"expected (2, 16), got {tuple(rot.shape)}")
            check("ResNet_s both tuple length", isinstance(both, tuple) and len(both) == 2)
            if isinstance(both, tuple) and len(both) == 2:
                check("ResNet_s both classifier shape", both[0].shape == (2, 10), f"expected (2, 10), got {tuple(both[0].shape)}")
                check("ResNet_s both rotation shape", both[1].shape == (2, 16), f"expected (2, 16), got {tuple(both[1].shape)}")
                check("ResNet_s outputs finite", torch.isfinite(cls).all().item() and torch.isfinite(rot).all().item() and torch.isfinite(both[0]).all().item() and torch.isfinite(both[1]).all().item())
                check("ResNet_s branch consistency", torch.allclose(cls, both[0], atol=1e-5) and torch.allclose(rot, both[1], atol=1e-5))
            else:
                skip_checks(4, "ResNet_s both branch did not return two tensors")
        else:
            skip_checks(7, "ResNet_s.forward returned None for one or more branches")
    except Exception as exc:
        skip_checks(10, f"ResNet_s.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/4: ResNet_s.forward gradient propagation
    # ==========================================================
    print("-" * 60)
    print("[Test 4/4] ResNet_s.forward - shared feature gradient reaches both heads")
    try:
        model = resnet32(num_classes=10, use_norm=False, num_trans=16).to(device)
        model.train()
        x = torch.randn(2, 3, 32, 32, device=device)
        cls, rot = model(x, both=True)
        check("ResNet_s gradient classifier output not None", cls is not None)
        check("ResNet_s gradient rotation output not None", rot is not None)
        if cls is not None and rot is not None:
            loss = cls.sum() + rot.sum()
            loss.backward()
            check("ResNet_s shared conv gradient", model.conv1.weight.grad is not None and torch.isfinite(model.conv1.weight.grad).all().item())
            check("ResNet_s classifier head gradient", model.linear.weight.grad is not None and torch.isfinite(model.linear.weight.grad).all().item())
            check("ResNet_s rotation head gradient", model.linear2.weight.grad is not None and torch.isfinite(model.linear2.weight.grad).all().item())
        else:
            skip_checks(3, "ResNet_s.forward did not return both branch tensors")
    except Exception as exc:
        skip_checks(5, f"ResNet_s.forward gradient test raised {type(exc).__name__}: {exc}")
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
