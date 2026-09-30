# ============================================================
# ground_truth.py - MSGazeNet Core Model Components
# Source:
#   Miscellaneous/MSGazeNet-main/models/unet_parts.py
#   Miscellaneous/MSGazeNet-main/models/aeri_unet.py
#   Miscellaneous/MSGazeNet-main/models/msgazenet.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

# -*- coding: utf-8 -*-

import math
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: models/unet_parts.py] ---
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# --- [Original file: models/aeri_unet.py] ---
class AERI_UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, bilinear=True):
        super(AERI_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 2)
        self.outlayer = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = self.outlayer(logits)
        return logits


# --- [Original file: models/msgazenet.py] ---
momentum = 0.001

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001, eps=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001, eps=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.drop_rate = drop_rate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=True) or None
        self.activate_before_residual = activate_before_residual

    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
        
class MSGazeNet(nn.Module):
    def __init__(self, first_stride, num_classes, depth=28, widen_factor=2, drop_rate=0.0, is_remix=False):
        super(MSGazeNet, self).__init__()
        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1_eye = nn.Conv2d(1, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv1_iris = nn.Conv2d(1, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.conv1_eyemask = nn.Conv2d(1, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=True)
        # 1st block
        self.block1_eye = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block1_iris = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        self.block1_eyemask = NetworkBlock(
            n, channels[0], channels[1], block, first_stride, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2_eye = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        self.block2_iris = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        self.block2_eyemask = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2]*3, channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001, eps=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=False)
        self.fc = nn.Sequential(
                  nn.Linear(channels[3], channels[3]*2),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.25),
                  nn.Linear(channels[3]*2, channels[3]),
                  nn.ReLU(inplace=True),
                  nn.Dropout(0.25),
                  nn.Linear(channels[3], num_classes)
                  )
        self.channels = channels[3]

        # rot_classifier for Remix Match
        self.is_remix = is_remix
        if is_remix:
            self.rot_classifier = nn.Linear(self.channels, 4)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()
   
    def forward(self, x, ood_test=False):
        out_x = self.conv1_eye(x['eye'])
        out_y = self.conv1_iris(x['iris'])
        out_z = self.conv1_eyemask(x['eye_mask'])
        out_x = self.block1_eye(out_x)
        out_y = self.block1_iris(out_y)
        out_z = self.block1_eyemask(out_z)
        out_x = self.block2_eye(out_x)
        out_y = self.block2_iris(out_y)
        out_z = self.block2_eyemask(out_z)
        out = torch.cat((out_x, out_y, out_z),1)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(-1, self.channels)
        output = self.fc(out)

        return output
                
                
class build_msgazenet:
    def __init__(self, first_stride=1, depth=28, widen_factor=2, bn_momentum=0.01, leaky_slope=0.0, dropRate=0.0,
                 use_embed=False, is_remix=False):
        self.first_stride = first_stride
        self.depth = depth
        self.widen_factor = widen_factor
        self.bn_momentum = bn_momentum
        self.dropRate = dropRate
        self.leaky_slope = leaky_slope
        self.use_embed = use_embed
        self.is_remix = is_remix

    def build(self, num_classes):
        return MSGazeNet(
            first_stride=self.first_stride,
            depth=self.depth,
            num_classes=num_classes,
            widen_factor=self.widen_factor,
            drop_rate=self.dropRate,
            is_remix=self.is_remix,
        )


# ============================================================
# __main__: Automated test suite for 2 ablated functions
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
    print("MSGazeNet: Multistream Gaze Estimation with AERI")
    print("Automated reproduction benchmark - 2 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/2: AERI_UNet.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/2] AERI_UNet.forward - anatomical eye region segmentation path")
    try:
        aeri = AERI_UNet(n_channels=1, n_classes=2, bilinear=True).to(device)
        aeri.eval()
        eye = torch.randn(2, 1, 64, 96, device=device, requires_grad=True)
        out = aeri(eye)
        check("AERI output not None", out is not None)
        if out is not None:
            check("AERI output shape", tuple(out.shape) == (2, 2, 64, 96), f"expected (2, 2, 64, 96), got {tuple(out.shape)}")
            check("AERI output finite", torch.isfinite(out).all().item())
            check("AERI sigmoid range", (out.min().item() >= 0.0) and (out.max().item() <= 1.0))
            out.sum().backward()
            first_grad = aeri.inc.double_conv[0].weight.grad
            last_grad = aeri.outc.conv.weight.grad
            grads_ok = first_grad is not None and first_grad.abs().sum().item() > 0 and last_grad is not None and last_grad.abs().sum().item() > 0
            check("AERI gradients reach encoder and segmentation head", grads_ok)
        else:
            skip_checks(4, "AERI_UNet.forward returned None")
    except Exception as e:
        skip_checks(5, f"AERI_UNet.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/2: MSGazeNet.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/2] MSGazeNet.forward - three-stream gaze regression fusion")
    try:
        model = MSGazeNet(first_stride=1, num_classes=2, depth=10, widen_factor=1, drop_rate=0.0).to(device)
        model.eval()
        block3_inputs = []
        def capture_block3_input(module, inputs):
            block3_inputs.append(inputs[0].detach())
        hook = model.block3.register_forward_pre_hook(capture_block3_input)
        data = {
            "eye": torch.randn(2, 1, 32, 48, device=device, requires_grad=True),
            "iris": torch.randn(2, 1, 32, 48, device=device, requires_grad=True),
            "eye_mask": torch.randn(2, 1, 32, 48, device=device, requires_grad=True),
        }
        output = model(data)
        hook.remove()
        check("MSGazeNet output not None", output is not None)
        if output is not None:
            check("MSGazeNet output shape", tuple(output.shape) == (2, 2), f"expected (2, 2), got {tuple(output.shape)}")
            check("MSGazeNet output finite", torch.isfinite(output).all().item())
            fusion_shape_ok = len(block3_inputs) == 1 and tuple(block3_inputs[0].shape) == (2, 96, 16, 24)
            check("MSGazeNet fuses three 32-channel streams before block3", fusion_shape_ok)
            output.sum().backward()
            stream_grads_ok = (
                model.conv1_eye.weight.grad is not None and model.conv1_eye.weight.grad.abs().sum().item() > 0 and
                model.conv1_iris.weight.grad is not None and model.conv1_iris.weight.grad.abs().sum().item() > 0 and
                model.conv1_eyemask.weight.grad is not None and model.conv1_eyemask.weight.grad.abs().sum().item() > 0
            )
            check("MSGazeNet gradients reach all three input streams", stream_grads_ok)
        else:
            skip_checks(4, "MSGazeNet.forward returned None")
    except Exception as e:
        skip_checks(5, f"MSGazeNet.forward raised {type(e).__name__}: {e}")
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
