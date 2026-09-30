# ============================================================
# ground_truth.py - UCAN Core Model Components (Self-contained)
# Source: Miscellaneous/UCAN-main/model.py
#
# Contains ONLY the model architecture definitions.
# No training, evaluation, dataset, checkpoint, or pipeline code.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


''' 
Generators 
'''


class Generator_DUSENET(nn.Module):
    """Generator network."""
    def __init__(self, input_dim=1, conv_dim=64, c_dim=3, repeat_num=5):
        super(Generator_DUSENET, self).__init__()
        block_init = []
        block_init.append(nn.Conv3d(input_dim + c_dim, conv_dim, kernel_size=7, stride=1, padding=3, bias=False))
        block_init.append(nn.InstanceNorm3d(conv_dim, affine=True, track_running_stats=True))
        block_init.append(nn.ReLU(inplace=True))
        block_init.append(ChannelSpatialSELayer3D(num_channels=conv_dim))
        self.block_init = nn.Sequential(*block_init)
        curr_dim = conv_dim
        block_init_outdim = curr_dim

        # Down-sampling layers.
        block_down1 = []
        block_down1.append(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_down1.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        block_down1.append(nn.ReLU(inplace=True))
        block_down1.append(ChannelSpatialSELayer3D(num_channels=curr_dim * 2))
        self.block_down1 = nn.Sequential(*block_down1)
        curr_dim = curr_dim * 2
        block_down1_outdim = curr_dim

        block_down2 = []
        block_down2.append(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_down2.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        block_down2.append(nn.ReLU(inplace=True))
        block_down2.append(ChannelSpatialSELayer3D(num_channels=curr_dim * 2))
        self.block_down2 = nn.Sequential(*block_down2)
        curr_dim = curr_dim * 2
        block_down2_outdim = curr_dim

        block_down3 = []
        block_down3.append(nn.Conv3d(curr_dim, curr_dim * 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_down3.append(nn.InstanceNorm3d(curr_dim * 2, affine=True, track_running_stats=True))
        block_down3.append(nn.ReLU(inplace=True))
        block_down3.append(ChannelSpatialSELayer3D(num_channels=curr_dim * 2))
        self.block_down3 = nn.Sequential(*block_down3)
        curr_dim = curr_dim * 2
        block_down3_outdim = curr_dim

        # Bottleneck layers.
        block_bottle = []
        for i in range(repeat_num):
            block_bottle.append(ResidualBlock(dim_in=curr_dim, dim_out=curr_dim))
        self.block_bottle = nn.Sequential(*block_bottle)

        # Up-sampling layers.
        block_up1 = []
        block_up1.append(nn.ConvTranspose3d(curr_dim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_up1.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        block_up1.append(nn.ReLU(inplace=True))
        block_up1.append(ChannelSpatialSELayer3D(num_channels=curr_dim // 2))
        self.block_up1 = nn.Sequential(*block_up1)
        curr_dim = curr_dim // 2

        block_up2 = []
        block_up2.append(nn.ConvTranspose3d(curr_dim + block_down2_outdim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_up2.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        block_up2.append(nn.ReLU(inplace=True))
        block_up2.append(ChannelSpatialSELayer3D(num_channels=curr_dim // 2))
        self.block_up2 = nn.Sequential(*block_up2)
        curr_dim = curr_dim // 2

        block_up3 = []
        block_up3.append(nn.ConvTranspose3d(curr_dim + block_down1_outdim, curr_dim // 2, kernel_size=4, stride=2, padding=1, bias=False))
        block_up3.append(nn.InstanceNorm3d(curr_dim // 2, affine=True, track_running_stats=True))
        block_up3.append(nn.ReLU(inplace=True))
        block_up3.append(ChannelSpatialSELayer3D(num_channels=curr_dim // 2))
        self.block_up3 = nn.Sequential(*block_up3)
        curr_dim = curr_dim // 2

        block_final = []
        block_final.append(nn.Conv3d(curr_dim + block_init_outdim, 1, kernel_size=7, stride=1, padding=3, bias=False))
        block_final.append(nn.Tanh())
        self.block_final = nn.Sequential(*block_final)

    def forward(self, x, c):
        # Replicate spatially and concatenate domain information.
        # Note that this type of label conditioning does not work at all if we use reflection padding in Conv2d.
        # This is because instance normalization ignores the shifting (or bias) effect.
        c = c.view(c.size(0), c.size(1), 1, 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3), x.size(4))
        x = torch.cat([x, c], dim=1)

        x_init = self.block_init(x)

        x_down1 = self.block_down1(x_init)
        x_down2 = self.block_down2(x_down1)
        x_down3 = self.block_down3(x_down2)

        x_bottle = self.block_bottle(x_down3)

        x_up1 = self.block_up1(x_bottle)
        x_up2 = self.block_up2(torch.cat((x_up1, x_down2), 1))
        x_up3 = self.block_up3(torch.cat((x_up2, x_down1), 1))

        x_final = self.block_final(torch.cat((x_up3, x_init), 1))
        return x_final


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization."""
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.InstanceNorm3d(dim_out, affine=True, track_running_stats=True))

    def forward(self, x):
        return x + self.main(x)


class ChannelSELayer3D(nn.Module):
    """
    3D extension of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
        *Zhu et al., AnatomyNet, arXiv:arXiv:1808.05238*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output tensor
        """
        batch_size, num_channels, D, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = self.avg_pool(input_tensor)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor.view(batch_size, num_channels)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size, num_channels, 1, 1, 1))

        return output_tensor


class SpatialSELayer3D(nn.Module):
    """
    3D extension of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """
        :param num_channels: No of input channels
        """
        super(SpatialSELayer3D, self).__init__()
        self.conv = nn.Conv3d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """
        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # channel squeeze
        batch_size, channel, D, H, W = input_tensor.size()

        if weights:
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)

        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        output_tensor = torch.mul(input_tensor, squeeze_tensor.view(batch_size, 1, D, H, W))

        return output_tensor


class ChannelSpatialSELayer3D(nn.Module):
    """
       3D extension of concurrent spatial and channel squeeze & excitation:
           *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, arXiv:1803.02579*
       """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer3D, self).__init__()
        self.cSE = ChannelSELayer3D(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer3D(num_channels)

    def forward(self, input_tensor):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, D, H, W)
        :return: output_tensor
        """
        # output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        output_tensor = (self.cSE(input_tensor) + self.sSE(input_tensor))/2
        return output_tensor


''' 
Discriminators
'''


class Discriminator_DC(nn.Module):
    """
    Discriminator network with PatchGAN
    1. D: Classify real or fake
    2. C: Classify domain
    """
    def __init__(self, image_size=128, conv_dim=64, c_dim=3, repeat_num=5):
        super(Discriminator_DC, self).__init__()
        layers = []
        layers.append(nn.Conv3d(1, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv3d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = int(image_size / np.power(2, repeat_num))
        self.main = nn.Sequential(*layers)
        self.conv1 = nn.Conv3d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv3d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        out_src = self.conv1(h)
        out_cls = self.conv2(h)
        return out_src, out_cls.view(out_cls.size(0), out_cls.size(1))


# ============================================================
# __main__: Automated test suite for 5 ablated functions
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
    print("UCAN: 3D Unified Anatomy-aware Cyclic Adversarial Network")
    print("Automated reproduction benchmark - 5 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/5: ChannelSELayer3D.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] ChannelSELayer3D.forward - channel squeeze-excitation")
    try:
        layer = ChannelSELayer3D(num_channels=4, reduction_ratio=2).to(device)
        x = torch.rand(2, 4, 4, 5, 6, device=device) + 0.25
        y = layer(x)
        check("ChannelSE output not None", y is not None)
        if y is not None:
            ratio = y / x
            channel_broadcast = ratio.std(dim=(2, 3, 4)).max().item() < 1e-5
            gate_range = ratio.min().item() >= 0.0 and ratio.max().item() <= 1.0
            check("ChannelSE output shape", tuple(y.shape) == (2, 4, 4, 5, 6), f"expected (2, 4, 4, 5, 6), got {tuple(y.shape)}")
            check("ChannelSE output finite", torch.isfinite(y).all().item())
            check("ChannelSE applies channel-wise gates in [0, 1]", channel_broadcast and gate_range)
        else:
            skip_checks(3, "ChannelSELayer3D.forward returned None")
    except Exception as e:
        skip_checks(4, f"ChannelSELayer3D.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/5: SpatialSELayer3D.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] SpatialSELayer3D.forward - spatial squeeze-excitation")
    try:
        layer = SpatialSELayer3D(num_channels=4).to(device)
        x = torch.rand(2, 4, 4, 5, 6, device=device) + 0.25
        y = layer(x)
        check("SpatialSE output not None", y is not None)
        if y is not None:
            ratio = y / x
            spatial_broadcast = ratio.std(dim=1).max().item() < 1e-5
            gate_range = ratio.min().item() >= 0.0 and ratio.max().item() <= 1.0
            check("SpatialSE output shape", tuple(y.shape) == (2, 4, 4, 5, 6), f"expected (2, 4, 4, 5, 6), got {tuple(y.shape)}")
            check("SpatialSE output finite", torch.isfinite(y).all().item())
            check("SpatialSE applies spatial gates shared across channels", spatial_broadcast and gate_range)
        else:
            skip_checks(3, "SpatialSELayer3D.forward returned None")
    except Exception as e:
        skip_checks(4, f"SpatialSELayer3D.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/5: ChannelSpatialSELayer3D.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] ChannelSpatialSELayer3D.forward - concurrent cSE/sSE fusion")
    try:
        layer = ChannelSpatialSELayer3D(num_channels=4, reduction_ratio=2).to(device)
        x = torch.rand(2, 4, 4, 5, 6, device=device) + 0.25
        y = layer(x)
        check("ChannelSpatialSE output not None", y is not None)
        if y is not None:
            with torch.no_grad():
                expected = (layer.cSE(x) + layer.sSE(x)) / 2
            check("ChannelSpatialSE output shape", tuple(y.shape) == (2, 4, 4, 5, 6), f"expected (2, 4, 4, 5, 6), got {tuple(y.shape)}")
            check("ChannelSpatialSE output finite", torch.isfinite(y).all().item())
            check("ChannelSpatialSE averages channel and spatial branches", torch.allclose(y, expected, atol=1e-6))
        else:
            skip_checks(3, "ChannelSpatialSELayer3D.forward returned None")
    except Exception as e:
        skip_checks(4, f"ChannelSpatialSELayer3D.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/5: Generator_DUSENET.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] Generator_DUSENET.forward - conditioned 3D DUSE-Net synthesis")
    try:
        model = Generator_DUSENET(input_dim=1, conv_dim=4, c_dim=3, repeat_num=1).to(device)
        model.eval()
        x = torch.randn(2, 1, 16, 16, 16, device=device)
        c_a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
        c_b = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], device=device)
        with torch.no_grad():
            y_a = model(x, c_a)
            y_b = model(x, c_b)
        check("Generator output not None", y_a is not None)
        if y_a is not None:
            condition_changes_output = not torch.allclose(y_a, y_b, atol=1e-6)
            range_ok = y_a.min().item() >= -1.0001 and y_a.max().item() <= 1.0001
            check("Generator output shape", tuple(y_a.shape) == (2, 1, 16, 16, 16), f"expected (2, 1, 16, 16, 16), got {tuple(y_a.shape)}")
            check("Generator output finite and tanh-bounded", torch.isfinite(y_a).all().item() and range_ok)
            check("Generator domain conditioning affects output", condition_changes_output)
        else:
            skip_checks(3, "Generator_DUSENET.forward returned None")
    except Exception as e:
        skip_checks(4, f"Generator_DUSENET.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 5/5: Discriminator_DC.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] Discriminator_DC.forward - PatchGAN source and domain heads")
    try:
        model = Discriminator_DC(image_size=16, conv_dim=4, c_dim=3, repeat_num=2).to(device)
        x = torch.randn(2, 1, 16, 16, 16, device=device)
        result = model(x)
        check("Discriminator returns two outputs", isinstance(result, tuple) and len(result) == 2)
        if isinstance(result, tuple) and len(result) == 2 and all(item is not None for item in result):
            out_src, out_cls = result
            check("Discriminator source and class shapes", tuple(out_src.shape) == (2, 1, 4, 4, 4) and tuple(out_cls.shape) == (2, 3), f"got {tuple(out_src.shape)} and {tuple(out_cls.shape)}")
            check("Discriminator outputs finite", torch.isfinite(out_src).all().item() and torch.isfinite(out_cls).all().item())
            check("Discriminator keeps PatchGAN spatial map and domain logits", out_src.ndim == 5 and out_cls.ndim == 2)
        else:
            skip_checks(3, "Discriminator_DC.forward returned None or malformed output")
    except Exception as e:
        skip_checks(4, f"Discriminator_DC.forward raised {type(e).__name__}: {e}")
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
