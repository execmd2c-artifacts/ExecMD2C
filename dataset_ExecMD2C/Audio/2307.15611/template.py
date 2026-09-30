# -*- coding: utf-8 -*-
"""Core GAN model components extracted from bin2bin-GAN-PLC.

This file consolidates the official generator and discriminator architecture
components. Demo entry points, data handling, parameter-update loops, and file-save
handling are intentionally excluded.
"""

import torch
import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super(Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False, padding_mode="reflect")
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2),
        )

        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)
        self.down = down

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    Input: x [N, 1, 256, 256]
    Output: G(x) [N, 1, 256, 256]
    """
    def __init__(self, in_channels=1, features=64):
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
        )
        self.down1 = Block(features, features * 2, down=True, act="leaky", use_dropout=False)
        self.down2 = Block(
            features * 2, features * 4, down=True, act="leaky", use_dropout=False
        )
        self.down3 = Block(
            features * 4, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down4 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down5 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.down6 = Block(
            features * 8, features * 8, down=True, act="leaky", use_dropout=False
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1), nn.ReLU()
        )

        self.up1 = Block(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up3 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=True
        )
        self.up4 = Block(
            features * 8 * 2, features * 8, down=False, act="relu", use_dropout=False
        )
        self.up5 = Block(
            features * 8 * 2, features * 4, down=False, act="relu", use_dropout=False
        )
        self.up6 = Block(
            features * 4 * 2, features * 2, down=False, act="relu", use_dropout=False
        )
        self.up7 = Block(features * 2 * 2, features, down=False, act="relu", use_dropout=False)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        """
        TODO: Run the pix2pix-style U-Net generator.

        Input:
            x: Tensor of shape (batch, in_channels, 256, 256), representing a
                lossy time-frequency spectrogram.

        Output:
            Tensor of shape (batch, in_channels, 256, 256), representing the
            inpainted spectrogram with values bounded by the final activation.

"""
        pass


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, k_size, stride, 1, bias=False, padding_mode="reflect"
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.conv(x)


class PatchDiscriminator(nn.Module):
    """
    PatchGAN based discriminator
    Input: [x, y] or [x, G(y)], where
     x = Lossy spectrogram [1, 256, 256]
     y = Clean spectrogram [1, 256, 256]
     G(x) = Inpainted spectrogram [1, 256, 256]
    Output: 2D map of logits
    """
    def __init__(self, in_channels=1, k_size=(4,4), features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=k_size,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels, feature, k_size, stride=1 if feature == features[-1] else 2),
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(
                in_channels, 1, k_size, stride=1, padding=1, padding_mode="reflect"
            ),
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        """
        TODO: Run the conditional PatchGAN discriminator.

        Input:
            x: Tensor of shape (batch, in_channels, height, width), the lossy
                spectrogram condition.
            y: Tensor of shape (batch, in_channels, height, width), either the clean
                target spectrogram or generator output.

        Output:
            Tensor of shape (batch, 1, patch_height, patch_width), containing patch
            logits.

"""
        pass


class PerPixelDiscriminator(nn.Module):
    """
    PatchGAN based discriminator
    Input: [x, y] or [x, G(y)], where
     x = Lossy spectrogram [1, 256, 256]
     y = Clean spectrogram [1, 256, 256]
     G(x) = Inpainted spectrogram [1, 256, 256]
    Output: 2D map of logits with the same shape as Input
    """
    def __init__(self, in_channels=1, features=[64, 128]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels * 2,
                features[0],
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.LeakyReLU(0.2),
        )

        self.model = nn.Sequential(
            nn.Conv2d(features[0],
                      features[1],
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=False,
                      ),
            nn.BatchNorm2d(features[1]),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features[1],
                      1,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      )
        )

    def forward(self, x, y):
        """
        TODO: Run the conditional per-pixel discriminator.

        Input:
            x: Tensor of shape (batch, in_channels, height, width), the lossy
                spectrogram condition.
            y: Tensor of shape (batch, in_channels, height, width), either the clean
                target spectrogram or generator output.

        Output:
            Tensor of shape (batch, 1, height, width), containing one logit per
            spatial bin.

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
    print("bin2bin-GAN-PLC: U-Net generator and conditional discriminators")
    print("=" * 70)

    print("-" * 70)
    print("[Test 1/4] Block and CNNBlock primitives")
    try:
        down = Block(1, 4, down=True, act="leaky", use_dropout=False)
        up = Block(4, 2, down=False, act="relu", use_dropout=True)
        cnn = CNNBlock(2, 5, k_size=(4, 4), stride=2)
        down.eval()
        up.eval()
        cnn.eval()
        x = torch.randn(2, 1, 32, 32)
        d = down(x)
        u = up(torch.randn(2, 4, 16, 16))
        c = cnn(torch.randn(2, 2, 32, 32))
        check("Block down output not None", d is not None)
        if d is not None:
            check("Block down halves spatial size", d.shape == (2, 4, 16, 16), str(tuple(d.shape)))
            check("Block down output finite", torch.isfinite(d).all().item())
        else:
            skip_checks(2, "down Block returned None")
        check("Block up doubles spatial size", u.shape == (2, 2, 32, 32), str(tuple(u.shape)))
        check("Block records dropout flag", up.use_dropout is True)
        check("CNNBlock output shape", c.shape == (2, 5, 16, 16), str(tuple(c.shape)))
        check("CNNBlock output finite", torch.isfinite(c).all().item())
    except Exception as exc:
        skip_checks(7, f"primitive blocks raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 2/4] Generator.forward")
    try:
        gen = Generator(in_channels=1, features=8)
        gen.eval()
        x = torch.randn(2, 1, 256, 256, requires_grad=True)
        y = gen(x)
        check("Generator output not None", y is not None)
        if y is not None:
            check("Generator preserves spectrogram shape", y.shape == (2, 1, 256, 256), str(tuple(y.shape)))
            check("Generator output finite", torch.isfinite(y).all().item())
            check("Generator tanh bounds output", (y.abs() <= 1.000001).all().item())
            check("Generator first skip concat doubles final channels", gen.final_up[0].in_channels == 16, str(gen.final_up[0].in_channels))
            y.mean().backward()
            check("Generator supports gradient flow", gen.initial_down[0].weight.grad is not None and torch.isfinite(gen.initial_down[0].weight.grad).all().item())
        else:
            skip_checks(5, "Generator returned None")
    except Exception as exc:
        skip_checks(6, f"Generator forward raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 3/4] PatchDiscriminator.forward")
    try:
        disc = PatchDiscriminator(in_channels=1, k_size=(4, 4), features=[8, 16, 32, 64])
        disc.eval()
        x = torch.randn(2, 1, 256, 256, requires_grad=True)
        y = torch.randn(2, 1, 256, 256)
        out = disc(x, y)
        out_changed = disc(x, y + 0.5)
        check("PatchDiscriminator output not None", out is not None)
        if out is not None:
            check("PatchDiscriminator output patch shape", out.shape == (2, 1, 30, 30), str(tuple(out.shape)))
            check("PatchDiscriminator output finite", torch.isfinite(out).all().item())
            check("PatchDiscriminator initial layer sees paired channels", disc.initial[0].in_channels == 2, str(disc.initial[0].in_channels))
            check("PatchDiscriminator condition affects logits", not torch.allclose(out, out_changed))
            out.mean().backward()
            check("PatchDiscriminator supports gradient flow", disc.initial[0].weight.grad is not None and torch.isfinite(disc.initial[0].weight.grad).all().item())
        else:
            skip_checks(5, "PatchDiscriminator returned None")
    except Exception as exc:
        skip_checks(6, f"PatchDiscriminator forward raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 4/4] PerPixelDiscriminator.forward")
    try:
        disc = PerPixelDiscriminator(in_channels=1, features=[8, 16])
        disc.eval()
        x = torch.randn(2, 1, 64, 64, requires_grad=True)
        y = torch.randn(2, 1, 64, 64)
        out = disc(x, y)
        out_changed = disc(x, y + 0.5)
        check("PerPixelDiscriminator output not None", out is not None)
        if out is not None:
            check("PerPixelDiscriminator preserves spatial shape", out.shape == (2, 1, 64, 64), str(tuple(out.shape)))
            check("PerPixelDiscriminator output finite", torch.isfinite(out).all().item())
            check("PerPixelDiscriminator initial layer sees paired channels", disc.initial[0].in_channels == 2, str(disc.initial[0].in_channels))
            check("PerPixelDiscriminator condition affects logits", not torch.allclose(out, out_changed))
            out.mean().backward()
            check("PerPixelDiscriminator supports gradient flow", disc.initial[0].weight.grad is not None and torch.isfinite(disc.initial[0].weight.grad).all().item())
        else:
            skip_checks(5, "PerPixelDiscriminator returned None")
    except Exception as exc:
        skip_checks(6, f"PerPixelDiscriminator forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
