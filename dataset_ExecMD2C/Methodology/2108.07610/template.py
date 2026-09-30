"""
ground_truth.py for DRAEM core model components.

Source-consolidated from:
- model_unet.py

Only neural network architecture components are included. Optimization loops,
data loaders, score computation, CLI wrappers, saved-weight I/O, and loss
orchestration are intentionally excluded.
"""

import torch
import torch.nn as nn


class ReconstructiveSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_width=128):
        super(ReconstructiveSubNetwork, self).__init__()
        self.encoder = EncoderReconstructive(in_channels, base_width)
        self.decoder = DecoderReconstructive(base_width, out_channels=out_channels)

    def forward(self, x):
        """
        TODO: Run the reconstructive DRAEM sub-network forward pass.

        Input:
            x: image tensor of shape (batch, in_channels, height, width).
        Output:
            reconstructed image tensor of shape
            (batch, out_channels, height, width).

        Encode the image through the reconstructive encoder to obtain the
        deepest bottleneck feature map, then decode that feature map back to the
        original spatial resolution. Preserve the exact encoder-to-decoder data
        flow and do not add skip connections that are not present in this
        reconstructive branch.
        """
        pass

class DiscriminativeSubNetwork(nn.Module):
    def __init__(self,in_channels=3, out_channels=3, base_channels=64, out_features=False):
        super(DiscriminativeSubNetwork, self).__init__()
        base_width = base_channels
        self.encoder_segment = EncoderDiscriminative(in_channels, base_width)
        self.decoder_segment = DecoderDiscriminative(base_width, out_channels=out_channels)
        #self.segment_act = torch.nn.Sigmoid()
        self.out_features = out_features
    def forward(self, x):
        """
        TODO: Run the discriminative DRAEM segmentation sub-network forward pass.

        Input:
            x: concatenated reconstruction/original tensor of shape
               (batch, in_channels, height, width), typically with six channels.
        Output:
            If out_features is False, return segmentation logits of shape
            (batch, out_channels, height, width). If out_features is True,
            return those logits plus the encoder feature tensors after the first
            pooling stage through the deepest stage.

        Use the discriminative encoder to produce all skip features, pass them to
        the discriminative decoder in the source order, and preserve the optional
        feature-return branch exactly.
        """
        pass

class EncoderDiscriminative(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderDiscriminative, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))

        self.mp5 = nn.Sequential(nn.MaxPool2d(2))
        self.block6 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        """
        TODO: Compute the discriminative encoder's six multi-scale feature maps.

        Input:
            x: tensor of shape (batch, in_channels, height, width).
        Output:
            Tuple (b1, b2, b3, b4, b5, b6) where spatial size is progressively
            reduced and channels follow base_width, 2x, 4x, then 8x widths.

        Apply each convolutional block and pooling stage in the same source
        order used by the original DRAEM implementation. Preserve the exact
        multi-scale tensors needed for decoder skip concatenation; changing the
        pooling sequence or omitting any returned feature will break the decoder
        channel and spatial alignment.
        """
        pass

class DecoderDiscriminative(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderDiscriminative, self).__init__()

        self.up_b = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db_b = nn.Sequential(
            nn.Conv2d(base_width*(8+8), base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 8),
            nn.ReLU(inplace=True)
        )


        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*(4+8), base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*(2+4), base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*(2+1), base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )



        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1,b2,b3,b4,b5,b6):
        """
        TODO: Decode discriminative encoder features into segmentation logits.

        Inputs:
            b1..b6: encoder feature tensors ordered from full resolution to the
            deepest representation. For a 128x128 input with base_width=8, their
            shapes are (batch, 8, 128, 128), (batch, 16, 64, 64),
            (batch, 32, 32, 32), (batch, 64, 16, 16),
            (batch, 64, 8, 8), and (batch, 64, 4, 4).
        Output:
            Segmentation logits of shape (batch, out_channels, height, width).

        Starting from the deepest feature, repeatedly upsample, concatenate with
        the matching encoder skip feature along the channel dimension, and apply
        the corresponding decoder block. The skip order and concatenation axis
        are essential for DRAEM's discriminative U-Net path.
        """
        pass



class EncoderReconstructive(nn.Module):
    def __init__(self, in_channels, base_width):
        super(EncoderReconstructive, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width,base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width*2,base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width*4,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))
        self.mp4 = nn.Sequential(nn.MaxPool2d(2))
        self.block5 = nn.Sequential(
            nn.Conv2d(base_width*8,base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True))


    def forward(self, x):
        """
        TODO: Compute the reconstructive encoder bottleneck feature.

        Input:
            x: image tensor of shape (batch, in_channels, height, width).
        Output:
            Deep bottleneck tensor with 8 * base_width channels and reduced
            spatial resolution; for a 128x128 input and base_width=8 this is
            (batch, 64, 8, 8).

        Follow the source encoder's sequence of convolutional blocks and pooling
        stages, returning only the final bottleneck feature. This branch does
        not expose skip features to the reconstructive decoder.
        """
        pass


class DecoderReconstructive(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super(DecoderReconstructive, self).__init__()

        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 8, base_width * 8, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 8),
                                 nn.ReLU(inplace=True))
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width*8, base_width*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*8),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 8, base_width * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width * 4),
                                 nn.ReLU(inplace=True))
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width*4, base_width*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width*2, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width*2),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width*2, base_width*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width*2, base_width*1, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width*1),
            nn.ReLU(inplace=True)
        )

        self.up4 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
                                 nn.BatchNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db4 = nn.Sequential(
            nn.Conv2d(base_width*1, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))
        #self.fin_out = nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1)

    def forward(self, b5):
        """
        TODO: Decode the reconstructive bottleneck back to an image tensor.

        Input:
            b5: bottleneck tensor of shape
                (batch, 8 * base_width, height / 16, width / 16).
        Output:
            Reconstruction tensor of shape
            (batch, out_channels, height, width).

        Apply the four reconstructive upsampling stages and their decoder blocks
        in order, without adding skip concatenations. Finish with the final
        output projection so that the spatial dimensions match the original
        image resolution.
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

    print("Running DRAEM core model benchmark checks...")
    device = torch.device("cpu")

    try:
        recon = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=8).to(device)
        recon.eval()
        x = torch.randn(2, 3, 128, 128, device=device)
        with torch.no_grad():
            y = recon(x)
        check("reconstructive output not None", y is not None)
        if y is not None:
            check("reconstructive output shape", tuple(y.shape) == (2, 3, 128, 128), f"got {tuple(y.shape)}")
            check("reconstructive output finite", torch.isfinite(y).all().item())
            check("reconstructive preserves spatial size", y.shape[-2:] == x.shape[-2:])
            check("reconstructive uses encoder decoder", isinstance(recon.encoder, EncoderReconstructive) and isinstance(recon.decoder, DecoderReconstructive))
        else:
            skip_checks(4, "reconstructive forward returned None")
    except Exception as exc:
        skip_checks(5, f"reconstructive checks raised {type(exc).__name__}: {exc}")

    try:
        enc = EncoderReconstructive(in_channels=3, base_width=8).to(device)
        dec = DecoderReconstructive(base_width=8, out_channels=3).to(device)
        enc.eval()
        dec.eval()
        x = torch.randn(2, 3, 128, 128, device=device)
        with torch.no_grad():
            latent = enc(x)
            rec = dec(latent) if latent is not None else None
        check("reconstructive encoder latent not None", latent is not None)
        if latent is not None:
            check("reconstructive encoder latent shape", tuple(latent.shape) == (2, 64, 8, 8), f"got {tuple(latent.shape)}")
            check("reconstructive encoder latent finite", torch.isfinite(latent).all().item())
        else:
            skip_checks(2, "reconstructive encoder returned None")
        check("reconstructive decoder output not None", rec is not None)
        if rec is not None:
            check("reconstructive decoder output shape", tuple(rec.shape) == (2, 3, 128, 128), f"got {tuple(rec.shape)}")
            check("reconstructive decoder output finite", torch.isfinite(rec).all().item())
        else:
            skip_checks(2, "reconstructive decoder returned None")
    except Exception as exc:
        skip_checks(6, f"reconstructive encoder/decoder checks raised {type(exc).__name__}: {exc}")

    try:
        disc = DiscriminativeSubNetwork(in_channels=6, out_channels=2, base_channels=8, out_features=True).to(device)
        disc.eval()
        x = torch.randn(2, 6, 128, 128, device=device)
        with torch.no_grad():
            outputs = disc(x)
        check("discriminative out_features tuple", isinstance(outputs, tuple) and len(outputs) == 6)
        if isinstance(outputs, tuple) and len(outputs) == 6:
            mask, b2, b3, b4, b5, b6 = outputs
            check("discriminative mask shape", tuple(mask.shape) == (2, 2, 128, 128), f"got {tuple(mask.shape)}")
            check("discriminative mask finite", torch.isfinite(mask).all().item())
            check("discriminative feature b2 shape", tuple(b2.shape) == (2, 16, 64, 64), f"got {tuple(b2.shape)}")
            check("discriminative deepest feature shape", tuple(b6.shape) == (2, 64, 4, 4), f"got {tuple(b6.shape)}")
            check("discriminative spatial restoration", mask.shape[-2:] == x.shape[-2:])
        else:
            skip_checks(5, "discriminative out_features branch did not return six tensors")
    except Exception as exc:
        skip_checks(6, f"discriminative branch checks raised {type(exc).__name__}: {exc}")

    try:
        enc = EncoderDiscriminative(in_channels=6, base_width=8).to(device)
        dec = DecoderDiscriminative(base_width=8, out_channels=2).to(device)
        enc.eval()
        dec.eval()
        x = torch.randn(2, 6, 128, 128, device=device)
        with torch.no_grad():
            feats = enc(x)
            mask = dec(*feats) if feats is not None else None
        check("discriminative encoder returns tuple", isinstance(feats, tuple) and len(feats) == 6)
        if isinstance(feats, tuple) and len(feats) == 6:
            expected_shapes = [(2, 8, 128, 128), (2, 16, 64, 64), (2, 32, 32, 32), (2, 64, 16, 16), (2, 64, 8, 8), (2, 64, 4, 4)]
            check("discriminative encoder feature shapes", [tuple(f.shape) for f in feats] == expected_shapes, f"got {[tuple(f.shape) for f in feats]}")
            check("discriminative encoder finite", all(torch.isfinite(f).all().item() for f in feats))
        else:
            skip_checks(2, "discriminative encoder did not return six tensors")
        check("discriminative decoder output not None", mask is not None)
        if mask is not None:
            check("discriminative decoder output shape", tuple(mask.shape) == (2, 2, 128, 128), f"got {tuple(mask.shape)}")
            check("discriminative decoder output finite", torch.isfinite(mask).all().item())
            probs = torch.softmax(mask, dim=1)
            check("discriminative logits softmax channel sum", torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5))
        else:
            skip_checks(3, "discriminative decoder returned None")
    except Exception as exc:
        skip_checks(7, f"discriminative encoder/decoder checks raised {type(exc).__name__}: {exc}")

    try:
        recon = ReconstructiveSubNetwork(in_channels=3, out_channels=3, base_width=8).to(device)
        disc = DiscriminativeSubNetwork(in_channels=6, out_channels=2, base_channels=8).to(device)
        recon.eval()
        disc.eval()
        x = torch.randn(2, 3, 128, 128, device=device)
        with torch.no_grad():
            rec = recon(x)
            joined = torch.cat((rec, x), dim=1) if rec is not None else None
            logits = disc(joined) if joined is not None else None
        check("draem joined input not None", joined is not None)
        if joined is not None:
            check("draem joined input shape", tuple(joined.shape) == (2, 6, 128, 128), f"got {tuple(joined.shape)}")
            check("draem joined input finite", torch.isfinite(joined).all().item())
        else:
            skip_checks(2, "joined reconstruction/original tensor was None")
        check("draem final logits not None", logits is not None)
        if logits is not None:
            check("draem final logits shape", tuple(logits.shape) == (2, 2, 128, 128), f"got {tuple(logits.shape)}")
            check("draem final logits finite", torch.isfinite(logits).all().item())
        else:
            skip_checks(2, "DRAEM segmentation logits were None")
    except Exception as exc:
        skip_checks(6, f"DRAEM integration checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
