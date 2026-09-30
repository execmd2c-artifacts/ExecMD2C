# ============================================================
# ground_truth.py - Keyword Spotting ConvMixer Core Model Components
# Source:
#   Speech/Keyword-Spotting-ConvMixer-main/models/activations.py
#   Speech/Keyword-Spotting-ConvMixer-main/models/utils.py
#   Speech/Keyword-Spotting-ConvMixer-main/models/main_layers.py
#   Speech/Keyword-Spotting-ConvMixer-main/models/ConvMixer.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: models/activations.py] ---
class Swish(nn.Module):
    """Swish is a smooth, non-monotonic function that
    consistently matches or outperforms ReLU on
    deep networks applied to a variety of challenging
    domains such as Image classification and Machine translation.
    """
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, inputs):
        return inputs * inputs.sigmoid()


# --- [Original file: models/utils.py] ---
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, pointwise=True):

        super(SeparableConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding,
                                dilation=dilation, bias=bias,)

        if pointwise:
            self.pointwise = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=bias,)
        else:
            self.pointwise = nn.Identity()

    def forward(self, x):
        x = self.conv1d(x)
        x = self.pointwise(x)
        return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, padding=0,
                 dilation=1, bias=False, pointwise=False):

        super(SeparableConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding,
                                dilation=dilation, bias=bias,)
        if pointwise:
            self.pointwise = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                       kernel_size=1, stride=1, padding=0, bias=bias,)
        else:
            self.pointwise = nn.Identity()

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pointwise(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        """ Perform feed forward layer for mixer
        Parameter args:
            dim: in_channel dimension
            hidden_dim: intermediate dimension during FFN
        """
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):
    def __init__(self, time_dim, freq_dim, dropout=0.):
        super(MixerBlock, self).__init__()

        self.time_mix = nn.Sequential(
            nn.LayerNorm(time_dim),
            FeedForward(time_dim, time_dim // 4, dropout),)

        self.freq_mix = nn.Sequential(
            nn.LayerNorm(freq_dim),
            FeedForward(freq_dim, freq_dim // 2, dropout),)

    def forward(self, x):
        """[TODO] Reproduce the ConvMixer time/frequency interaction block.

        Input:
            x: Tensor with shape (batch, freq_dim, time_dim), where channels
               represent the learned frequency/feature dimension and the last
               axis represents temporal positions.

        Output:
            Tensor with shape (batch, freq_dim, time_dim).

"""
        pass


# --- [Original file: models/main_layers.py] ---
class ConvMixerBlock(nn.Module):
    """ Performs convolution with mlp mixer. Processing steps ::
        1) freq depthwise separable convolution
        2) time depthwise separable convolution
        3) mlp mixer
        4) skip connection

    """
    def __init__(self, temporal_length, num_temporal_channels=64,
                 temporal_kernel_size=3, temporal_padding=1,
                 freq_domain_kernel_size=5, freq_domain_padding=2,
                 num_freq_filters=64,
                 dropout=0.,
                 bias=False):
        super(ConvMixerBlock, self).__init__()

        ## frequency domain encoding
        self.frequency_domain_encoding = nn.Sequential(
            nn.Conv2d(1, num_freq_filters, kernel_size=3, stride=1, padding=1, bias=bias),
            Swish(),
            SeparableConv2d(num_freq_filters, num_freq_filters,
                            kernel_size=(freq_domain_kernel_size, 1),
                            stride=1, padding=(freq_domain_padding, 0), bias=bias),
            Swish(),
            nn.Conv2d(num_freq_filters, 1, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.BatchNorm2d(1),
            Swish(),)

        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(num_temporal_channels, num_temporal_channels,
                            kernel_size=temporal_kernel_size,
                            stride=1, padding=temporal_padding, bias=bias),
            nn.BatchNorm1d(num_temporal_channels),
            Swish(),)

        self.dropout = nn.Dropout(p=dropout)

        ## mixer
        self.mixer = nn.Sequential(
            MixerBlock(time_dim=temporal_length, freq_dim=num_temporal_channels, dropout=0.),
            Swish(),)

    def forward(self, x):
        """[TODO] Reproduce the feature-interactive ConvMixer block.

        Input:
            x: Tensor with shape (batch, num_temporal_channels,
               temporal_length).

        Output:
            Tensor with shape (batch, num_temporal_channels,
            temporal_length).

"""
        pass


class PreConvBlock(nn.Module):

    def __init__(self, time_length, time_channels=64,
                 kernel_size=3, padding=1,
                 dropout=0., bias=False):

        super(PreConvBlock, self).__init__()

        ## temporal domain encoding
        self.temporal_domain_encoding = nn.Sequential(
            SeparableConv1d(time_channels, time_channels, kernel_size=kernel_size,
                            stride=1, padding=padding, bias=bias),
            nn.BatchNorm1d(time_channels),
            Swish(),
            SeparableConv1d(time_channels, time_channels, kernel_size=1,
                            stride=1, padding=0, bias=bias),
            nn.BatchNorm1d(time_channels),
            Swish(),)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """[TODO] Reproduce the pre-ConvMixer temporal residual block.

        Input:
            x: Tensor with shape (batch, time_channels, time_length).

        Output:
            Tensor with shape (batch, time_channels, time_length).

"""
        pass


# --- [Original file: models/ConvMixer.py] ---
class KWSConvMixer(nn.Module):
    def __init__(self, input_size,
                 num_classes,
                 feat_dim=64,
                 dropout=0.):

        """ KWS Convolutional Mixer Model
        input:: audio spectrogram, default input shape [BS, 98, 64]
        output:: prediction of command classes, default 12 classes
        """

        super(KWSConvMixer, self).__init__()

        self.num_classes = num_classes
        self.temporal_dim, self.frequency_dim = input_size

        ## init conv (channel): output shape BS x feat_dim x T
        self.conv1 = nn.Sequential(
            SeparableConv1d(self.frequency_dim, feat_dim,
                            kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(feat_dim),
            Swish(),)

        self.preConvMixer = PreConvBlock(self.temporal_dim, feat_dim,
                                         kernel_size=7, padding=3,
                                         dropout=dropout)

        self.convMixer1 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=9, temporal_padding=4,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=64,
                                         dropout=dropout)

        self.convMixer2 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=11, temporal_padding=5,
                                         freq_domain_kernel_size=5, freq_domain_padding=2,
                                         num_freq_filters=32,
                                         dropout=dropout)

        self.convMixer3 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=13, temporal_padding=6,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=16,
                                         dropout=dropout)

        self.convMixer4 = ConvMixerBlock(self.temporal_dim, feat_dim,
                                         temporal_kernel_size=15, temporal_padding=7,
                                         freq_domain_kernel_size=7, freq_domain_padding=3,
                                         num_freq_filters=8,
                                         dropout=dropout)

        self.conv2 = nn.Sequential(
            SeparableConv1d(feat_dim, feat_dim*2,
                            kernel_size=17, stride=1, padding=8, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),)

        self.conv3 = nn.Sequential(
            SeparableConv1d(feat_dim*2, feat_dim*2,
                            kernel_size=19, stride=1, padding=18, dilation=2, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),)

        self.conv4 = nn.Sequential(
            SeparableConv1d(feat_dim*2, feat_dim*2,
                            kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(feat_dim*2),
            Swish(),)

        self.pooling = torch.nn.AdaptiveMaxPool1d(1)
        self.mlp_head = nn.Sequential(nn.Linear(feat_dim*2, self.num_classes, bias=True))

    def forward(self, x):
        """[TODO] Reproduce the end-to-end keyword-spotting ConvMixer path.

        Input:
            x: Tensor with shape (batch, temporal_dim, frequency_dim), e.g.
               log-mel spectrogram frames.

        Output:
            Tuple of:
                - logits with shape (batch, num_classes)
                - pooled embedding with shape (batch, 2 * feat_dim)

"""
        pass


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
    print("Keyword Spotting ConvMixer")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: MixerBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] MixerBlock.forward - time/frequency feature interaction")
    try:
        module = MixerBlock(time_dim=16, freq_dim=8, dropout=0.0).to(device)
        x = torch.randn(2, 8, 16, device=device, requires_grad=True)
        y = module(x)
        check("MixerBlock output not None", y is not None)
        if y is not None:
            check("MixerBlock output shape", tuple(y.shape) == (2, 8, 16), f"expected (2, 8, 16), got {tuple(y.shape)}")
            check("MixerBlock output finite", torch.isfinite(y).all().item())
            check("MixerBlock changes features through mixer residuals", not torch.allclose(y, x))
            (y ** 2).sum().backward()
            grad_ok = module.time_mix[1].net[0].weight.grad is not None and module.freq_mix[1].net[0].weight.grad is not None
            check("MixerBlock routes gradient to both time and frequency mixers", grad_ok)
        else:
            skip_checks(4, "MixerBlock.forward returned None")
    except Exception as e:
        skip_checks(5, f"MixerBlock.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: PreConvBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] PreConvBlock.forward - temporal residual pre-mixing")
    try:
        module = PreConvBlock(time_length=16, time_channels=8, kernel_size=3, padding=1, dropout=0.0).to(device)
        x = torch.randn(2, 8, 16, device=device, requires_grad=True)
        y = module(x)
        check("PreConvBlock output not None", y is not None)
        if y is not None:
            check("PreConvBlock output shape", tuple(y.shape) == (2, 8, 16), f"expected (2, 8, 16), got {tuple(y.shape)}")
            check("PreConvBlock output finite", torch.isfinite(y).all().item())
            zero_module = PreConvBlock(time_length=16, time_channels=8, kernel_size=3, padding=1, dropout=0.0).to(device)
            with torch.no_grad():
                zero_module.temporal_domain_encoding[0].conv1d.weight.zero_()
                zero_module.temporal_domain_encoding[0].pointwise.weight.zero_()
                zero_module.temporal_domain_encoding[3].conv1d.weight.zero_()
                zero_module.temporal_domain_encoding[3].pointwise.weight.zero_()
            zero_module.eval()
            zero_y = zero_module(x.detach())
            check("PreConvBlock preserves identity residual when branch is zeroed", torch.allclose(zero_y, x.detach(), atol=1e-6))
            (y ** 2).sum().backward()
            check("PreConvBlock routes gradient to temporal separable conv", module.temporal_domain_encoding[0].conv1d.weight.grad is not None and module.temporal_domain_encoding[0].conv1d.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "PreConvBlock.forward returned None")
    except Exception as e:
        skip_checks(5, f"PreConvBlock.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: ConvMixerBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] ConvMixerBlock.forward - frequency/temporal/mixer residual unit")
    try:
        module = ConvMixerBlock(
            temporal_length=16,
            num_temporal_channels=8,
            temporal_kernel_size=3,
            temporal_padding=1,
            freq_domain_kernel_size=3,
            freq_domain_padding=1,
            num_freq_filters=4,
            dropout=0.0,
        ).to(device)
        x = torch.randn(2, 8, 16, device=device, requires_grad=True)
        y = module(x)
        check("ConvMixerBlock output not None", y is not None)
        if y is not None:
            check("ConvMixerBlock output shape", tuple(y.shape) == (2, 8, 16), f"expected (2, 8, 16), got {tuple(y.shape)}")
            check("ConvMixerBlock output finite", torch.isfinite(y).all().item())
            check("ConvMixerBlock combines multiple residual branches", not torch.allclose(y, x))
            (y ** 2).sum().backward()
            grad_ok = (
                module.frequency_domain_encoding[0].weight.grad is not None and
                module.temporal_domain_encoding[0].conv1d.weight.grad is not None and
                module.mixer[0].time_mix[1].net[0].weight.grad is not None
            )
            check("ConvMixerBlock routes gradient to frequency, temporal, and mixer paths", grad_ok)
        else:
            skip_checks(4, "ConvMixerBlock.forward returned None")
    except Exception as e:
        skip_checks(5, f"ConvMixerBlock.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: KWSConvMixer.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] KWSConvMixer.forward - end-to-end keyword spotting classifier")
    try:
        model = KWSConvMixer(input_size=(16, 8), num_classes=12, feat_dim=8, dropout=0.0).to(device)
        x = torch.randn(2, 16, 8, device=device, requires_grad=True)
        logits, embedding = model(x)
        check("KWSConvMixer output not None", logits is not None and embedding is not None)
        if logits is not None and embedding is not None:
            check("KWSConvMixer logits shape", tuple(logits.shape) == (2, 12), f"expected (2, 12), got {tuple(logits.shape)}")
            check("KWSConvMixer embedding shape", tuple(embedding.shape) == (2, 16), f"expected (2, 16), got {tuple(embedding.shape)}")
            check("KWSConvMixer outputs finite", torch.isfinite(logits).all().item() and torch.isfinite(embedding).all().item())
            logits.sum().backward()
            check("KWSConvMixer routes gradient to initial acoustic projection", model.conv1[0].conv1d.weight.grad is not None and model.conv1[0].conv1d.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "KWSConvMixer.forward returned None output(s)")
    except Exception as e:
        skip_checks(5, f"KWSConvMixer.forward raised {type(e).__name__}: {e}")
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
