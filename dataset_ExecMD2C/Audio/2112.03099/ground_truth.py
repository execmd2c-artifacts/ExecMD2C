# -*- coding: utf-8 -*-
"""Core vocoder components extracted from vocoder-benchmark.

This benchmark file consolidates the source-provided Parallel WaveGAN and
MelGAN model components into a single importable module. The original
repository is a multi-vocoder benchmark framework; command-line code,
corpus loading, objective wrappers, and orchestration wrappers are intentionally
excluded.
"""

import logging
import math
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch._tensor import Tensor


class CausalConv1d(torch.nn.Module):
    """CausalConv1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation: int = 1,
        bias: bool = True,
        pad: str = "ConstantPad1d",
        pad_params: Dict[str, float] = {"value": 0.0},
    ) -> None:
        """Initialize CausalConv1d module."""
        super(CausalConv1d, self).__init__()
        self.pad = getattr(torch.nn, pad)((kernel_size - 1) * dilation, **pad_params)
        self.conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size, dilation=dilation, bias=bias
        )

    def forward(self, x):
        """Calculate forward propagation."""
        return self.conv(self.pad(x))[:, :, : x.size(2)]


class CausalConvTranspose1d(torch.nn.Module):
    """CausalConvTranspose1d module with customized initialization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        bias: bool = True,
    ) -> None:
        """Initialize CausalConvTranspose1d module."""
        super(CausalConvTranspose1d, self).__init__()
        self.deconv = torch.nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )
        self.stride = stride

    def forward(self, x):
        """Calculate forward propagation."""
        return self.deconv(x)[:, :, : -self.stride]


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias) -> None:
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class ResidualBlock(torch.nn.Module):
    """Residual block module in WaveNet."""

    def __init__(
        self,
        kernel_size: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        dropout: float = 0.0,
        dilation: int = 1,
        bias: bool = True,
        use_causal_conv: bool = False,
    ) -> None:
        """Initialize ResidualBlock module."""
        super(ResidualBlock, self).__init__()
        self.dropout = dropout
        if use_causal_conv:
            padding = (kernel_size - 1) * dilation
        else:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            padding = (kernel_size - 1) // 2 * dilation
        self.use_causal_conv = use_causal_conv

        self.conv = Conv1d(
            residual_channels,
            gate_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if aux_channels > 0:
            self.conv1x1_aux = Conv1d1x1(aux_channels, gate_channels, bias=False)
        else:
            self.conv1x1_aux = None

        gate_out_channels = gate_channels // 2
        self.conv1x1_out = Conv1d1x1(gate_out_channels, residual_channels, bias=bias)
        self.conv1x1_skip = Conv1d1x1(gate_out_channels, skip_channels, bias=bias)

    def forward(self, x: Tensor, c):
        """Calculate forward propagation."""
        residual = x
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x)

        x = x[:, :, : residual.size(-1)] if self.use_causal_conv else x

        splitdim = 1
        xa, xb = x.split(x.size(splitdim) // 2, dim=splitdim)

        if c is not None:
            assert self.conv1x1_aux is not None
            c = self.conv1x1_aux(c)
            ca, cb = c.split(c.size(splitdim) // 2, dim=splitdim)
            xa, xb = xa + ca, xb + cb

        x = torch.tanh(xa) * torch.sigmoid(xb)

        s = self.conv1x1_skip(x)
        x = (self.conv1x1_out(x) + residual) * math.sqrt(0.5)

        return x, s


class Stretch2d(torch.nn.Module):
    """Stretch2d module."""

    def __init__(self, x_scale, y_scale, mode: str = "nearest") -> None:
        """Initialize Stretch2d module."""
        super(Stretch2d, self).__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale
        self.mode = mode

    def forward(self, x):
        """Calculate forward propagation."""
        return F.interpolate(
            x, scale_factor=(self.y_scale, self.x_scale), mode=self.mode
        )


class Conv2d(torch.nn.Conv2d):
    """Conv2d module with customized initialization."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize Conv2d module."""
        super(Conv2d, self).__init__(*args, **kwargs)

    def reset_parameters(self) -> None:
        """Reset parameters."""
        self.weight.data.fill_(1.0 / np.prod(self.kernel_size))
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class UpsampleNetwork(torch.nn.Module):
    """Upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        use_causal_conv: bool = False,
    ) -> None:
        """Initialize upsampling network module."""
        super(UpsampleNetwork, self).__init__()
        self.use_causal_conv = use_causal_conv
        self.up_layers = torch.nn.ModuleList()
        for scale in upsample_scales:
            stretch = Stretch2d(scale, 1, interpolate_mode)
            self.up_layers += [stretch]

            assert (
                freq_axis_kernel_size - 1
            ) % 2 == 0, "Not support even number freq axis kernel size."
            freq_axis_padding = (freq_axis_kernel_size - 1) // 2
            kernel_size = (freq_axis_kernel_size, scale * 2 + 1)
            if use_causal_conv:
                padding = (freq_axis_padding, scale * 2)
            else:
                padding = (freq_axis_padding, scale)
            conv = Conv2d(1, 1, kernel_size=kernel_size, padding=padding, bias=False)
            self.up_layers += [conv]

            if nonlinear_activation is not None:
                nonlinear = getattr(torch.nn, nonlinear_activation)(
                    **nonlinear_activation_params
                )
                self.up_layers += [nonlinear]

    def forward(self, c):
        """Calculate forward propagation."""
        c = c.unsqueeze(1)
        for f in self.up_layers:
            if self.use_causal_conv and isinstance(f, Conv2d):
                c = f(c)[..., : c.size(-1)]
            else:
                c = f(c)
        return c.squeeze(1)


class ConvInUpsampleNetwork(torch.nn.Module):
    """Convolution + upsampling network module."""

    def __init__(
        self,
        upsample_scales,
        nonlinear_activation=None,
        nonlinear_activation_params={},
        interpolate_mode: str = "nearest",
        freq_axis_kernel_size: int = 1,
        aux_channels: int = 80,
        aux_context_window: int = 0,
        use_causal_conv: bool = False,
    ) -> None:
        """Initialize convolution + upsampling network module."""
        super(ConvInUpsampleNetwork, self).__init__()
        self.aux_context_window = aux_context_window
        self.use_causal_conv = use_causal_conv and aux_context_window > 0
        kernel_size = (
            aux_context_window + 1 if use_causal_conv else 2 * aux_context_window + 1
        )
        self.conv_in = Conv1d(
            aux_channels, aux_channels, kernel_size=kernel_size, bias=False
        )
        self.upsample = UpsampleNetwork(
            upsample_scales=upsample_scales,
            nonlinear_activation=nonlinear_activation,
            nonlinear_activation_params=nonlinear_activation_params,
            interpolate_mode=interpolate_mode,
            freq_axis_kernel_size=freq_axis_kernel_size,
            use_causal_conv=use_causal_conv,
        )

    def forward(self, c):
        """Calculate forward propagation."""
        c_ = self.conv_in(c)
        c = c_[:, :, : -self.aux_context_window] if self.use_causal_conv else c_
        return self.upsample(c)


class ResidualStack(torch.nn.Module):
    """Residual stack module introduced in MelGAN."""

    def __init__(
        self,
        kernel_size: int = 3,
        channels: int = 32,
        dilation: int = 1,
        bias: bool = True,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params={},
        use_causal_conv: bool = False,
    ) -> None:
        """Initialize ResidualStack module."""
        super(ResidualStack, self).__init__()

        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                getattr(torch.nn, pad)((kernel_size - 1) // 2 * dilation, **pad_params),
                torch.nn.Conv1d(
                    channels, channels, kernel_size, dilation=dilation, bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )
        else:
            self.stack = torch.nn.Sequential(
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                CausalConv1d(
                    channels,
                    channels,
                    kernel_size,
                    dilation=dilation,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
                torch.nn.Conv1d(channels, channels, 1, bias=bias),
            )

        self.skip_layer = torch.nn.Conv1d(channels, channels, 1, bias=bias)

    def forward(self, c):
        """Calculate forward propagation."""
        return self.stack(c) + self.skip_layer(c)


class MelGANGenerator(torch.nn.Module):
    """MelGAN generator module."""

    def __init__(
        self,
        in_channels: int = 80,
        out_channels: int = 1,
        kernel_size: int = 7,
        channels: int = 512,
        bias: bool = True,
        upsample_scales=[8, 8, 2, 2],
        stack_kernel_size: int = 3,
        stacks: int = 3,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params={},
        use_final_nonlinear_activation: bool = True,
        use_weight_norm: bool = True,
        use_causal_conv: bool = False,
    ) -> None:
        """Initialize MelGANGenerator module."""
        super(MelGANGenerator, self).__init__()

        assert channels >= np.prod(upsample_scales)
        assert channels % (2 ** len(upsample_scales)) == 0
        if not use_causal_conv:
            assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        layers = []
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(in_channels, channels, kernel_size, bias=bias),
            ]
        else:
            layers += [
                CausalConv1d(
                    in_channels,
                    channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]

        for i, upsample_scale in enumerate(upsample_scales):
            layers += [
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
            ]
            if not use_causal_conv:
                layers += [
                    torch.nn.ConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        padding=upsample_scale // 2 + upsample_scale % 2,
                        output_padding=upsample_scale % 2,
                        bias=bias,
                    )
                ]
            else:
                layers += [
                    CausalConvTranspose1d(
                        channels // (2**i),
                        channels // (2 ** (i + 1)),
                        upsample_scale * 2,
                        stride=upsample_scale,
                        bias=bias,
                    )
                ]

            for j in range(stacks):
                layers += [
                    ResidualStack(
                        kernel_size=stack_kernel_size,
                        channels=channels // (2 ** (i + 1)),
                        dilation=stack_kernel_size**j,
                        bias=bias,
                        nonlinear_activation=nonlinear_activation,
                        nonlinear_activation_params=nonlinear_activation_params,
                        pad=pad,
                        pad_params=pad_params,
                        use_causal_conv=use_causal_conv,
                    )
                ]

        layers += [
            getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params)
        ]
        if not use_causal_conv:
            layers += [
                getattr(torch.nn, pad)((kernel_size - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                ),
            ]
        else:
            layers += [
                CausalConv1d(
                    channels // (2 ** (i + 1)),
                    out_channels,
                    kernel_size,
                    bias=bias,
                    pad=pad,
                    pad_params=pad_params,
                ),
            ]
        if use_final_nonlinear_activation:
            layers += [torch.nn.Tanh()]

        self.melgan = torch.nn.Sequential(*layers)

        if use_weight_norm:
            self.apply_weight_norm()

        self.reset_parameters()
        self.pqmf = None

    def forward(self, c):
        """Calculate forward propagation."""
        return self.melgan(c)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m) -> None:
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self) -> None:
        """Reset parameters."""

        def _reset_parameters(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)

    def inference(self, c):
        """Perform inference."""
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        c = self.melgan(c.transpose(1, 0).unsqueeze(0))
        if self.pqmf is not None:
            c = self.pqmf.synthesis(c)
        return c.squeeze(0).transpose(1, 0)


class MelGANDiscriminator(torch.nn.Module):
    """MelGAN discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_sizes=[5, 3],
        channels: int = 16,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        downsample_scales: List[int] = [4, 4, 4, 4],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params={},
    ) -> None:
        """Initilize MelGAN discriminator module."""
        super(MelGANDiscriminator, self).__init__()
        self.layers = torch.nn.ModuleList()

        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1
        assert kernel_sizes[1] % 2 == 1

        self.layers += [
            torch.nn.Sequential(
                getattr(torch.nn, pad)((np.prod(kernel_sizes) - 1) // 2, **pad_params),
                torch.nn.Conv1d(
                    in_channels, channels, np.prod(kernel_sizes), bias=bias
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        in_chs = channels
        for downsample_scale in downsample_scales:
            out_chs = min(in_chs * downsample_scale, max_downsample_channels)
            self.layers += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=downsample_scale * 10 + 1,
                        stride=downsample_scale,
                        padding=downsample_scale * 5,
                        groups=in_chs // 4,
                        bias=bias,
                    ),
                    getattr(torch.nn, nonlinear_activation)(
                        **nonlinear_activation_params
                    ),
                )
            ]
            in_chs = out_chs

        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_sizes[0],
                    padding=(kernel_sizes[0] - 1) // 2,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            torch.nn.Conv1d(
                out_chs,
                out_channels,
                kernel_sizes[1],
                padding=(kernel_sizes[1] - 1) // 2,
                bias=bias,
            ),
        ]

    def forward(self, x):
        """Calculate forward propagation."""
        outs = []
        for f in self.layers:
            x = f(x)
            outs += [x]

        return outs


class MelGANMultiScaleDiscriminator(torch.nn.Module):
    """MelGAN multi-scale discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        scales: int = 3,
        downsample_pooling: str = "AvgPool1d",
        downsample_pooling_params: Dict[str, int] = {
            "kernel_size": 4,
            "stride": 2,
            "padding": 1,
            "count_include_pad": False,
        },
        kernel_sizes: List[int] = [5, 3],
        channels: int = 16,
        max_downsample_channels: int = 1024,
        bias: bool = True,
        downsample_scales: List[int] = [4, 4, 4, 4],
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        pad: str = "ReflectionPad1d",
        pad_params={},
        use_weight_norm: bool = True,
    ) -> None:
        """Initilize MelGAN multi-scale discriminator module."""
        super(MelGANMultiScaleDiscriminator, self).__init__()
        self.discriminators = torch.nn.ModuleList()

        for _ in range(scales):
            self.discriminators += [
                MelGANDiscriminator(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_sizes=kernel_sizes,
                    channels=channels,
                    max_downsample_channels=max_downsample_channels,
                    bias=bias,
                    downsample_scales=downsample_scales,
                    nonlinear_activation=nonlinear_activation,
                    nonlinear_activation_params=nonlinear_activation_params,
                    pad=pad,
                    pad_params=pad_params,
                )
            ]
        self.pooling = getattr(torch.nn, downsample_pooling)(
            **downsample_pooling_params
        )

        if use_weight_norm:
            self.apply_weight_norm()

        self.reset_parameters()

    def forward(self, x):
        """Calculate forward propagation."""
        outs = []
        for f in self.discriminators:
            outs += [f(x)]
            x = self.pooling(x)

        return outs

    def remove_weight_norm(self) -> None:
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m) -> None:
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def reset_parameters(self) -> None:
        """Reset parameters."""

        def _reset_parameters(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(
                m, torch.nn.ConvTranspose1d
            ):
                m.weight.data.normal_(0.0, 0.02)
                logging.debug(f"Reset parameters in {m}.")

        self.apply(_reset_parameters)


class ParallelWaveGANGenerator(torch.nn.Module):
    """Parallel WaveGAN Generator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        aux_channels: int = 80,
        aux_context_window: int = 2,
        dropout: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_causal_conv: bool = False,
        upsample_conditional_features: bool = True,
        upsample_net: str = "ConvInUpsampleNetwork",
        upsample_params: Dict[str, List[int]] = {"upsample_scales": [4, 4, 4, 4]},
    ) -> None:
        """Initialize Parallel WaveGAN Generator module."""
        super(ParallelWaveGANGenerator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aux_channels = aux_channels
        self.aux_context_window = aux_context_window
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = Conv1d1x1(in_channels, residual_channels, bias=True)

        if upsample_conditional_features:
            upsample_params.update(
                {
                    "use_causal_conv": use_causal_conv,
                }
            )
            if upsample_net == "MelGANGenerator":
                assert aux_context_window == 0
                upsample_params.update(
                    {
                        "use_weight_norm": False,
                        "use_final_nonlinear_activation": False,
                    }
                )
                self.upsample_net = MelGANGenerator(**upsample_params)
            else:
                if upsample_net == "ConvInUpsampleNetwork":
                    upsample_params.update(
                        {
                            "aux_channels": aux_channels,
                            "aux_context_window": aux_context_window,
                        }
                    )
                upsample_classes = {
                    "UpsampleNetwork": UpsampleNetwork,
                    "ConvInUpsampleNetwork": ConvInUpsampleNetwork,
                }
                self.upsample_net = upsample_classes[upsample_net](**upsample_params)
            self.upsample_factor = np.prod(upsample_params["upsample_scales"])
        else:
            self.upsample_net = None
            self.upsample_factor = 1

        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=aux_channels,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        self.last_conv_layers = torch.nn.ModuleList(
            [
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                torch.nn.ReLU(inplace=True),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x: float, c) -> float:
        """Calculate forward propagation."""
        if c is not None and self.upsample_net is not None:
            c = self.upsample_net(c)
            assert c.size(-1) == x.size(-1)

        x = self.first_conv(x)
        skips = 0
        for f in self.conv_layers:
            x, h = f(x, c)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        for f in self.last_conv_layers:
            x = f(x)

        return x

    def remove_weight_norm(self) -> None:
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m) -> None:
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self) -> None:
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    @staticmethod
    def _get_receptive_field_size(layers, stacks, kernel_size, dilation=lambda x: 2**x):
        assert layers % stacks == 0
        layers_per_cycle = layers // stacks
        dilations = [dilation(i % layers_per_cycle) for i in range(layers)]
        return (kernel_size - 1) * sum(dilations) + 1

    @property
    def receptive_field_size(self):
        """Return receptive field size."""
        return self._get_receptive_field_size(
            self.layers, self.stacks, self.kernel_size
        )

    def inference(self, c=None, x=None):
        """Perform inference."""
        if x is not None:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            x = x.transpose(1, 0).unsqueeze(0)
        else:
            assert c is not None
            x = torch.randn(1, 1, len(c) * self.upsample_factor).to(
                next(self.parameters()).device
            )
        if c is not None:
            if not isinstance(c, torch.Tensor):
                c = torch.tensor(c, dtype=torch.float).to(
                    next(self.parameters()).device
                )
            c = c.transpose(1, 0).unsqueeze(0)
            c = torch.nn.ReplicationPad1d(self.aux_context_window)(c)
        return self.forward(x, c).squeeze(0).transpose(1, 0)


class ParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 10,
        conv_channels: int = 64,
        dilation_factor: int = 1,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
        bias: bool = True,
        use_weight_norm: bool = True,
    ) -> None:
        """Initialize Parallel WaveGAN Discriminator module."""
        super(ParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = torch.nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor**i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(
                    conv_in_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                getattr(torch.nn, nonlinear_activation)(
                    inplace=True, **nonlinear_activation_params
                ),
            ]
            self.conv_layers += conv_layer
        padding = (kernel_size - 1) // 2
        last_conv_layer = Conv1d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_layers += [last_conv_layer]

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation."""
        for f in self.conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self) -> None:
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m) -> None:
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)


class ResidualParallelWaveGANDiscriminator(torch.nn.Module):
    """Parallel WaveGAN Discriminator module."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        kernel_size: int = 3,
        layers: int = 30,
        stacks: int = 3,
        residual_channels: int = 64,
        gate_channels: int = 128,
        skip_channels: int = 64,
        dropout: float = 0.0,
        bias: bool = True,
        use_weight_norm: bool = True,
        use_causal_conv: bool = False,
        nonlinear_activation: str = "LeakyReLU",
        nonlinear_activation_params: Dict[str, float] = {"negative_slope": 0.2},
    ) -> None:
        """Initialize Parallel WaveGAN Discriminator module."""
        super(ResidualParallelWaveGANDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.layers = layers
        self.stacks = stacks
        self.kernel_size = kernel_size

        assert layers % stacks == 0
        layers_per_stack = layers // stacks

        self.first_conv = torch.nn.Sequential(
            Conv1d1x1(in_channels, residual_channels, bias=True),
            getattr(torch.nn, nonlinear_activation)(
                inplace=True, **nonlinear_activation_params
            ),
        )

        self.conv_layers = torch.nn.ModuleList()
        for layer in range(layers):
            dilation = 2 ** (layer % layers_per_stack)
            conv = ResidualBlock(
                kernel_size=kernel_size,
                residual_channels=residual_channels,
                gate_channels=gate_channels,
                skip_channels=skip_channels,
                aux_channels=-1,
                dilation=dilation,
                dropout=dropout,
                bias=bias,
                use_causal_conv=use_causal_conv,
            )
            self.conv_layers += [conv]

        self.last_conv_layers = torch.nn.ModuleList(
            [
                getattr(torch.nn, nonlinear_activation)(
                    inplace=True, **nonlinear_activation_params
                ),
                Conv1d1x1(skip_channels, skip_channels, bias=True),
                getattr(torch.nn, nonlinear_activation)(
                    inplace=True, **nonlinear_activation_params
                ),
                Conv1d1x1(skip_channels, out_channels, bias=True),
            ]
        )

        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x: float) -> float:
        """Calculate forward propagation."""
        x = self.first_conv(x)

        skips = 0
        for f in self.conv_layers:
            x, h = f(x, None)
            skips += h
        skips *= math.sqrt(1.0 / len(self.conv_layers))

        x = skips
        for f in self.last_conv_layers:
            x = f(x)
        return x

    def apply_weight_norm(self) -> None:
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m) -> None:
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)
                logging.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def remove_weight_norm(self) -> None:
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m) -> None:
            try:
                logging.debug(f"Weight norm is removed from {m}.")
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:
                return

        self.apply(_remove_weight_norm)


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
    print("vocoder-benchmark: Parallel WaveGAN and MelGAN core modules")
    print("=" * 70)

    print("-" * 70)
    print("[Test 1/6] causal convolutions and residual block")
    try:
        causal = CausalConv1d(2, 3, kernel_size=3, dilation=2)
        x_causal = torch.randn(2, 2, 9, requires_grad=True)
        y_causal = causal(x_causal)
        deconv = CausalConvTranspose1d(3, 2, kernel_size=4, stride=2)
        y_deconv = deconv(torch.randn(2, 3, 5))
        rb = ResidualBlock(
            kernel_size=3,
            residual_channels=4,
            gate_channels=8,
            skip_channels=5,
            aux_channels=3,
            dropout=0.0,
            dilation=2,
            use_causal_conv=False,
        )
        x = torch.randn(2, 4, 16, requires_grad=True)
        c = torch.randn(2, 3, 16)
        residual_out, skip_out = rb(x, c)
        check("CausalConv1d preserves time length", y_causal.shape == (2, 3, 9), str(tuple(y_causal.shape)))
        check("CausalConvTranspose1d doubles time with stride two", y_deconv.shape == (2, 2, 10), str(tuple(y_deconv.shape)))
        check("ResidualBlock returns residual output", residual_out.shape == (2, 4, 16), str(tuple(residual_out.shape)))
        check("ResidualBlock returns skip output", skip_out.shape == (2, 5, 16), str(tuple(skip_out.shape)))
        check("ResidualBlock outputs finite tensors", torch.isfinite(residual_out).all().item() and torch.isfinite(skip_out).all().item())
        (residual_out.mean() + skip_out.mean()).backward()
        check("ResidualBlock keeps input gradients", x.grad is not None and torch.isfinite(x.grad).all().item())
    except Exception as exc:
        skip_checks(6, f"causal/residual modules raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 2/6] upsampling networks")
    try:
        upsample = UpsampleNetwork(upsample_scales=[2, 2])
        c = torch.randn(2, 3, 5)
        y = upsample(c)
        conv_up = ConvInUpsampleNetwork(
            upsample_scales=[2, 2],
            aux_channels=3,
            aux_context_window=1,
            use_causal_conv=False,
        )
        y_conv = conv_up(torch.randn(2, 3, 7))
        stretch = Stretch2d(3, 2)
        stretched = stretch(torch.randn(2, 1, 4, 5))
        check("UpsampleNetwork output not None", y is not None)
        if y is not None:
            check("UpsampleNetwork multiplies time by scale product", y.shape == (2, 3, 20), str(tuple(y.shape)))
            check("UpsampleNetwork output finite", torch.isfinite(y).all().item())
        else:
            skip_checks(2, "UpsampleNetwork returned None")
        check("ConvInUpsampleNetwork handles context window", y_conv.shape == (2, 3, 20), str(tuple(y_conv.shape)))
        check("Stretch2d scales frequency and time axes", stretched.shape == (2, 1, 8, 15), str(tuple(stretched.shape)))
    except Exception as exc:
        skip_checks(5, f"upsampling modules raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 3/6] MelGAN generator and residual stack")
    try:
        stack = ResidualStack(kernel_size=3, channels=4, dilation=2)
        stack_out = stack(torch.randn(2, 4, 12))
        gen = MelGANGenerator(
            in_channels=4,
            out_channels=1,
            kernel_size=3,
            channels=16,
            upsample_scales=[2, 2],
            stack_kernel_size=3,
            stacks=1,
            use_weight_norm=False,
            use_final_nonlinear_activation=True,
        )
        mel = gen(torch.randn(2, 4, 5))
        check("ResidualStack preserves channel and time shape", stack_out.shape == (2, 4, 12), str(tuple(stack_out.shape)))
        check("ResidualStack output finite", torch.isfinite(stack_out).all().item())
        check("MelGANGenerator output not None", mel is not None)
        if mel is not None:
            check("MelGANGenerator upsamples time", mel.shape == (2, 1, 20), str(tuple(mel.shape)))
            check("MelGANGenerator tanh bounds output", (mel.abs() <= 1.000001).all().item())
            mel.mean().backward()
            first_weight = next(gen.parameters())
            check("MelGANGenerator supports gradient flow", first_weight.grad is not None and torch.isfinite(first_weight.grad).all().item())
        else:
            skip_checks(3, "MelGANGenerator returned None")
    except Exception as exc:
        skip_checks(6, f"MelGAN generator modules raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 4/6] MelGAN discriminators")
    try:
        disc = MelGANDiscriminator(
            channels=4,
            max_downsample_channels=16,
            downsample_scales=[2, 2],
        )
        x = torch.randn(2, 1, 64)
        disc_outs = disc(x)
        multi = MelGANMultiScaleDiscriminator(
            scales=2,
            channels=4,
            max_downsample_channels=16,
            downsample_scales=[2],
            use_weight_norm=False,
        )
        multi_outs = multi(x)
        check("MelGANDiscriminator returns per-layer outputs", isinstance(disc_outs, list) and len(disc_outs) == 5, str(len(disc_outs) if isinstance(disc_outs, list) else type(disc_outs)))
        if isinstance(disc_outs, list) and len(disc_outs) > 0:
            check("MelGANDiscriminator final channel count", disc_outs[-1].shape[1] == 1, str(tuple(disc_outs[-1].shape)))
            check("MelGANDiscriminator outputs finite", all(torch.isfinite(t).all().item() for t in disc_outs))
        else:
            skip_checks(2, "MelGANDiscriminator returned no layer outputs")
        check("MelGANMultiScaleDiscriminator returns one list per scale", isinstance(multi_outs, list) and len(multi_outs) == 2, str(len(multi_outs) if isinstance(multi_outs, list) else type(multi_outs)))
        if isinstance(multi_outs, list) and len(multi_outs) == 2:
            check("MelGANMultiScaleDiscriminator nests layer outputs", all(isinstance(scale_out, list) for scale_out in multi_outs))
            check("MelGANMultiScaleDiscriminator downsamples between scales", multi_outs[1][0].shape[-1] <= multi_outs[0][0].shape[-1], f"{multi_outs[0][0].shape[-1]}->{multi_outs[1][0].shape[-1]}")
        else:
            skip_checks(2, "MelGANMultiScaleDiscriminator returned invalid output")
    except Exception as exc:
        skip_checks(7, f"MelGAN discriminator modules raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 5/6] ParallelWaveGAN generator")
    try:
        gen = ParallelWaveGANGenerator(
            layers=4,
            stacks=2,
            residual_channels=4,
            gate_channels=8,
            skip_channels=4,
            aux_channels=3,
            dropout=0.0,
            use_weight_norm=False,
            upsample_conditional_features=True,
            upsample_params={"upsample_scales": [2, 2]},
            aux_context_window=1,
        )
        x = torch.randn(2, 1, 20)
        c = torch.randn(2, 3, 7)
        y = gen(x, c)
        check("ParallelWaveGANGenerator output not None", y is not None)
        if y is not None:
            check("ParallelWaveGANGenerator preserves waveform time", y.shape == (2, 1, 20), str(tuple(y.shape)))
            check("ParallelWaveGANGenerator output finite", torch.isfinite(y).all().item())
            y.mean().backward()
            check("ParallelWaveGANGenerator keeps first conv gradient", gen.first_conv.weight.grad is not None and torch.isfinite(gen.first_conv.weight.grad).all().item())
        else:
            skip_checks(3, "ParallelWaveGANGenerator returned None")
        check("ParallelWaveGANGenerator tracks upsample factor", gen.upsample_factor == 4, str(gen.upsample_factor))
        check("ParallelWaveGANGenerator receptive field formula", gen.receptive_field_size == 13, str(gen.receptive_field_size))
    except Exception as exc:
        skip_checks(6, f"ParallelWaveGANGenerator raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 6/6] ParallelWaveGAN discriminators")
    try:
        disc = ParallelWaveGANDiscriminator(
            layers=3,
            conv_channels=4,
            use_weight_norm=False,
        )
        x = torch.randn(2, 1, 16)
        y = disc(x)
        residual_disc = ResidualParallelWaveGANDiscriminator(
            layers=2,
            stacks=1,
            residual_channels=4,
            gate_channels=8,
            skip_channels=4,
            use_weight_norm=False,
        )
        ry = residual_disc(torch.randn(2, 1, 16))
        check("ParallelWaveGANDiscriminator output not None", y is not None)
        if y is not None:
            check("ParallelWaveGANDiscriminator preserves waveform time", y.shape == (2, 1, 16), str(tuple(y.shape)))
            check("ParallelWaveGANDiscriminator output finite", torch.isfinite(y).all().item())
        else:
            skip_checks(2, "ParallelWaveGANDiscriminator returned None")
        check("ResidualParallelWaveGANDiscriminator output not None", ry is not None)
        if ry is not None:
            check("ResidualParallelWaveGANDiscriminator output shape", ry.shape == (2, 1, 16), str(tuple(ry.shape)))
            check("ResidualParallelWaveGANDiscriminator output finite", torch.isfinite(ry).all().item())
        else:
            skip_checks(2, "ResidualParallelWaveGANDiscriminator returned None")
    except Exception as exc:
        skip_checks(6, f"ParallelWaveGAN discriminator modules raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
