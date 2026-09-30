# Source-faithful benchmark extraction for control-a-video-main.
# Model code above __main__ is copied from model/video_diffusion/models/resnet.py
# and model/video_diffusion/models/unet_3d_blocks_control.py with only one-file
# import consolidation edits.

from typing import List, Tuple

import torch
from torch import nn
import torch.nn as nn
from einops import rearrange


# Copied from model/video_diffusion/models/resnet.py
class PseudoConv3d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, temporal_kernel_size=None, **kwargs):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            **kwargs,
        )
        if temporal_kernel_size is None:
            temporal_kernel_size = kernel_size

        self.conv_temporal = (
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=temporal_kernel_size,
                padding=temporal_kernel_size // 2,
            )
            if kernel_size > 1
            else None
        )

        if self.conv_temporal is not None:
            nn.init.dirac_(self.conv_temporal.weight.data)  # initialized to be identity
            nn.init.zeros_(self.conv_temporal.bias.data)

    def forward(self, x):
        """
        TODO: Reproduce the pseudo-3D convolution forward pass.

        Input:
            x: either an image tensor with shape (B, C, H, W) or a video tensor with
            shape (B, C, F, H, W).

        Output:
            Tensor with the same rank as the input. Image output shape is produced by the
            inherited 2D convolution; video output shape must preserve the frame dimension.

        Required behavior:
            Detect whether the input is video, fold frames into the batch before applying the
            inherited spatial Conv2d, then restore the video layout. If a temporal convolution
            exists and the input is video, apply it along the frame axis independently at each
            spatial location, then restore the original video layout. For image inputs or disabled
            temporal convolution, return the spatial convolution output directly. Preserve gradient
            flow through both spatial and temporal convolution parameters.
        """
        pass


# Copied from model/video_diffusion/models/unet_3d_blocks_control.py
def set_zero_parameters(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


def zero_conv(channels):
    return set_zero_parameters(PseudoConv3d(channels, channels, 1, padding=0))


class ControlNetInputHintBlock(nn.Module):
    def __init__(self, hint_channels: int = 3, channels: int = 320):
        super().__init__()
        #  Layer configurations are from reference implementation.
        self.input_hint_block = nn.Sequential(
            PseudoConv3d(hint_channels, 16, 3, padding=1),
            nn.SiLU(),
            PseudoConv3d(16, 16, 3, padding=1),
            nn.SiLU(),
            PseudoConv3d(16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            PseudoConv3d(32, 32, 3, padding=1),
            nn.SiLU(),
            PseudoConv3d(32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            PseudoConv3d(96, 96, 3, padding=1),
            nn.SiLU(),
            PseudoConv3d(96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            set_zero_parameters(PseudoConv3d(256, channels, 3, padding=1)),
        )
    def forward(self, hint: torch.Tensor):
        """
        TODO: Reproduce the ControlNet input hint embedding forward pass.

        Input:
            hint: conditioning video/image tensor with shape (B, hint_channels, F, H, W) or
            a compatible image tensor accepted by the block.

        Output:
            Embedded conditioning tensor with channel count equal to the configured output channels
            and spatial resolution reduced by the strided pseudo-3D convolutions.

        Required behavior:
            Pass the hint tensor through the existing sequential hint embedding stack exactly once.
            The final zero-initialized pseudo-3D convolution must keep the initial output near zero
            while preserving tensor shape semantics and differentiability through the full stack.
        """
        pass


class ControlNetPseudoZeroConv3dBlock(nn.Module):
    def __init__(
        self,
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "CrossAttnDownBlockPseudo3D",
            "DownBlockPseudo3D",
        ),
        layers_per_block: int = 2,
    ):
        super().__init__()
        self.input_zero_conv = zero_conv(block_out_channels[0])
        zero_convs = []
        for i, down_block_type in enumerate(down_block_types):
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            for _ in range(layers_per_block):
                zero_convs.append(zero_conv(output_channel))
            if not is_final_block:
                zero_convs.append(zero_conv(output_channel))
        self.zero_convs = nn.ModuleList(zero_convs)
        self.mid_zero_conv = zero_conv(block_out_channels[-1])

    def forward(
        self,
        down_block_res_samples: List[torch.Tensor],
        mid_block_sample: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        TODO: Reproduce the ControlNet zero-convolution residual projection block.

        Inputs:
            down_block_res_samples: ordered residual tensors from the down path. The first tensor
            is projected by the dedicated input zero-conv and the remaining tensors are projected
            by the per-block zero-conv list.
            mid_block_sample: residual tensor from the mid block.

        Output:
            List of projected residual tensors preserving the original residual order, followed by
            the projected mid-block tensor.

        Required behavior:
            Apply the dedicated input zero-conv to the first down residual, apply each configured
            zero-conv to the remaining down residuals in order, append the zero-conv projection of
            the mid-block residual, and return all projections as a list. Initial outputs should be
            zero-valued because the projection parameters are zero-initialized, while shapes must
            match their corresponding inputs.
        """
        pass


if __name__ == "__main__":
    import traceback

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

    print("Running control-a-video benchmark checks...")
    torch.manual_seed(13)

    try:
        conv = PseudoConv3d(3, 4, 3, padding=1)
        image = torch.randn(2, 3, 8, 8, requires_grad=True)
        video = torch.randn(2, 3, 3, 8, 8, requires_grad=True)
        image_out = conv(image)
        video_out = conv(video)
        check("pseudo_conv_functionality", image_out is not None and video_out is not None)
        if image_out is None or video_out is None:
            skip_checks(3, "PseudoConv3d returned None")
        else:
            check("pseudo_conv_shapes", tuple(image_out.shape) == (2, 4, 8, 8) and tuple(video_out.shape) == (2, 4, 3, 8, 8),
                  f"image={tuple(image_out.shape)}, video={tuple(video_out.shape)}")
            check("pseudo_conv_finite", torch.isfinite(image_out).all().item() and torch.isfinite(video_out).all().item())
            (image_out.sum() + video_out.sum()).backward()
            grad_ok = image.grad is not None and video.grad is not None and image.grad.abs().sum().item() > 0 and video.grad.abs().sum().item() > 0
            check("pseudo_conv_gradient", grad_ok)
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"PseudoConv3d checks raised {type(exc).__name__}: {exc}")

    try:
        hint_block = ControlNetInputHintBlock(hint_channels=3, channels=8)
        hint = torch.randn(1, 3, 2, 32, 32, requires_grad=True)
        hint_out = hint_block(hint)
        check("hint_block_functionality", hint_out is not None)
        if hint_out is None:
            skip_checks(3, "ControlNetInputHintBlock returned None")
        else:
            check("hint_block_shape", tuple(hint_out.shape) == (1, 8, 2, 4, 4), str(tuple(hint_out.shape)))
            check("hint_block_finite", torch.isfinite(hint_out).all().item())
            check("hint_block_zero_initialized", torch.allclose(hint_out, torch.zeros_like(hint_out), atol=1e-6))
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"ControlNetInputHintBlock checks raised {type(exc).__name__}: {exc}")

    try:
        zero_block = ControlNetPseudoZeroConv3dBlock(
            block_out_channels=(4, 8),
            down_block_types=("CrossAttnDownBlockPseudo3D", "DownBlockPseudo3D"),
            layers_per_block=1,
        )
        down_samples = [
            torch.randn(1, 4, 2, 8, 8, requires_grad=True),
            torch.randn(1, 4, 2, 8, 8, requires_grad=True),
            torch.randn(1, 4, 2, 8, 8, requires_grad=True),
            torch.randn(1, 8, 2, 4, 4, requires_grad=True),
        ]
        mid_sample = torch.randn(1, 8, 2, 4, 4, requires_grad=True)
        outputs = zero_block(down_samples, mid_sample)
        check("zero_block_functionality", outputs is not None)
        if outputs is None:
            skip_checks(3, "ControlNetPseudoZeroConv3dBlock returned None")
        else:
            expected_shapes = [tuple(t.shape) for t in down_samples] + [tuple(mid_sample.shape)]
            output_shapes = [tuple(t.shape) for t in outputs]
            check("zero_block_shapes", output_shapes == expected_shapes, str(output_shapes))
            check("zero_block_finite", all(torch.isfinite(t).all().item() for t in outputs))
            check("zero_block_zero_initialized", all(torch.allclose(t, torch.zeros_like(t), atol=1e-6) for t in outputs))
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"ControlNetPseudoZeroConv3dBlock checks raised {type(exc).__name__}: {exc}")

    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    if failed != 0:
        raise SystemExit(1)
