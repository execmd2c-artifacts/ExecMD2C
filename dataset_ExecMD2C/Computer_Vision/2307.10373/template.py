# ============================================================
# ground_truth_ablated.py - TokenFlow Core Model Components (Source-faithful Ablated)
# Source: 2307.10373/TokenFlow-master
#
# Contains ONLY the model architecture definitions and direct dependencies.
# Model code above __main__ is copied from the source repository with only
# one-file consolidation import adjustments.
# No training, inference pipeline, dataset, video I/O, or sampling loop code.
# ============================================================
#
# Benchmark contract:
# - This file is the ablated test template paired with `ground_truth.py`.
# - Only core algorithm bodies are replaced by TODO docstrings and `pass`.
# - All imports, class definitions, `__init__` methods, simple utilities, and the
#   complete `__main__` grading suite must remain intact.
# - The paired files must share the exact same `__main__` test suite.
# - Tests use the standard three-layer rubric:
#   1. functionality: output is not None and no crash;
#   2. basic correctness: output shape and finite values;
#   3. core semantic constraints specific to the ablated mechanism.
# - The tests are behavioral grading checks, not reference-parity tests.
# - Running this file before completing TODOs is expected to fail.
# ============================================================

# --- Third-party imports ---
from typing import Type
import os
import torch
import torch.nn as nn


# --- [Original file: util.py] ---
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def batch_cosine_sim(x, y):
    if type(x) is list:
        x = torch.cat(x, dim=0)
    if type(y) is list:
        y = torch.cat(y, dim=0)
    x = x / x.norm(dim=-1, keepdim=True)
    y = y / y.norm(dim=-1, keepdim=True)
    similarity = x @ y.T
    return similarity


# --- [Original file: tokenflow_utils.py] ---
def register_pivotal(diffusion_model, is_pivotal):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "pivotal_pass", is_pivotal)


def register_batch_idx(diffusion_model, batch_idx):
    for _, module in diffusion_model.named_modules():
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module, "BasicTransformerBlock"):
            setattr(module, "batch_idx", batch_idx)


def register_time(model, t):
    conv_module = model.unet.up_blocks[1].resnets[1]
    setattr(conv_module, 't', t)
    down_res_dict = {0: [0, 1], 1: [0, 1], 2: [0, 1]}
    up_res_dict = {1: [0, 1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    for res in up_res_dict:
        for block in up_res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    for res in down_res_dict:
        for block in down_res_dict[res]:
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn1
            setattr(module, 't', t)
            module = model.unet.down_blocks[res].attentions[block].transformer_blocks[0].attn2
            setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn1
    setattr(module, 't', t)
    module = model.unet.mid_block.attentions[0].transformer_blocks[0].attn2
    setattr(module, 't', t)


def load_source_latents_t(t, latents_path):
    latents_t_path = os.path.join(latents_path, f'noisy_latents_{t}.pt')
    assert os.path.exists(latents_t_path), f'Missing latents at t {t} path {latents_t_path}'
    latents = torch.load(latents_t_path)
    return latents


def register_conv_injection(model, injection_schedule):
    def conv_forward(self):
        def forward(input_tensor, temb):
            """
            [TODO] Run the injected ResNet convolution path used by TokenFlow PnP.

            Input:
                input_tensor: (batch, channels, height, width) - latent feature map for source,
                    unconditional, and conditional branches concatenated along batch.
                temb: (batch, time_channels) or None - diffusion timestep embedding.

            Output: (batch, out_channels, out_height, out_width) - residual block output with
                source features optionally copied into edited branches.

"""
            pass

        return forward

    conv_module = model.unet.up_blocks[1].resnets[1]
    conv_module.forward = conv_forward(conv_module)
    setattr(conv_module, 'injection_schedule', injection_schedule)


def register_extended_attention_pnp(model, injection_schedule):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            """
            [TODO] Compute PnP TokenFlow extended self-attention with scheduled query/key injection.

            Input:
                x: (3 * n_frames, sequence_length, dim) - hidden tokens for source,
                    unconditional edit, and conditional edit branches.
                encoder_hidden_states: (3 * n_frames, context_length, dim) or None - optional context.
                attention_mask: optional attention mask accepted for API compatibility.

            Output: (3 * n_frames, sequence_length, dim) - attention output after every edited
                frame attends to tokens propagated from the full video.

"""
            pass

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)
            setattr(module.attn1, 'injection_schedule', [])

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def register_extended_attention(model):
    def sa_forward(self):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, encoder_hidden_states=None, attention_mask=None):
            """
            [TODO] Compute TokenFlow extended self-attention across video frames.

            Input:
                x: (3 * n_frames, sequence_length, dim) - hidden tokens for source,
                    unconditional edit, and conditional edit branches.
                encoder_hidden_states: (3 * n_frames, context_length, dim) or None - optional context.
                attention_mask: optional attention mask accepted for API compatibility.

            Output: (3 * n_frames, sequence_length, dim) - attention output with edited
                branches attending to tokens gathered from all frames.

"""
            pass

        return forward

    for _, module in model.unet.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            module.attn1.forward = sa_forward(module.attn1)

    res_dict = {1: [1, 2], 2: [0, 1, 2], 3: [0, 1, 2]}
    # we are injecting attention in blocks 4 - 11 of the decoder, so not in the first block of the lowest resolution
    for res in res_dict:
        for block in res_dict[res]:
            module = model.unet.up_blocks[res].attentions[block].transformer_blocks[0].attn1
            module.forward = sa_forward(module)


def make_tokenflow_attention_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:

    class TokenFlowBlock(block_class):

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            """
            [TODO] Run a TokenFlow transformer block with pivotal-token propagation.

            Input:
                hidden_states: (3 * n_frames, sequence_length, dim) - hidden tokens for source,
                    unconditional edit, and conditional edit branches.
                attention_mask: optional self-attention mask accepted for compatibility.
                encoder_hidden_states: (3 * n_frames, context_length, dim) or None - cross-attention context.
                encoder_attention_mask: optional cross-attention mask.
                timestep: diffusion timestep or adaptive normalization timestep input.
                cross_attention_kwargs: optional keyword arguments passed to attention modules.
                class_labels: optional labels used by adaptive zero normalization.

            Output: (3 * n_frames, sequence_length, dim) - transformer block output after TokenFlow
                propagation, optional cross-attention, and feed-forward residual updates.

"""
            pass

    return TokenFlowBlock


def set_tokenflow(
        model: torch.nn.Module):
    """
    Sets the tokenflow attention blocks in a model.
    """

    for _, module in model.named_modules():
        if isinstance_str(module, "BasicTransformerBlock"):
            make_tokenflow_block_fn = make_tokenflow_attention_block
            module.__class__ = make_tokenflow_block_fn(module.__class__)

            # Something needed for older versions of diffusers
            if not hasattr(module, "use_ada_layer_norm_zero"):
                module.use_ada_layer_norm = False
                module.use_ada_layer_norm_zero = False

    return model


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

    def init_identity_linear(linear):
        with torch.no_grad():
            linear.weight.zero_()
            for i in range(min(linear.weight.shape)):
                linear.weight[i, i] = 1.0
            if linear.bias is not None:
                linear.bias.zero_()

    def init_identity_conv(conv):
        with torch.no_grad():
            conv.weight.zero_()
            for i in range(min(conv.out_channels, conv.in_channels)):
                conv.weight[i, i, 0, 0] = 1.0
            if conv.bias is not None:
                conv.bias.zero_()

    class SimpleSelfAttention(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.proj = nn.Linear(dim, dim)

        def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            return self.proj(hidden_states)

    class MockTokenAttention(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.Linear(dim, dim, bias=False)
            init_identity_linear(self.to_q)
            init_identity_linear(self.to_k)
            init_identity_linear(self.to_v)
            init_identity_linear(self.to_out)

        def head_to_batch_dim(self, tensor):
            batch, sequence_length, dim = tensor.shape
            head_dim = dim // self.heads
            tensor = tensor.view(batch, sequence_length, self.heads, head_dim)
            return tensor.permute(0, 2, 1, 3).reshape(batch * self.heads, sequence_length, head_dim)

        def batch_to_head_dim(self, tensor):
            batch_heads, sequence_length, head_dim = tensor.shape
            batch = batch_heads // self.heads
            tensor = tensor.view(batch, self.heads, sequence_length, head_dim)
            return tensor.permute(0, 2, 1, 3).reshape(batch, sequence_length, self.heads * head_dim)

        def forward(self, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
            return self.to_out(hidden_states)

    class BasicTransformerBlock(nn.Module):
        def __init__(self, dim, attention=None):
            super().__init__()
            self.only_cross_attention = False
            self.use_ada_layer_norm = False
            self.use_ada_layer_norm_zero = False
            self.norm1 = nn.LayerNorm(dim)
            self.attn1 = attention if attention is not None else SimpleSelfAttention(dim)
            self.attn2 = None
            self.norm3 = nn.LayerNorm(dim)
            self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    class MockResnet(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm1 = nn.Identity()
            self.nonlinearity = nn.Identity()
            self.upsample = None
            self.downsample = None
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.time_emb_proj = nn.Linear(channels, channels, bias=False)
            self.time_embedding_norm = "default"
            self.norm2 = nn.Identity()
            self.dropout = nn.Identity()
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
            self.conv_shortcut = None
            self.output_scale_factor = 1.0
            self.t = None
            init_identity_conv(self.conv1)
            init_identity_conv(self.conv2)

    class AttentionWrapper(nn.Module):
        def __init__(self, dim, heads):
            super().__init__()
            self.transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(dim, attention=MockTokenAttention(dim, heads))
            ])

    class UpBlock(nn.Module):
        def __init__(self, dim, heads, channels):
            super().__init__()
            self.resnets = nn.ModuleList([MockResnet(channels) for _ in range(3)])
            self.attentions = nn.ModuleList([AttentionWrapper(dim, heads) for _ in range(3)])

    class MockUNet(nn.Module):
        def __init__(self, dim, heads, channels):
            super().__init__()
            self.up_blocks = nn.ModuleList([UpBlock(dim, heads, channels) for _ in range(4)])

    class MockDiffusionModel(nn.Module):
        def __init__(self, dim=8, heads=2, channels=4):
            super().__init__()
            self.unet = MockUNet(dim, heads, channels)

    class ToyUNet(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.block = BasicTransformerBlock(dim)

        def forward(self, hidden_states):
            return self.block(hidden_states)

    print("=" * 70)
    print("TokenFlow Core Model Components")
    print("Automated Test Suite - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: register_conv_injection inner forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] register_conv_injection - scheduled conv feature injection")
    try:
        model = MockDiffusionModel().to(device)
        register_conv_injection(model, injection_schedule=[7])
        resnet = model.unet.up_blocks[1].resnets[1]
        resnet.t = 7
        x = torch.randn(6, 4, 3, 3, device=device)
        y = resnet(x, None)
        check("conv output not None", y is not None)
        if y is not None:
            check("conv output shape", tuple(y.shape) == (6, 4, 3, 3), f"got {tuple(y.shape)}")
            check("conv output finite", torch.isfinite(y).all().item())
            source_residual = y[:2] - x[:2]
            uncond_residual = y[2:4] - x[2:4]
            cond_residual = y[4:6] - x[4:6]
            injected = torch.allclose(uncond_residual, source_residual, atol=1e-6) and torch.allclose(
                cond_residual, source_residual, atol=1e-6
            )
            check("conv copies source residual into edited branches", injected)
        else:
            skip_checks(3, "conv output is None")
    except Exception as exc:
        skip_checks(4, f"conv injection raised {type(exc).__name__}: {exc}")
    print()

    # ==============================================================
    # Test 2/4: register_extended_attention_pnp inner forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] register_extended_attention_pnp - scheduled q/k injection")
    try:
        model = MockDiffusionModel().to(device)
        register_extended_attention_pnp(model, injection_schedule=[5])
        attn = model.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1
        x = torch.randn(6, 4, 8, device=device)
        attn.t = 5
        y = attn(x)
        attn.t = 4
        y_unscheduled = attn(x)
        check("pnp attention output not None", y is not None)
        if y is not None:
            check("pnp attention output shape", tuple(y.shape) == (6, 4, 8), f"got {tuple(y.shape)}")
            check("pnp attention output finite", torch.isfinite(y).all().item())
            if y_unscheduled is not None:
                changed = not torch.allclose(y[2:], y_unscheduled[2:], atol=1e-6)
                check("pnp scheduled injection changes edited branches", changed)
            else:
                check("pnp unscheduled output available", False, "unscheduled output is None")
        else:
            skip_checks(3, "pnp attention output is None")
    except Exception as exc:
        skip_checks(4, f"pnp attention raised {type(exc).__name__}: {exc}")
    print()

    # ==============================================================
    # Test 3/4: register_extended_attention inner forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] register_extended_attention - all-frame edited attention")
    try:
        model = MockDiffusionModel().to(device)
        register_extended_attention(model)
        attn = model.unet.up_blocks[1].attentions[1].transformer_blocks[0].attn1
        x = torch.randn(6, 4, 8, device=device)
        y = attn(x)
        x_perturbed = x.clone()
        x_perturbed[3] = x_perturbed[3] + 4.0
        y_perturbed = attn(x_perturbed)
        check("extended attention output not None", y is not None)
        if y is not None:
            check("extended attention output shape", tuple(y.shape) == (6, 4, 8), f"got {tuple(y.shape)}")
            check("extended attention output finite", torch.isfinite(y).all().item())
            if y_perturbed is not None:
                source_stable = torch.allclose(y[0], y_perturbed[0], atol=1e-5)
                edited_cross_frame = not torch.allclose(y[2], y_perturbed[2], atol=1e-5)
                check("source branch remains frame-local", source_stable)
                check("edited branch attends across frames", edited_cross_frame)
            else:
                skip_checks(2, "perturbed attention output is None")
        else:
            skip_checks(4, "extended attention output is None")
    except Exception as exc:
        skip_checks(5, f"extended attention raised {type(exc).__name__}: {exc}")
    print()

    # ==============================================================
    # Test 4/4: TokenFlowBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] TokenFlowBlock.forward - pivotal and propagation passes")
    try:
        model = ToyUNet(dim=8).to(device)
        set_tokenflow(model)
        hidden_states = torch.randn(6, 5, 8, device=device)
        register_pivotal(model, True)
        pivotal_output = model(hidden_states)
        check("pivotal output not None", pivotal_output is not None)
        if pivotal_output is not None:
            check("pivotal output shape", tuple(pivotal_output.shape) == (6, 5, 8), f"got {tuple(pivotal_output.shape)}")
            check("pivotal output finite", torch.isfinite(pivotal_output).all().item())
            check("pivotal cache created", hasattr(model.block, "pivot_hidden_states") and hasattr(model.block, "kf_attn_output"))
        else:
            skip_checks(3, "pivotal output is None")

        register_pivotal(model, False)
        register_batch_idx(model, 0)
        propagated_output = model(hidden_states)
        check("propagated output not None", propagated_output is not None)
        if propagated_output is not None:
            check(
                "propagated output shape",
                tuple(propagated_output.shape) == (6, 5, 8),
                f"got {tuple(propagated_output.shape)}",
            )
            check("propagated output finite", torch.isfinite(propagated_output).all().item())
        else:
            skip_checks(2, "propagated output is None")
    except Exception as exc:
        skip_checks(7, f"TokenFlowBlock raised {type(exc).__name__}: {exc}")
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
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
