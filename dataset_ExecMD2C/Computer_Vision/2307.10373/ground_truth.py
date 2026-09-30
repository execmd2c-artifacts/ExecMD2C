# ============================================================
# ground_truth.py - TokenFlow Core Model Components (Source-faithful)
# Source: 2307.10373/TokenFlow-master
#
# Contains ONLY the model architecture definitions and direct dependencies.
# Model code above __main__ is copied from the source repository with only
# one-file consolidation import adjustments.
# No training, inference pipeline, dataset, video I/O, or sampling loop code.
# ============================================================
#
# Benchmark contract:
# - This file is the answer key: it keeps the original implementations.
# - The paired ablated file is `ground_truth_ablated.py`.
# - The paired files must share the exact same `__main__` test suite.
# - Tests use the standard three-layer rubric:
#   1. functionality: output is not None and no crash;
#   2. basic correctness: output shape and finite values;
#   3. core semantic constraints specific to the ablated mechanism.
# - The tests are behavioral grading checks, not reference-parity tests.
# - Do not add training, dataset, pipeline, or downloaded checkpoint requirements.
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
            hidden_states = input_tensor

            hidden_states = self.norm1(hidden_states)
            hidden_states = self.nonlinearity(hidden_states)

            if self.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = self.upsample(input_tensor)
                hidden_states = self.upsample(hidden_states)
            elif self.downsample is not None:
                input_tensor = self.downsample(input_tensor)
                hidden_states = self.downsample(hidden_states)

            hidden_states = self.conv1(hidden_states)

            if temb is not None:
                temb = self.time_emb_proj(self.nonlinearity(temb))[:, :, None, None]

            if temb is not None and self.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = self.norm2(hidden_states)

            if temb is not None and self.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = self.nonlinearity(hidden_states)

            hidden_states = self.dropout(hidden_states)
            hidden_states = self.conv2(hidden_states)
            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                source_batch_size = int(hidden_states.shape[0] // 3)
                # inject unconditional
                hidden_states[source_batch_size:2 * source_batch_size] = hidden_states[:source_batch_size]
                # inject conditional
                hidden_states[2 * source_batch_size:] = hidden_states[:source_batch_size]

            if self.conv_shortcut is not None:
                input_tensor = self.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

            return output_tensor

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
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            if self.injection_schedule is not None and (self.t in self.injection_schedule or self.t == 1000):
                # inject unconditional
                q[n_frames:2 * n_frames] = q[:n_frames]
                k[n_frames:2 * n_frames] = k[:n_frames]
                # inject conditional
                q[2 * n_frames:] = q[:n_frames]
                k[2 * n_frames:] = k[:n_frames]

            k_source = k[:n_frames]
            k_uncond = k[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            v_source = v[:n_frames]
            v_uncond = v[n_frames:2 * n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2 * n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames:2 * n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)


            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)
            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

            out_source_all = []
            out_uncond_all = []
            out_cond_all = []

            single_batch = n_frames<=12
            b = n_frames if single_batch else 1

            for frame in range(0, n_frames, b):
                out_source = []
                out_uncond = []
                out_cond = []
                for j in range(h):
                    sim_source_b = torch.bmm(q_src[frame: frame+ b, j], k_src[frame: frame+ b, j].transpose(-1, -2)) * self.scale
                    sim_uncond_b = torch.bmm(q_uncond[frame: frame+ b, j], k_uncond[frame: frame+ b, j].transpose(-1, -2)) * self.scale
                    sim_cond = torch.bmm(q_cond[frame: frame+ b, j], k_cond[frame: frame+ b, j].transpose(-1, -2)) * self.scale

                    out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[frame: frame+ b, j]))
                    out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[frame: frame+ b, j]))
                    out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[frame: frame+ b, j]))

                out_source = torch.cat(out_source, dim=0)
                out_uncond = torch.cat(out_uncond, dim=0)
                out_cond = torch.cat(out_cond, dim=0)
                if single_batch:
                    out_source = out_source.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_uncond = out_uncond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                    out_cond = out_cond.view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
                out_source_all.append(out_source)
                out_uncond_all.append(out_uncond)
                out_cond_all.append(out_cond)

            out_source = torch.cat(out_source_all, dim=0)
            out_uncond = torch.cat(out_uncond_all, dim=0)
            out_cond = torch.cat(out_cond_all, dim=0)

            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

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
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            n_frames = batch_size // 3
            is_cross = encoder_hidden_states is not None
            encoder_hidden_states = encoder_hidden_states if is_cross else x
            q = self.to_q(x)
            k = self.to_k(encoder_hidden_states)
            v = self.to_v(encoder_hidden_states)

            k_source = k[:n_frames]
            k_uncond = k[n_frames: 2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            k_cond = k[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_source = v[:n_frames]
            v_uncond = v[n_frames:2*n_frames].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)
            v_cond = v[2*n_frames:].reshape(1, n_frames * sequence_length, -1).repeat(n_frames, 1, 1)

            q_source = self.head_to_batch_dim(q[:n_frames])
            q_uncond = self.head_to_batch_dim(q[n_frames: 2*n_frames])
            q_cond = self.head_to_batch_dim(q[2 * n_frames:])
            k_source = self.head_to_batch_dim(k_source)
            k_uncond = self.head_to_batch_dim(k_uncond)
            k_cond = self.head_to_batch_dim(k_cond)
            v_source = self.head_to_batch_dim(v_source)
            v_uncond = self.head_to_batch_dim(v_uncond)
            v_cond = self.head_to_batch_dim(v_cond)

            out_source = []
            out_uncond = []
            out_cond = []

            q_src = q_source.view(n_frames, h, sequence_length, dim // h)
            k_src = k_source.view(n_frames, h, sequence_length, dim // h)
            v_src = v_source.view(n_frames, h, sequence_length, dim // h)
            q_uncond = q_uncond.view(n_frames, h, sequence_length, dim // h)
            k_uncond = k_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_uncond = v_uncond.view(n_frames, h, sequence_length * n_frames, dim // h)
            q_cond = q_cond.view(n_frames, h, sequence_length, dim // h)
            k_cond = k_cond.view(n_frames, h, sequence_length * n_frames, dim // h)
            v_cond = v_cond.view(n_frames, h, sequence_length * n_frames, dim // h)

            for j in range(h):
                sim_source_b = torch.bmm(q_src[:, j], k_src[:, j].transpose(-1, -2)) * self.scale
                sim_uncond_b = torch.bmm(q_uncond[:, j], k_uncond[:, j].transpose(-1, -2)) * self.scale
                sim_cond = torch.bmm(q_cond[:, j], k_cond[:, j].transpose(-1, -2)) * self.scale

                out_source.append(torch.bmm(sim_source_b.softmax(dim=-1), v_src[:, j]))
                out_uncond.append(torch.bmm(sim_uncond_b.softmax(dim=-1), v_uncond[:, j]))
                out_cond.append(torch.bmm(sim_cond.softmax(dim=-1), v_cond[:, j]))

            out_source = torch.cat(out_source, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_uncond = torch.cat(out_uncond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)
            out_cond = torch.cat(out_cond, dim=0).view(h, n_frames,sequence_length, dim // h).permute(1, 0, 2, 3).reshape(h * n_frames, sequence_length, -1)

            out = torch.cat([out_source, out_uncond, out_cond], dim=0)
            out = self.batch_to_head_dim(out)

            return to_out(out)

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

            batch_size, sequence_length, dim = hidden_states.shape
            n_frames = batch_size // 3
            mid_idx = n_frames // 2
            hidden_states = hidden_states.view(3, n_frames, sequence_length, dim)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            norm_hidden_states = norm_hidden_states.view(3, n_frames, sequence_length, dim)
            if self.pivotal_pass:
                self.pivot_hidden_states = norm_hidden_states
            else:
                idx1 = []
                idx2 = []
                batch_idxs = [self.batch_idx]
                if self.batch_idx > 0:
                    batch_idxs.append(self.batch_idx - 1)

                sim = batch_cosine_sim(norm_hidden_states[0].reshape(-1, dim),
                                        self.pivot_hidden_states[0][batch_idxs].reshape(-1, dim))
                if len(batch_idxs) == 2:
                    sim1, sim2 = sim.chunk(2, dim=1)
                    # sim: n_frames * seq_len, len(batch_idxs) * seq_len
                    idx1.append(sim1.argmax(dim=-1))  # n_frames * seq_len
                    idx2.append(sim2.argmax(dim=-1))  # n_frames * seq_len
                else:
                    idx1.append(sim.argmax(dim=-1))
                idx1 = torch.stack(idx1 * 3, dim=0) # 3, n_frames * seq_len
                idx1 = idx1.squeeze(1)
                if len(batch_idxs) == 2:
                    idx2 = torch.stack(idx2 * 3, dim=0) # 3, n_frames * seq_len
                    idx2 = idx2.squeeze(1)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            if self.pivotal_pass:
                # norm_hidden_states.shape = 3, n_frames * seq_len, dim
                self.attn_output = self.attn1(
                        norm_hidden_states.view(batch_size, sequence_length, dim),
                        encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                        **cross_attention_kwargs,
                    )
                # 3, n_frames * seq_len, dim - > 3 * n_frames, seq_len, dim
                self.kf_attn_output = self.attn_output
            else:
                batch_kf_size, _, _ = self.kf_attn_output.shape
                self.attn_output = self.kf_attn_output.view(3, batch_kf_size // 3, sequence_length, dim)[:,
                                   batch_idxs]  # 3, n_frames, seq_len, dim --> 3, len(batch_idxs), seq_len, dim
            if self.use_ada_layer_norm_zero:
                self.attn_output = gate_msa.unsqueeze(1) * self.attn_output

            # gather values from attn_output, using idx as indices, and get a tensor of shape 3, n_frames, seq_len, dim
            if not self.pivotal_pass:
                if len(batch_idxs) == 2:
                    attn_1, attn_2 = self.attn_output[:, 0], self.attn_output[:, 1]
                    attn_output1 = attn_1.gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))
                    attn_output2 = attn_2.gather(dim=1, index=idx2.unsqueeze(-1).repeat(1, 1, dim))

                    s = torch.arange(0, n_frames).to(idx1.device) + batch_idxs[0] * n_frames
                    # distance from the pivot
                    p1 = batch_idxs[0] * n_frames + n_frames // 2
                    p2 = batch_idxs[1] * n_frames + n_frames // 2
                    d1 = torch.abs(s - p1)
                    d2 = torch.abs(s - p2)
                    # weight
                    w1 = d2 / (d1 + d2)
                    w1 = torch.sigmoid(w1)

                    w1 = w1.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(3, 1, sequence_length, dim)
                    attn_output1 = attn_output1.view(3, n_frames, sequence_length, dim)
                    attn_output2 = attn_output2.view(3, n_frames, sequence_length, dim)
                    attn_output = w1 * attn_output1 + (1 - w1) * attn_output2
                else:
                    attn_output = self.attn_output[:,0].gather(dim=1, index=idx1.unsqueeze(-1).repeat(1, 1, dim))

                attn_output = attn_output.reshape(
                        batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            else:
                attn_output = self.attn_output
            hidden_states = hidden_states.reshape(batch_size, sequence_length, dim)  # 3 * n_frames, seq_len, dim
            hidden_states = attn_output + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                hidden_states = attn_output + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]


            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = ff_output + hidden_states

            return hidden_states

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
