# ============================================================
# ground_truth.py - EDMP Core Temporal U-Net Model Components
# Source: EDMP-main
#
# Contains ONLY the diffusion denoiser architecture components.
# No training loop, dataset, guide, environment, robot geometry, or
# inference pipeline code is included.
# ============================================================

import os

import einops
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as tvtf
from einops.layers.torch import Rearrange


# --- [Original file: diffusion/models/blocks.py] ---
#################################################################################################################
#---------------------------------------------- LOWER LEVEL BLOCKS ---------------------------------------------#
#################################################################################################################

#--------------------------------------------- 1D CONVOLUTION BLOCK --------------------------------------------#

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            Rearrange('batch channels horizon -> batch channels 1 horizon'),
            nn.GroupNorm(n_groups, out_channels),
            Rearrange('batch channels 1 horizon -> batch channels horizon'),
            nn.Mish(),
        )
        # Why is padding half of kernel size? This makes sure that in the first convolution, half the kernel has zero elements and the other
        # half has the first few elements of the input. The same applies to the last convolution.

    def forward(self, x):

        return self.block(x)
    
#----------------------------------------- SINUSOIDAL POSITION EMBEDDING ---------------------------------------#

class SinusoidalPosEmb(nn.Module):

    def __init__(self, dim, device):
        
        super().__init__()
        self.dim = dim
        self.device = device

    def forward(self, x):

        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=self.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb

#------------------------------------------------- TEMPORAL MLP ------------------------------------------------#

class TimeMLP(nn.Module):

    def __init__(self, time_embed_dim, out_channels):

        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),      # This is probably extending by a dimension
        )

    def forward(self, t):

        return self.time_mlp(t)

#-------------------------------------------- INITIAL TIME EMBEDDING -------------------------------------------#

class TimeEmbedding(nn.Module):

    def __init__(self, dim, device):

        super().__init__()
        self.device = device
        
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim, self.device),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t):

        return self.time_mlp(t)

#################################################################################################################
#------------------------------------------ INTERMEDIATE LEVEL BLOCKS ------------------------------------------#
#################################################################################################################

#---------------------------------------------- LINEAR ATTENTION -----------------------------------------------#

class LinearAttention(nn.Module):
    
    def __init__(self, dim, heads=4, dim_head=32):
        
        super().__init__()
        
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1)

    def forward(self, x):

        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b (h c) d -> b h c d', h=self.heads), qkv)

        q = q * self.scale
        k = k.softmax(dim = -1)

        # Produces the context of each element in k w.r.t all other elements in v. (weighted sum of every pair of rows possible)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        # Weighted sum of every pair of columns possible:
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        
        # Recombine the the 3 chunks that where seperated to get back the same dimension as input
        out = einops.rearrange(out, 'b h c d -> b (h c) d')
        
        # Convolve back from hidden channels to the original number of channels (dim)
        out = self.to_out(out)

        return out

#------------------------------------------ RESIDUAL CONVOLUTION BLOCK ------------------------------------------#

class ResidualConvolutionBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, time_embed_dim, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = TimeMLP(time_embed_dim, out_channels)

        # The convolution that forms the residual connection between input and output
        # If the input and the output have the same number of channels, this is an identity matrix.
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x time_embed_dim ]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)

        return out
    
#------------------------------------------- RESIDUAL ATTENTION BLOCK -------------------------------------------#

class ResidualAttentionBlock(nn.Module):

    def __init__(self, dim, eps = 1e-5):
        
        super().__init__()

        self.attention = LinearAttention(dim)

        # Layer Norm Parameters:
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1))

    def forward(self, x):

        # Layer Norm
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        out = (x - mean) / (var + self.eps).sqrt() * self.g + self.b

        # Attention Layer
        out = self.attention(out)

        # Residual Connection
        out = out + x

        return out

#################################################################################################################
#---------------------------------------------- HIGH LEVEL BLOCKS ----------------------------------------------#
#################################################################################################################

class DownSampler(nn.Module):
    
    def __init__(self, dim_in, dim_out, time_dim, is_last = False):

        super().__init__()
        
        self.down = nn.ModuleList([ResidualConvolutionBlock(dim_in, dim_out, time_embed_dim = time_dim),
                                   ResidualConvolutionBlock(dim_out, dim_out, time_embed_dim = time_dim),
                                   nn.Identity(),  # Replace this with ResidualAttentionBlock
                                   nn.Conv1d(dim_out, dim_out, kernel_size = 3, stride = 2, padding = 1) if not is_last else nn.Identity()])

    def forward(self, x, t):
        
        x = self.down[0](x, t)
        x = self.down[1](x, t)
        h = self.down[2](x)   
        out = self.down[3](h)

        return out, h

class MiddleBlock(nn.Module):
    
    def __init__(self, mid_dim, time_dim):

        super().__init__()

        self.middle = nn.ModuleList([ResidualConvolutionBlock(mid_dim, mid_dim, time_embed_dim = time_dim),
                                     nn.Identity(),  # Replace this with ResidualAttentionBlock
                                     ResidualConvolutionBlock(mid_dim, mid_dim, time_embed_dim = time_dim)])

    def forward(self, x, t):

        x = self.middle[0](x, t)
        x = self.middle[1](x)
        out = self.middle[2](x, t)

        return out

class UpSampler(nn.Module):
    
    def __init__(self, dim_in, dim_out, time_dim, is_last = False):

        super().__init__()

        self.up = nn.ModuleList([ResidualConvolutionBlock(dim_out * 2, dim_in, time_embed_dim = time_dim),
                                 ResidualConvolutionBlock(dim_in, dim_in, time_embed_dim = time_dim),
                                 nn.Identity(),  # Replace this with ResidualAttentionBlock
                                 nn.ConvTranspose1d(dim_in, dim_in, kernel_size = 4, stride = 2, padding = 1) if not is_last else nn.Identity()])

    def forward(self, x, h, t):

        x = torch.cat([x, h], dim = 1)

        x = self.up[0](x, t)
        x = self.up[1](x, t)
        x = self.up[2](x)   
        out = self.up[3](x)

        return out


# --- [Original file: diffusion/models/temporalunet.py] ---
class TemporalUNet(nn.Module):

    def __init__(self, model_name, input_dim, time_dim, device, dims = (32, 64, 128, 256)):

        super(TemporalUNet, self).__init__()

        dims = [input_dim, *dims]  # length of dims is 5

        # Initial Time Embedding:
        self.time_embedding = TimeEmbedding(time_dim, device)

        # Down Sampling:
        self.down_samplers = nn.ModuleList([])
        for i in range(len(dims) - 2):      # Loops 0, 1, 2
            self.down_samplers.append(DownSampler(dims[i], dims[i+1], time_dim))
        self.down_samplers.append(DownSampler(dims[-2], dims[-1], time_dim, is_last = True))  # 3 -> 4

        # Middle Block:
        self.middle_block = MiddleBlock(dims[-1], time_dim)

        # Up Sampling:
        self.up_samplers = nn.ModuleList([])
        for i in range(len(dims) - 1, 1, -1):  # Loops 4, 3, 2  since the last one is a seperate convolution
            self.up_samplers.append(UpSampler(dims[i-1], dims[i], time_dim))

        # Final Convolution:
        self.final_conv = nn.Sequential(Conv1dBlock(dims[1], dims[1], kernel_size = 5),
                                        nn.Conv1d(dims[1], input_dim, kernel_size = 1))
        
        self.model_name = model_name
        if not os.path.exists(model_name):
            os.mkdir(model_name)
            self.losses = np.array([])
        else:
            self.load()

        _ = self.to(device)

    def forward(self, x, t):
        """
        x => Tensor of size (batch_size, traj_len*2)
        t => Integer representing the diffusion timestep of x
        """
        
        # Get the time embedding from t:
        time_emb = self.time_embedding(t)

        # Down Sampling Layers:
        h_list = []
        for i in range(len(self.down_samplers)):
            x, h = self.down_samplers[i](x, time_emb)
            h_list.append(h)

        # Middle Layer:
        x = self.middle_block(x, time_emb)

        # Up Sampling Layers:
        for i in range(len(self.up_samplers)):
            h_temp = h_list.pop()
            # print(f"Shape of x: {x.shape}\t Shape of h_list: {h_temp.shape}")
            x = self.up_samplers[i](x, h_temp, time_emb)   # How does pop work and not h_list[i]
            if x.shape[2] == 8 or x.shape[2] == 14 or x.shape[2] == 26 or x.shape[2] == 8: # Upsampling doubles the dimensions of the input. So, we are manually cropping the extra size to match the size of it's corresponding h (context/residual from the downsampling layer)
                x = tvtf.crop(x, 0, 0, x.shape[1], x.shape[2] - 1)

        # Final Convolution
        out = self.final_conv(x)

        return out

    def save(self):

        torch.save(self.state_dict(), self.model_name + "/weights_latest.pt")
        np.save(self.model_name + "/losses.npy", self.losses)

    def save_checkpoint(self, checkpoint):
        
        torch.save(self.state_dict(), self.model_name + "/weights_" + str(checkpoint) + ".pt")
        np.save(self.model_name + "/latest_checkpoint.npy", checkpoint)
    
    def load(self):

        self.losses = np.load(self.model_name + "/losses.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_latest.pt"))
        print("Loaded Model at " + str(self.losses.size) + " epochs")

    def load_checkpoint(self, checkpoint):

        _ = input("Press Enter if you are running the model for inference, or Ctrl+C\n(Never load a checkpoint for training! This will overwrite progress)")
        
        latest_checkpoint = np.load(self.model_name + "/latest_checkpoint.npy")
        self.load_state_dict(torch.load(self.model_name + "/weights_" + str(checkpoint) + ".pt"))
        self.losses = np.load(self.model_name + "/losses.npy")[:checkpoint]


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    import shutil
    import tempfile

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
    print("EDMP Temporal U-Net Diffusion Denoiser")
    print("Automated Test Suite - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: ResidualConvolutionBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] ResidualConvolutionBlock.forward - time-conditioned residual convolution")
    try:
        block = ResidualConvolutionBlock(4, 8, time_embed_dim=16).to(device)
        x = torch.randn(2, 4, 15, device=device, requires_grad=True)
        t1 = torch.randn(2, 16, device=device)
        t2 = t1 + 3.0
        y = block(x, t1)
        check("ResidualConv output not None", y is not None)
        if y is not None:
            check("ResidualConv output shape", tuple(y.shape) == (2, 8, 15), f"expected (2, 8, 15), got {tuple(y.shape)}")
            check("ResidualConv output finite", torch.isfinite(y).all().item())
            y_changed_time = block(x.detach(), t2)
            check("ResidualConv time embedding affects output", not torch.allclose(y.detach(), y_changed_time.detach(), atol=1e-6))
            y.sum().backward()
            check("ResidualConv gradients reach input", x.grad is not None and x.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "ResidualConvolutionBlock.forward returned None")
    except Exception as exc:
        print(f"  [ResidualConvolutionBlock.forward] ERROR: {exc}")
        skip_checks(5, "ResidualConvolutionBlock.forward raised an exception")
    print()

    # ==============================================================
    # Test 2/4: DownSampler.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] DownSampler.forward - residual feature extraction and downsampling")
    try:
        block = DownSampler(4, 8, time_dim=16).to(device)
        x = torch.randn(2, 4, 15, device=device, requires_grad=True)
        t = torch.randn(2, 16, device=device)
        out, h = block(x, t)
        check("DownSampler output not None", out is not None and h is not None)
        if out is not None and h is not None:
            check("DownSampler downsample shape", tuple(out.shape) == (2, 8, 8), f"expected (2, 8, 8), got {tuple(out.shape)}")
            check("DownSampler skip shape", tuple(h.shape) == (2, 8, 15), f"expected (2, 8, 15), got {tuple(h.shape)}")
            check("DownSampler outputs finite", torch.isfinite(out).all().item() and torch.isfinite(h).all().item())
            (out.sum() + h.sum()).backward()
            check("DownSampler gradients reach input", x.grad is not None and x.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "DownSampler.forward returned None")
    except Exception as exc:
        print(f"  [DownSampler.forward] ERROR: {exc}")
        skip_checks(5, "DownSampler.forward raised an exception")
    print()

    # ==============================================================
    # Test 3/4: UpSampler.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] UpSampler.forward - skip concatenation and transposed upsampling")
    try:
        block = UpSampler(8, 16, time_dim=16).to(device)
        x = torch.randn(2, 16, 8, device=device, requires_grad=True)
        h = torch.randn(2, 16, 8, device=device, requires_grad=True)
        t = torch.randn(2, 16, device=device)
        y = block(x, h, t)
        check("UpSampler output not None", y is not None)
        if y is not None:
            check("UpSampler output shape", tuple(y.shape) == (2, 8, 16), f"expected (2, 8, 16), got {tuple(y.shape)}")
            check("UpSampler output finite", torch.isfinite(y).all().item())
            y_zero_skip = block(x.detach(), torch.zeros_like(h), t)
            check("UpSampler skip branch affects output", not torch.allclose(y.detach(), y_zero_skip.detach(), atol=1e-6))
            y.sum().backward()
            check("UpSampler gradients reach both inputs",
                  x.grad is not None and x.grad.abs().sum().item() > 0 and
                  h.grad is not None and h.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "UpSampler.forward returned None")
    except Exception as exc:
        print(f"  [UpSampler.forward] ERROR: {exc}")
        skip_checks(5, "UpSampler.forward raised an exception")
    print()

    # ==============================================================
    # Test 4/4: TemporalUNet.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] TemporalUNet.forward - full denoiser U-Net path")
    tmp_root = tempfile.mkdtemp()
    try:
        model_path = os.path.join(tmp_root, "TemporalUNetModel_test")
        model = TemporalUNet(model_name=model_path, input_dim=7, time_dim=16, device=device, dims=(8, 16, 32, 64)).to(device)
        x = torch.randn(2, 7, 25, device=device, requires_grad=True)
        timestep = torch.tensor([5.0, 9.0], device=device)
        y = model(x, timestep)
        check("TemporalUNet output not None", y is not None)
        if y is not None:
            check("TemporalUNet output shape", tuple(y.shape) == (2, 7, 25), f"expected (2, 7, 25), got {tuple(y.shape)}")
            check("TemporalUNet output finite", torch.isfinite(y).all().item())
            y_changed_time = model(x.detach(), timestep + 4.0)
            check("TemporalUNet timestep affects output", not torch.allclose(y.detach(), y_changed_time.detach(), atol=1e-6))
            y.mean().backward()
            grad = model.final_conv[-1].weight.grad
            check("TemporalUNet gradients reach final convolution", grad is not None and grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "TemporalUNet.forward returned None")
    except Exception as exc:
        print(f"  [TemporalUNet.forward] ERROR: {exc}")
        skip_checks(5, "TemporalUNet.forward raised an exception")
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
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
