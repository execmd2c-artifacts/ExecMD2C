"""Ground-truth core model components for the DExter benchmark.

This file consolidates the denoising model components used by DExter: the
codec-conditioned U-Net blocks and the DiffWave-style classifier-free denoiser.
Only architecture-level components are included; corpus handling, runner loops,
media export, visual demos, and artifact writing are intentionally excluded.
"""

import math
from functools import partial
from inspect import isfunction
from math import sqrt

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from tqdm import tqdm


EPSILON = 1e-6


# --- [Original file: model/utils.py] ---

class Normalization():
    """
    This class is for normalizing the spectrograms batch by batch.
    The normalization used is min-max, two modes 'framewise' and 'imagewise' can be selected.
    In this paper, we found that 'imagewise' normalization works better than 'framewise'
    
    If framewise is used, then X must follow the shape of (B, F, T)
    """
    def __init__(self, min, max, mode='imagewise'):
        if mode == 'framewise':
            def normalize(x):
                size = x.shape
                x_max = x.max(1, keepdim=True)[0] # Finding max values for each frame
                x_min = x.min(1, keepdim=True)[0]  
                x_std = (x-x_min)/(x_max-x_min) # If there is a column with all zero, nan will occur
                x_std[torch.isnan(x_std)]=0 # Making nan to 0
                x_scaled = x_std * (max - min) + min
                return x_scaled
        elif mode == 'imagewise':
            def normalize(x):
                # x_max = x.view(size[0], size[1]*size[2]).max(1, keepdim=True)[0]
                # x_min = x.view(size[0], size[1]*size[2]).min(1, keepdim=True)[0]
                x_max = x.flatten(1).max(1, keepdim=True)[0]
                x_min = x.flatten(1).min(1, keepdim=True)[0]
                x_max = x_max.unsqueeze(1) # Make it broadcastable
                x_min = x_min.unsqueeze(1) # Make it broadcastable 
                x_std = (x-x_min)/(x_max-x_min)
                x_scaled = x_std * (max - min) + min
                x_scaled[torch.isnan(x_scaled)]=min # if piano roll is empty, turn them to min
                return x_scaled
        elif mode == 'rowwise':
            def normalize(x):
                x_max = x.max(2)[0].unsqueeze(2)
                x_min = x.min(2)[0].unsqueeze(2)
                x_std = (x-x_min)/(x_max-x_min)
                x_scaled = x_std * (max - min) + min
                x_scaled[torch.isnan(x_scaled)]=min 
                return x_scaled            
        else:
            print(f'please choose the correct mode')
        self.normalize = normalize

    def __call__(self, x):
        return self.normalize(x)


# --- [Original file: task/diffusion.py] ---

def linear_beta_schedule(beta_start, beta_end, timesteps):
    return torch.linspace(beta_start, beta_end, timesteps)


class CodecDiffusion(pl.LightningModule):
    def __init__(self,
                 lr,
                 timesteps,
                 loss_type,
                 loss_keys,
                 beta_start,
                 beta_end,
                 training,
                 sampling,
                 samples_root,
                 debug=False,
                 generation_filter=0.0,
                 **kwargs
                ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # define beta schedule (beta is variance)
        self.betas = linear_beta_schedule(beta_start, beta_end, timesteps=timesteps)

        # define alphas 
        alphas = 1. - self.betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - alphas_cumprod_prev) / (1- alphas_cumprod)
        self.inner_loop = tqdm(range(self.hparams.timesteps), desc='sampling loop time step')
        
        self.reverse_diffusion = getattr(self, sampling['type'])
        # self.reverse_diffusion = getattr(self, sampling.type)
        self.alphas = alphas


# --- [Original file: model/unet.py] ---

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)


def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
#         print(f"after math.log {embeddings.shape=}")        
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings) # taking exp
        embeddings = time[:, None] * embeddings[None, :] # boardcasting (B, dim//2)
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # apply sin and cos (B, dim)
        return embeddings


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, "b c -> b c 1 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)


class ConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)    


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)    


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)    


class CodecConvNextBlock(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        
        self.cond_ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        
        self.cond_net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )        

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond, time_emb=None):
        h = self.ds_conv(x)
        cond_h = self.cond_ds_conv(cond)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            time = self.mlp(time_emb)
            h = h + cond_h + rearrange(time, "b c -> b c 1 1")

        h = self.net(h)
        cond_h = self.cond_net(cond_h)
        return h + self.res_conv(x), cond_h


class CodecConvNextBlockUp(nn.Module):
    """https://arxiv.org/abs/2201.03545"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if exists(time_emb_dim)
            else None
        )
    
        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.cond_ds_conv = nn.Conv2d(dim//3, dim, 7, padding=3, groups=1)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )
        
        self.cond_net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )        

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond, time_emb=None):
        h = self.ds_conv(x)
        cond_h = self.cond_ds_conv(cond)
        if exists(self.mlp) and exists(time_emb):
            assert exists(time_emb), "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + cond_h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        cond_h = self.cond_net(cond_h)
        return h + self.res_conv(x), cond_h    


class DenoiserUnet(CodecDiffusion):
    # Unet conditioned on spectrogram
    def __init__(
        self,
        dim,
        condition,
        p_codec_rows,
        s_codec_rows,
        c_codec_rows,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        spec_args={},
        **kwargs,
    ):
        super().__init__(**kwargs)

        init_dim = default(init_dim, dim) # 256
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.init_fc = nn.Linear(p_codec_rows, init_dim)
        
        # Initial layers for spectrograms
        self.condition_init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)
        self.condition_init_fc = nn.Linear((s_codec_rows + c_codec_rows), init_dim)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if condition == 'trainable_score':
            trainable_parameters = torch.full((s_codec_rows + c_codec_rows, self.hparams.seg_len), -1).float() 
            
            trainable_parameters = nn.Parameter(trainable_parameters, requires_grad=True)
            self.register_parameter("trainable_parameters", trainable_parameters)
            self.uncon_dropout = self.trainable_dropout        
            
        elif condition == 'fixed':
            self.uncon_dropout = self.fixed_dropout
        else:
            raise ValueError("unrecognized condition '{condition}'")


        if use_convnext:
            block_klass = partial(CodecConvNextBlock, mult=convnext_mult)
            up_block_klass = partial(CodecConvNextBlockUp, mult=convnext_mult)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        
        self.condition_downs = nn.ModuleList([])
        self.condition_ups = nn.ModuleList([])
        
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )     

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        up_block_klass(dim_out * 3, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_block = block_klass(dim, dim)
        self.final_conv = nn.Conv2d(dim, out_dim, 1)
        self.final_fc = nn.Linear(dim, p_codec_rows)
        

    # def forward(self, x, waveform, diffusion_step):
    def forward(self, x_t, s_codec, c_codec, diffusion_step, sampling=False):
        """
        x_t : (B, 1, T, N)
        s_codec : (B, N, 4)
        c_codec : (B, N, 4)
        """

        if self.training: # only use dropout during training
            s_codec = self.uncon_dropout(s_codec, self.hparams.cond_dropout) # making some score 0 to be unconditional
            c_codec = self.uncon_dropout(c_codec, self.hparams.cond_dropout) 

        condition = torch.cat((s_codec, c_codec), dim=2)
        condition = rearrange(condition, "b w h -> b 1 w h")
        condition = self.condition_init_conv(condition.float())  # (B, dim, N, 4)
        condition = self.condition_init_fc(condition) # (B, dim, N, dim)
        
        x_t = self.init_conv(x_t) # (B, dim, N, 5)
        x_t = self.init_fc(x_t) # (B, dim, N, dim)
        t = self.time_mlp(diffusion_step) if exists(self.time_mlp) else None # (B, dim*4)
        h = []

        # downsample
        counter = 0
        for block1, block2, attn, downsample, condition_downsample in self.downs:
            x_t, condition = block1(x_t, condition, t)
            x_t, condition = block2(x_t, condition, t)
            x_t = attn(x_t)
            h.append([x_t, condition])
            x_t = downsample(x_t)
            condition = condition_downsample(condition)
            counter += 1 

        # bottleneck
        x_t, condition = self.mid_block1(x_t, condition, t)
        x_t = self.mid_attn(x_t)
        x_t, condition = self.mid_block2(x_t, condition, t)
        

        # upsample
        for block1, block2, attn, upsample, condition_upsample in self.ups:       
            x_t = torch.cat((x_t, *h.pop()), dim=1)
            x_t, condition = block1(x_t, condition, t)
            x_t, condition = block2(x_t, condition, t)
            x_t = attn(x_t)
            x_t = upsample(x_t)
            condition = condition_upsample(condition)
            
        x_t, condition = self.final_block(x_t, condition)
        x_t = self.final_conv(x_t)
        x_t = self.final_fc(x_t)

        return x_t, condition

    def fixed_dropout(self, x, p, masked_value=-1):
        mask = torch.distributions.Bernoulli(probs=(p)).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = masked_value
        return x

    def trainable_dropout(self, x, p):
        mask = torch.distributions.Bernoulli(probs=(p)).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = self.trainable_parameters
        return x


# --- [Original file: model/diffwave.py] ---

Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


def Conv2d(*args, **kwargs):
    layer = nn.Conv2d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, max_steps):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_steps), persistent=False)
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high - low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)  # [T,1]
        dims = torch.arange(64).unsqueeze(0)          # [1,64]
        table = steps * 10.0**(dims * 4.0 / 63.0)     # [T,64]
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class ResidualBlock(nn.Module):
    def __init__(self,
                 residual_channels,
                 dilation,
                 kernel_size=3,
                 uncond=False,
                 condition_rows=4):
        '''
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable s_codec conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels,
                                   2 * residual_channels,
                                   kernel_size,
                                   padding=((kernel_size-1)*(dilation-1)+kernel_size-1)//2,
                                   dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond: # conditional model
            self.conditioner_projection = Conv1d(condition_rows, 2 * residual_channels, 1)
        else: # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None: # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class ClassifierFreeDenoiser(CodecDiffusion):
    def __init__(self,
                 residual_channels,
                 unconditional,
                 condition,
                 p_codec_rows,
                 s_codec_rows,
                 c_codec_rows,
                 norm_args,
                 seg_len,
                 residual_layers = 30,
                 kernel_size = 3,
                 dilation_base = 1,
                 dilation_bound = 4,
                 cond_dropout = 0.5,
                 **kwargs):
        
        self.cond_dropout = cond_dropout
        super().__init__(**kwargs)

        self.condition_normalize = Normalization(norm_args[0], norm_args[1], norm_args[2])

        self.input_projection = Conv1d(p_codec_rows, residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(self.betas))
        
        if condition == 'trainable_score':
            trainable_parameters = torch.full((s_codec_rows, self.hparams.seg_len), -1).float() 
            
            trainable_parameters = nn.Parameter(trainable_parameters, requires_grad=True)
            self.register_parameter("trainable_parameters", trainable_parameters)
            self.uncon_dropout = self.trainable_dropout        
            
        elif condition == 'fixed':
            self.uncon_dropout = self.fixed_dropout
        else:
            raise ValueError("unrecognized condition '{condition}'")
        
        # Original dilation was 2**(i % dilation_cycle_length)           
        self.residual_layers = nn.ModuleList([
            ResidualBlock(residual_channels, dilation_base**(i % dilation_bound), kernel_size, 
                                  uncond=unconditional, condition_rows=s_codec_rows)
            for i in range(residual_layers)
        ])

        #FiLM condition parameter (beta and gamma) generator for score information (MIDI frame roll)
        #The socre length is fixed for 20 sec and 32 resolution/second for the MIDI frame roll. 
        # self.film_layer = nn.Linear(in_features= 200*c_codec_rows, out_features= 2*residual_layers, bias= True)

        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, p_codec_rows, 1)
        nn.init.zeros_(self.output_projection.weight)
        

    def forward(self, x_t, s_codec, c_codec, diffusion_step, sampling=False):
        """
        x_t : (B, 1, T, N)
        s_codec : (B, 4, N)
        c_codec : (B, 7, N)
        """
        
        x_t = x_t.squeeze(1).transpose(1, 2) # (B, 5, LEN)

        s_codec = s_codec.transpose(1, 2).float()
        s_codec = self.condition_normalize(s_codec)
        # c_codec = self.condition_normalize(c_codec)
        if self.training: # only use dropout during training
            s_codec = self.uncon_dropout(s_codec, self.hparams.cond_dropout) # making some score 0 to be unconditional
            # c_codec = self.uncon_dropout(c_codec, self.hparams.cond_dropout) 

        # Generate FiLM conditions (beta and gamma) by FiLM generator (Linear layer)
        # c_codec_flat = torch.flatten(c_codec, start_dim = 1).float()
        # film_feat = self.film_layer(c_codec_flat)

        x = self.input_projection(x_t) # (B, 512, LEN)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step) # (B, 512)
            
        skip = None
        index = 0
        for layer in self.residual_layers:
            index += 1

            # gamma = film_feat[:, 2*(index-1)]
            # beta = film_feat[:, 2*index-1]
            
            # all shapes: (B, 512, LEN)
            x, skip_connection = layer(x, diffusion_step, s_codec)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers)) 
        x = self.skip_projection(x) # (B, 512, LEN)
        x = F.relu(x)
        x = self.output_projection(x) # (B, 4, LEN)

        return x.transpose(1,2).unsqueeze(1), s_codec # (B, 1, LEN, 4)
    
    
    def fixed_dropout(self, x, p, masked_value=-1):
        mask = torch.distributions.Bernoulli(probs=(p)).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = masked_value
        return x
    
    def trainable_dropout(self, x, p):
        mask = torch.distributions.Bernoulli(probs=(p)).sample((x.shape[0],)).long()
        mask_idx = mask.nonzero()
        x[mask_idx] = self.trainable_parameters
        return x


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

    diffusion_kwargs = dict(
        lr=1e-4,
        timesteps=8,
        loss_type="l1",
        loss_keys=["diffusion_loss"],
        beta_start=0.0001,
        beta_end=0.02,
        training={"mode": "epsilon", "target": "gen_noise"},
        sampling={"type": "fixed_dropout", "w": 1.0},
        samples_root="",
        seg_len=8,
        cond_dropout=0.0,
    )
    denoiser_kwargs = dict(diffusion_kwargs)
    denoiser_kwargs.pop("seg_len")

    print("DExter benchmark checks")

    print("[Test 1/5] CodecConvNextBlock.forward")
    try:
        block = CodecConvNextBlock(dim=8, dim_out=16, time_emb_dim=32, mult=1)
        x = torch.randn(2, 8, 6, 6)
        cond = torch.randn(2, 8, 6, 6)
        t = torch.randn(2, 32)
        out, cond_out = block(x, cond, t)
        check("CodecConvNextBlock output not None", out is not None and cond_out is not None)
        if out is not None and cond_out is not None:
            check("CodecConvNextBlock output shape", tuple(out.shape) == (2, 16, 6, 6), str(tuple(out.shape)))
            check("CodecConvNextBlock condition shape", tuple(cond_out.shape) == (2, 16, 6, 6), str(tuple(cond_out.shape)))
            check("CodecConvNextBlock finite outputs", torch.isfinite(out).all().item() and torch.isfinite(cond_out).all().item())
            check("CodecConvNextBlock changes channel width", out.size(1) == 16 and cond_out.size(1) == 16)
        else:
            skip_checks(4, "CodecConvNextBlock returned None")
    except Exception as exc:
        skip_checks(5, f"CodecConvNextBlock.forward raised {type(exc).__name__}: {exc}")

    print("[Test 2/5] CodecConvNextBlockUp.forward")
    try:
        block = CodecConvNextBlockUp(dim=24, dim_out=8, time_emb_dim=32, mult=1)
        x = torch.randn(2, 24, 4, 4)
        cond = torch.randn(2, 8, 4, 4)
        t = torch.randn(2, 32)
        out, cond_out = block(x, cond, t)
        check("CodecConvNextBlockUp output not None", out is not None and cond_out is not None)
        if out is not None and cond_out is not None:
            check("CodecConvNextBlockUp output shape", tuple(out.shape) == (2, 8, 4, 4), str(tuple(out.shape)))
            check("CodecConvNextBlockUp condition shape", tuple(cond_out.shape) == (2, 8, 4, 4), str(tuple(cond_out.shape)))
            check("CodecConvNextBlockUp finite outputs", torch.isfinite(out).all().item() and torch.isfinite(cond_out).all().item())
            check("CodecConvNextBlockUp consumes one-third condition width", block.cond_ds_conv.in_channels == 8,
                  str(block.cond_ds_conv.in_channels))
        else:
            skip_checks(4, "CodecConvNextBlockUp returned None")
    except Exception as exc:
        skip_checks(5, f"CodecConvNextBlockUp.forward raised {type(exc).__name__}: {exc}")

    print("[Test 3/5] DenoiserUnet.forward")
    try:
        model = DenoiserUnet(
            dim=8,
            condition="fixed",
            p_codec_rows=5,
            s_codec_rows=4,
            c_codec_rows=7,
            channels=1,
            dim_mults=(1, 2),
            convnext_mult=1,
            **diffusion_kwargs,
        )
        model.eval()
        x_t = torch.randn(2, 1, 8, 5)
        s_codec = torch.randn(2, 8, 4)
        c_codec = torch.randn(2, 8, 7)
        diffusion_step = torch.tensor([1, 3], dtype=torch.long)
        out, condition = model(x_t, s_codec, c_codec, diffusion_step)
        check("DenoiserUnet output not None", out is not None and condition is not None)
        if out is not None and condition is not None:
            check("DenoiserUnet output shape", tuple(out.shape) == (2, 1, 8, 5), str(tuple(out.shape)))
            check("DenoiserUnet condition shape", tuple(condition.shape) == (2, 8, 8, 8), str(tuple(condition.shape)))
            check("DenoiserUnet finite output", torch.isfinite(out).all().item())
            check("DenoiserUnet preserves codec row count", out.size(-1) == 5, str(out.size(-1)))
        else:
            skip_checks(4, "DenoiserUnet returned None")
    except Exception as exc:
        skip_checks(5, f"DenoiserUnet.forward raised {type(exc).__name__}: {exc}")

    print("[Test 4/5] ResidualBlock.forward")
    try:
        block = ResidualBlock(residual_channels=8, dilation=2, kernel_size=3, uncond=False, condition_rows=4)
        x = torch.randn(2, 8, 10)
        diffusion_step = torch.randn(2, 512)
        conditioner = torch.randn(2, 4, 10)
        residual, skip = block(x, diffusion_step, conditioner)
        check("ResidualBlock output not None", residual is not None and skip is not None)
        if residual is not None and skip is not None:
            check("ResidualBlock residual shape", tuple(residual.shape) == (2, 8, 10), str(tuple(residual.shape)))
            check("ResidualBlock skip shape", tuple(skip.shape) == (2, 8, 10), str(tuple(skip.shape)))
            check("ResidualBlock finite outputs", torch.isfinite(residual).all().item() and torch.isfinite(skip).all().item())
            check("ResidualBlock preserves temporal length", residual.size(-1) == x.size(-1) and skip.size(-1) == x.size(-1))
        else:
            skip_checks(4, "ResidualBlock returned None")
    except Exception as exc:
        skip_checks(5, f"ResidualBlock.forward raised {type(exc).__name__}: {exc}")

    print("[Test 5/5] ClassifierFreeDenoiser.forward")
    try:
        model = ClassifierFreeDenoiser(
            residual_channels=8,
            unconditional=False,
            condition="fixed",
            p_codec_rows=5,
            s_codec_rows=4,
            c_codec_rows=7,
            norm_args=[0, 1, "rowwise"],
            seg_len=8,
            residual_layers=2,
            kernel_size=3,
            dilation_base=2,
            dilation_bound=2,
            **denoiser_kwargs,
        )
        model.eval()
        x_t = torch.randn(2, 1, 8, 5)
        s_codec = torch.randn(2, 8, 4)
        c_codec = torch.randn(2, 8, 7)
        diffusion_step = torch.tensor([2, 4], dtype=torch.long)
        out, cond = model(x_t, s_codec, c_codec, diffusion_step)
        check("ClassifierFreeDenoiser output not None", out is not None and cond is not None)
        if out is not None and cond is not None:
            check("ClassifierFreeDenoiser output shape", tuple(out.shape) == (2, 1, 8, 5), str(tuple(out.shape)))
            check("ClassifierFreeDenoiser condition shape", tuple(cond.shape) == (2, 4, 8), str(tuple(cond.shape)))
            check("ClassifierFreeDenoiser finite output", torch.isfinite(out).all().item())
            check("ClassifierFreeDenoiser preserves p-codec rows", out.size(-1) == 5, str(out.size(-1)))
        else:
            skip_checks(4, "ClassifierFreeDenoiser returned None")
    except Exception as exc:
        skip_checks(5, f"ClassifierFreeDenoiser.forward raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
