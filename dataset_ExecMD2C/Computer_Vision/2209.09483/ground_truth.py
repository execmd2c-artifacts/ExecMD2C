# Source-faithful benchmark extraction for DiffusionUnit-main.
# Model code above __main__ is copied from model/blocks.py and model/architectures.py
# with only one-file import consolidation edits.

import numpy as np
from lib.pointops.functions import pointops
import torch.nn as nn
import torch
import torch.nn.functional as F
import copy


# Copied from model/blocks.py
def block_decider(name):
    if name == 'dw_kpconv':
        return DepthwiseKPConv
    if name == 'residual':
        return ResidualBlock
    if name == 'upsample':
        return Upsampling 
    if name == 'downsample':
        return Downsampling 
    if name == 'simple':
        return SimpleBlock 
    if name == 'diffusion_unit':
        return DiffusionUnit 

# blocks 
class SimpleBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        func = config.convolution
        self.func = block_decider(func)(d_in, d_out, config)
        self.nsample = nsample
    
    def forward(self, p, x, o):
        N, C = x.size()
        pj_xj = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)  # (m, nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        return p, x, o
 
   
class ResidualBlock(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        func = config.convolution
        bottleneck_ratio = config.bottleneck_ratio
        if bottleneck_ratio is None:
            self.reduction = self.expansion = nn.Identity()
            self.func = block_decider(func)(d_in, d_out, config)
        else:
            d_mid = d_in // bottleneck_ratio
            self.reduction = nn.Sequential(
                nn.Linear(d_in, d_mid),
                nn.BatchNorm1d(d_mid),
                nn.ReLU(inplace=True)
            )
            self.func = block_decider(func)(d_mid, d_mid, config)
            self.expansion = nn.Sequential(
                nn.Linear(d_mid, d_out),
                nn.BatchNorm1d(d_out),
                nn.ReLU(inplace=True)
            )
        self.nsample = nsample
    
    def forward(self, p, x, o):
        N, C = x.size()
        identity = x
        x = self.reduction(x)
        pj_xj = pointops.queryandgroup(self.nsample, p, p, x, None, o, o, use_xyz=True)  # (m, nsample, 3+c)
        pj, xj = pj_xj[:, :, 0:3], pj_xj[:, :, 3:]
        x = self.func(p, pj, x, xj)
        x = self.expansion(x)
        x = identity + x
        return p, x, o
        

class Downsampling(nn.Module):
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        self.d_in = d_in
        self.nsample = 16 
        self.stride = stride 
        self.mlp = nn.Sequential(
            nn.Linear(d_in+3, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, p, x, o):
        identity = x

        n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride
        for i in range(1, o.shape[0]):
            count += (o[i].item() - o[i-1].item()) // self.stride
            n_o.append(count)
        n_o = torch.cuda.IntTensor(n_o)
        idx = pointops.furthestsampling(p, o, n_o)  # (m)
        n_p = p[idx.long(), :]  # (m, 3)
        
        pj_xj = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)  
        pj, xj = pj_xj[:, :, :3], pj_xj[:, :, 3:]
        pj = pj / (torch.max(torch.norm(pj, dim=-1, keepdim=True), dim=1, keepdim=True)[0] + 1e-8)
        pj_xj = torch.cat([pj, xj], dim=-1)
        x = self.mlp(pj_xj.max(1)[0])
        return n_p, x, n_o 


class Upsampling(nn.Module):
    def __init__(self, d_in_sparse_dense, d_out, nsample, stride, config):
        super().__init__()
        d_in_sparse, d_in_dense = d_in_sparse_dense
        self.nsample = nsample
        self.d_out = d_out

        self.mlp = nn.Sequential(
            nn.Linear(d_in_sparse+ d_in_dense, d_out),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, p1,x1,o1, p2,x2,o2):
        '''
            pxo1: dense 
            pxo2: sparse  
        '''
        interpolated = pointops.interpolation(p2, p1, x2, o2, o1)
        x = self.mlp(torch.cat([x1, interpolated], dim=1))
        return p1, x, o1

# Convolutions
class DepthwiseKPConv(nn.Module):
    def __init__(self, d_in, d_out, config):
        super().__init__()
        if d_in == d_out:
            d_mid = d_in
            num_group = d_mid
            self.first_layer = False
        else:
            d_in = d_in + 3
            num_group = 1
            self.first_layer = True
        kernel_point = np.load('../model/kernels/dispositions/k_015_center_3D.npy').reshape(1,-1, 3) # (1, n_k, 3)

        num_kernel = kernel_point.shape[1]
        self.sigma = 0.3
        self.scale = (self.sigma ** 2) * 2 + 1e-10

        self.kernel_point = nn.Parameter(
            torch.tensor(kernel_point, dtype=torch.float32),
            requires_grad=False
        )
         
        self.depthwise = nn.Sequential(
            nn.Conv1d(d_in, d_out, num_kernel, groups=num_group),
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, p, pj, x, xj):
        l2_dist = torch.norm(pj, p=2, dim=2, keepdim=True) 
        pj = pj / (torch.max(l2_dist, dim=1, keepdim=True)[0] + 1e-10)
        sqr_dist = self.kernel_point[:, :, None, :] - pj[:, None, :, :] # (n, n_k, n_pj, c) 
        sqr_dist = (sqr_dist ** 2).sum(3) # (n, n_k, n_pj)
        corr = torch.exp(-sqr_dist/self.scale)
        if self.first_layer:
            xj = torch.cat([pj, xj], dim=-1)
        x = torch.matmul(corr, xj) # (n, n_k, c)
        x = self.depthwise(x.permute(0, 2, 1).contiguous())
        x = x.squeeze(2)
        return x

# Units
class DiffusionUnit(nn.Module):
    '''
        A slightly more efficient and mathematically equivalent implementation of Diffusion Unit
    '''
    def __init__(self, d_in, d_out, nsample, stride, config):
        super().__init__()
        self.nsample = nsample
        self.pre_linear = nn.Linear(d_in, d_in)
        self.activation = nn.ReLU(inplace=True)
        self.varphi = nn.Sequential(
            nn.Linear(d_in, d_in),
            nn.BatchNorm1d(d_in),
            nn.ReLU(inplace=True)
        )   
    def forward(self, p, u, o):
        N, C = u.shape
        u_t = u
        u = self.pre_linear(u)
        u_n = pointops.queryandgroup(self.nsample, p, p, u, None, o, o, use_xyz=False) # (n,nsample,c)

        nabla_u = u_n - u.unsqueeze(1) # (n, nsample, c)
        nabla_u = self.activation(nabla_u)
        u_tt = self.varphi(nabla_u.mean(1)) + u_t 

        return p, u_tt, o


# Copied from model/architectures.py
class PartsegNetOnehot(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config.d_in_initial
        d_out = config.d_out_initial 
        n_cls = config.num_classes
        nsample = config.nsample
        stride_list = config.strides
        stride = 1
        stride_idx = 0
        d_out_prev = d_in

        # construct encoder 
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        layer_ind_major = 0
        layer_ind_sub = -1
        for block_i, block_name in enumerate(config.architecture):

            layer_ind_sub += 1

            # Detect change to next layer for skip connection
            if np.any([tmp in block_name for tmp in ['strided', 'downsample']]):
                self.encoder_skip_dims.append(d_out_prev)
                self.encoder_skips.append(block_i)

                layer_ind_major += 1
                layer_ind_sub = 0

            # Detect upsampling block to stop
            if 'upsample' in block_name:
                break

            # update feature dim            
            d_in = d_out_prev
            # if subsample
            if 'strided' in block_name or 'downsample' in block_name:
                stride = stride_list[stride_idx]
                stride_idx += 1
                d_out *= 2
            else:
                stride = 1
            
            if 'unit' in block_name:
                nsample = config.nsample
            else:
                nsample = config.nsample_conv

            self.encoder_blocks.append(
                block_decider(block_name)(
                    d_in, d_out, nsample, stride, config
                )
            )

            d_out_prev = d_out


        # construct decoder 
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = []

        # Find first upsampling block
        start_i = 0
        for block_i, block_name in enumerate(config.architecture):
            if 'upsample' in block_name:
                start_i = block_i
                break

        # Loop over consecutive blocks
        layer_ind_major = 0
        layer_ind_sub = -1 
        for block_i, block_name in enumerate(config.architecture[start_i:]):

            layer_ind_sub += 1

            d_in = d_out

            # detect the upsample layer
            if 'upsample' in block_name:

                layer_ind_major += 1
                layer_ind_sub = 0

                self.decoder_upsample.append(block_i)
                
                # if upsample, out_dim / 2 
                d_out = max(d_out // 2, config.decoder_out_dim)                

                self.decoder_blocks.append(
                    block_decider(block_name)(
                        [d_in, self.encoder_skip_dims.pop()], 
                        d_out, nsample, stride, config
                    )
                )
            else:
                # if not upsample, then dim remain same
                self.decoder_blocks.append(
                    block_decider(block_name)(
                        d_in, d_in, nsample, stride, config
                    )
                )
         
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_out+16, d_out), # one-hot labels
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_out, n_cls)
        )        
    
    def forward(self, p, x, o, one_hot):
        # p, x, o, one_hot: (n,3), (n,c), (b), (b,16)
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []

        # encoder
        for block_i, block in enumerate(self.encoder_blocks):
           
            if block_i in self.encoder_skips:
                p_from_encoder.append(p) 
                x_from_encoder.append(x)
                o_from_encoder.append(o)
            p, x, o = block(p, x, o)

                     
        # decoder
        for block_i, block in enumerate(self.decoder_blocks):
           
            if block_i in self.decoder_upsample:
                p_dense = p_from_encoder.pop()
                x_dense = x_from_encoder.pop()
                o_dense = o_from_encoder.pop()
                p, x, o = block(p_dense, x_dense, o_dense, p, x, o) 
            else:
                p, x, o = block(p, x, o) 
            
        # concat one_hot (b,16) to each cloud.
        x = torch.cat([x, one_hot], dim=1) # concat

        # classification     
        x = self.classifier(x)
        return x


class SceneSegNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        d_in = config.d_in_initial
        d_out = config.d_out_initial 
        n_cls = config.num_classes
        nsample = config.nsample
        stride_list = config.strides
        stride = 1
        stride_idx = 0
        d_prev = d_in 
        level = 0

        # construct encoder 
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        layer_ind_major = 0
        layer_ind_sub = -1
        
        for block_i, block_name in enumerate(config.architecture):
            layer_ind_sub += 1
            # Detect change to next layer for skip connection
            if np.any([tmp in block_name for tmp in ['strided', 'downsample']]):
                self.encoder_skip_dims.append(d_prev)
                self.encoder_skips.append(block_i)
                layer_ind_major += 1
                layer_ind_sub = 0 
                level += 1

            # Detect upsampling block to stop
            if 'upsample' in block_name:
                break

            # update feature dim            
            d_in = d_prev 
            # if subsample
            if 'strided' in block_name or 'downsample' in block_name:
                stride = stride_list[stride_idx]
                stride_idx += 1
                d_out *= 2
            else:
                stride = 1
            
            # stack modules
            if 'unit' in block_name:
                nsample = config.nsample
            else:
                nsample = config.nsample_conv
            if level == 0:
                nsample = nsample // 2 

            self.encoder_blocks.append(
                block_decider(block_name)(
                    d_in, d_out, nsample, stride, config 
                )
            )

            d_prev = d_out


        # construct decoder 
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsample = []

        # Find first upsampling block
        start_i = 0
        for block_i, block_name in enumerate(config.architecture):
            if 'upsample' in block_name:
                start_i = block_i
                break

        # Loop over consecutive blocks
        layer_ind_major = 0
        layer_ind_sub = -1 
        for block_i, block_name in enumerate(config.architecture[start_i:]):
            layer_ind_sub += 1

            d_in = d_out

            # detect the upsample layer
            if 'upsample' in block_name:
                layer_ind_major += 1
                layer_ind_sub = 0
                level -= 1

                self.decoder_upsample.append(block_i)
                
                # if upsample, out_dim / 2 
                d_out = max(d_out // 2, config.decoder_out_dim)                

                if 'unit' in block_name:
                    nsample = config.nsample
                else:
                    nsample = config.nsample_conv
                if level == 0:
                    nsample = nsample // 2 

                self.decoder_blocks.append(
                    block_decider(block_name)(
                        [d_in, self.encoder_skip_dims.pop()], 
                        d_out, nsample, stride, config 
                    )
                )
            else:
                # if not upsample, then dim remain same
                if 'unit' in block_name:
                    nsample = config.nsample
                else:
                    nsample = config.nsample_conv
                if level == 0:
                    nsample = nsample // 2 

                self.decoder_blocks.append(
                    block_decider(block_name)(
                        d_in, d_in, nsample, stride, config 
                    )
                )
         
        # classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_out, d_out), 
            nn.BatchNorm1d(d_out),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(d_out, n_cls)
        )        
    
    def forward(self, p, x, o):
        p_from_encoder = []
        x_from_encoder = []
        o_from_encoder = []

        # encoder
        for block_i, block in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                p_from_encoder.append(p) 
                x_from_encoder.append(x)
                o_from_encoder.append(o)
            p, x, o = block(p, x, o)
                     
        # decoder
        for block_i, block in enumerate(self.decoder_blocks):
            if block_i in self.decoder_upsample:
                p_dense = p_from_encoder.pop()
                x_dense = x_from_encoder.pop()
                o_dense = o_from_encoder.pop()
                p, x, o = block(p_dense, x_dense, o_dense, p, x, o) 
            else:
                p, x, o = block(p, x, o) 
            
        # classification     
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    from types import SimpleNamespace
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

    def fake_queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
        if new_xyz is None:
            new_xyz = xyz
        m = new_xyz.shape[0]
        n = xyz.shape[0]
        base = torch.arange(nsample, device=xyz.device).unsqueeze(0).repeat(m, 1) % n
        grouped_xyz = xyz[base.reshape(-1).long(), :].view(m, nsample, 3) - new_xyz.unsqueeze(1)
        grouped_feat = feat[base.reshape(-1).long(), :].view(m, nsample, feat.shape[1])
        if use_xyz:
            return torch.cat((grouped_xyz, grouped_feat), -1).contiguous()
        return grouped_feat.contiguous()

    original_queryandgroup = pointops.queryandgroup
    pointops.queryandgroup = fake_queryandgroup

    try:
        print("Running DiffusionUnit benchmark checks...")
        torch.manual_seed(7)

        cfg = SimpleNamespace(
            convolution='dw_kpconv',
            bottleneck_ratio=None,
            d_in_initial=4,
            d_out_initial=4,
            num_classes=3,
            nsample_conv=4,
            nsample=4,
            strides=[1],
            architecture=['diffusion_unit'],
            decoder_out_dim=4,
        )

        p = torch.randn(8, 3).contiguous()
        x = torch.randn(8, 4, requires_grad=True).contiguous()
        o = torch.cuda.IntTensor([8]) if torch.cuda.is_available() else torch.IntTensor([8])

        try:
            unit = DiffusionUnit(4, 4, 4, 1, cfg)
            p_out, x_out, o_out = unit(p, x, o)
            check("diffusion_unit_functionality", x_out is not None)
            if x_out is None:
                skip_checks(3, "DiffusionUnit returned None")
            else:
                check("diffusion_unit_shape", tuple(x_out.shape) == (8, 4), str(tuple(x_out.shape)))
                check("diffusion_unit_finite", torch.isfinite(x_out).all().item())
                loss = x_out.sum()
                loss.backward(retain_graph=True)
                grad_ok = x.grad is not None and torch.isfinite(x.grad).all().item() and x.grad.abs().sum().item() > 0
                check("diffusion_unit_gradient", grad_ok)
        except Exception as exc:
            traceback.print_exc()
            skip_checks(4, f"DiffusionUnit checks raised {type(exc).__name__}: {exc}")

        try:
            model = SceneSegNet(cfg)
            x_model = torch.randn(8, 4, requires_grad=True).contiguous()
            y = model(p, x_model, o)
            check("scene_seg_functionality", y is not None)
            if y is None:
                skip_checks(3, "SceneSegNet returned None")
            else:
                check("scene_seg_shape", tuple(y.shape) == (8, 3), str(tuple(y.shape)))
                check("scene_seg_finite", torch.isfinite(y).all().item())
                y.sum().backward()
                model_grad_ok = x_model.grad is not None and torch.isfinite(x_model.grad).all().item() and x_model.grad.abs().sum().item() > 0
                check("scene_seg_gradient", model_grad_ok)
        except Exception as exc:
            traceback.print_exc()
            skip_checks(4, f"SceneSegNet checks raised {type(exc).__name__}: {exc}")

        try:
            original_np_load = np.load
            np.load = lambda *args, **kwargs: np.zeros((15, 3), dtype=np.float32)
            try:
                conv = DepthwiseKPConv(4, 4, cfg)
            finally:
                np.load = original_np_load
            pj = torch.randn(8, 4, 3).contiguous()
            x_center = torch.randn(8, 4).contiguous()
            x_neighbor = torch.randn(8, 4, 4).contiguous()
            z = conv(p, pj, x_center, x_neighbor)
            check("kpconv_functionality", z is not None)
            if z is None:
                skip_checks(2, "DepthwiseKPConv returned None")
            else:
                check("kpconv_shape", tuple(z.shape) == (8, 4), str(tuple(z.shape)))
                check("kpconv_finite", torch.isfinite(z).all().item())
        except Exception as exc:
            traceback.print_exc()
            skip_checks(3, f"DepthwiseKPConv checks raised {type(exc).__name__}: {exc}")

    finally:
        pointops.queryandgroup = original_queryandgroup

    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    if failed != 0:
        raise SystemExit(1)
