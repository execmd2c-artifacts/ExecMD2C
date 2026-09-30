"""Ground-truth core model components for the RawGAT-ST benchmark.

This file consolidates only the architecture-level components from the default
RawGAT-ST multiplicative-fusion model: Sinc-style raw waveform convolution,
residual spectro-temporal encoders, graph attention, graph pooling, and the
top-level anti-spoofing classifier.
"""

from collections import OrderedDict
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- [Original file: model.py] ---

class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

    def forward(self, x):
        '''
        x   :(#bs, #node, #dim)
        '''
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        # apply activation
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x):
        '''
        Calculates pairwise multiplication of nodes.
        - for attention map
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, #dim)
        '''

        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)

        return x * x_mirror

    def _derive_att_map(self, x):
        '''
        x           :(#bs, #node, #dim)
        out_shape   :(#bs, #node, #node, 1)
        '''
        att_map = self._pairwise_mul_nodes(x)
        # size: (#bs, #node, #node, #dim_out)
        att_map = torch.tanh(self.att_proj(att_map))
        # size: (#bs, #node, #node, 1)
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = F.softmax(att_map, dim=-2)

        return att_map

    def _project(self, x, att_map):
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)

        return x1 + x2

    def _apply_BN(self, x):
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)

        return x

    def _init_new_params(self, *size):
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class Pool(nn.Module):

    def __init__(self, k:float, in_dim:int, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim=in_dim

    def forward(self, h):
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)  
        new_h = self.top_k_graph(scores, h, self.k)

        return new_h


    def top_k_graph(self,scores,h, k):
        """
        args
        ====
        scores: attention-based weights (#bs,#node,1)
        h: graph (#bs,#node,#dim)
        k: ratio of remaining nodes, (float)
         
        """
        num_nodes = h.shape[1]
        batch_size=h.shape[0]

        # first reflect the weights and then rank them
        H= h*scores
        _, idx = torch.topk(scores, max(2, int(k*num_nodes)),dim=1)
        new_g=[]

        for i in range(batch_size):
            new_g.append(H[i,idx[i][:int(len(idx[i]))],:])
            
        new_g = torch.stack(new_g,dim=0)
         
        return new_g


class CONV(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)


    def __init__(self, device,out_channels, kernel_size, in_channels=1,sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1,mask=False):
        super(CONV,self).__init__()
        if in_channels != 1:
            
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)
       
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate=sample_rate

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.device=device   
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask=mask
        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')
        self.device=device
        
        
        NFFT = 512
        f=int(self.sample_rate/2)*np.linspace(0,1,int(NFFT/2)+1)
        fmel=self.to_mel(f)
        fmelmax=np.max(fmel)
        fmelmin=np.min(fmel)
        filbandwidthsmel=np.linspace(fmelmin,fmelmax,self.out_channels+1)
        filbandwidthsf=self.to_hz(filbandwidthsmel)
        
        self.mel=filbandwidthsf
        self.hsupp=torch.arange(-(self.kernel_size-1)/2, (self.kernel_size-1)/2+1)
        self.band_pass=torch.zeros(self.out_channels,self.kernel_size)
    
       
        
    def forward(self,x,mask=False):
        for i in range(len(self.mel)-1):
            fmin=self.mel[i]
            fmax=self.mel[i+1]
            hHigh=(2*fmax/self.sample_rate)*np.sinc(2*fmax*self.hsupp/self.sample_rate)
            hLow=(2*fmin/self.sample_rate)*np.sinc(2*fmin*self.hsupp/self.sample_rate)
            hideal=hHigh-hLow
            
            self.band_pass[i,:]=Tensor(np.hamming(self.kernel_size))*Tensor(hideal)
        
        band_pass_filter=self.band_pass.to(self.device)

        # Frequency masking: We randomly mask (1/5)th of no. of sinc filters channels (70)
        if (mask==True):
            for i1 in range(1):
                A=np.random.uniform(0,14) 
                A=int(A)
                A0=random.randint(0,band_pass_filter.shape[0]-A)
                band_pass_filter[A0:A0+A,:]=0
        else:
            band_pass_filter=band_pass_filter
        
        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)
        
        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first = False):
        super(Residual_block, self).__init__()
        self.first = first
        
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features = nb_filts[0])
            self.conv1 = nn.Conv2d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			kernel_size = (2,3),
			padding = (1,1),
			stride = 1)
        self.selu = nn.SELU(inplace=True)
        
        
        
        self.conv_1 = nn.Conv2d(in_channels = 1,
			out_channels = nb_filts[1],
			kernel_size = (2,3),
			padding = (1,1),
			stride = 1)
        self.bn2 = nn.BatchNorm2d(num_features = nb_filts[1])
        self.conv2 = nn.Conv2d(in_channels = nb_filts[1],
			out_channels = nb_filts[1],
			
			kernel_size = (2,3),
                        padding = (0,1),
			stride = 1)
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels = nb_filts[0],
				out_channels = nb_filts[1],
				padding = (0,1),
				kernel_size = (1,3),
				stride = 1)
            
        else:
            self.downsample = False
        self.mp = nn.MaxPool2d((1,3))

        
    def forward(self, x):
        identity = x
        
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
            out=self.conv1(x)
        else:
            x=x
            out = self.conv_1(x)
            
        
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        return out


class RawGAT_ST(nn.Module):
    def __init__(self, d_args, device):
        super(RawGAT_ST, self).__init__()
        self.device=device
        
        '''
        Sinc conv. layer
        '''
        self.conv_time=CONV(device=self.device,
			out_channels = d_args['out_channels'],
			kernel_size = d_args['first_conv'],
                        in_channels = d_args['in_channels']
        )
        
        self.first_bn = nn.BatchNorm2d(num_features = 1)
        
        self.selu = nn.SELU(inplace=True)
        
        # Note that here you can also use only one encoder to reduce the network parameters which is jsut half of the 0.44M (mentioned in the paper). I was doing some subband analysis and forget to remove the use of two encoders.  I also checked with one encoder and found same results. 
        
        self.encoder1=nn.Sequential(
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][1], first = True)),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][1])),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][2])),
                        
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][3])),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][3])),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][3]))
        )


        self.encoder2=nn.Sequential(
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][1], first = True)),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][1])),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][2])),
                        
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][3])),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][3])),
                        nn.Sequential(Residual_block(nb_filts = d_args['filts'][3]))
        )

        # Graph attention and pooling layer for Spectral-RawGAT
        self.GAT_layer1=GraphAttentionLayer(d_args['filts'][-1][-1],32)
        self.pool1=Pool(0.64, 32, 0.3)

        # Graph attention and pooling layer for Temporal-RawGAT
        self.GAT_layer2=GraphAttentionLayer(d_args['filts'][-1][-1],32)
        self.pool2=Pool(0.81, 32, 0.3)

        # Graph attention and pooling layer for Spectro-Temporal RawGAT
        self.GAT_layer3=GraphAttentionLayer(32,16)
        self.pool3=Pool(0.64, 16, 0.3)
        
        #Projection layers 
        self.proj1 = nn.Linear(14,12)
        self.proj2 = nn.Linear(23,12)
        self.proj = nn.Linear(16,1)

        # classifier layer with nclass=2 and 7 is number of nodes remaining after pooling layer in Spectro-temporal graph attention layer 
        self.proj_node = nn.Linear(7,2)
        
        
    def forward(self, x, Freq_aug=False):
        """
        x= (#bs,samples)
        """

        #follow sincNet recipe

        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        
        
        x=x.view(nb_samp,1,len_seq)
       
        # Freq masking during training only

        if (Freq_aug==True):
            x=self.conv_time(x,mask=True)  #(#bs,sinc_filt(70),64472)
            
        else:
            x=self.conv_time(x,mask=False)
        
        """
        Different with the our RawNet2 model, we interpret the output of sinc-convolution layer as 2-dimensional image with one channel (like 2-D representation).
        """
        x = x.unsqueeze(dim=1)  # 2-D (#bs,1,sinc-filt(70),64472)
        
        x = F.max_pool2d(torch.abs(x), (3,3))  #[#bs, C(1),F(23),T(21490)]
        

        x = self.first_bn(x)
        x = self.selu(x)
        
        # encoder structure for spectral GAT
        e1=self.encoder1(x)            # [#bs, C(64), F(23), T(29)]
        
        # max-pooling along time with absolute value  (Attention in spectral part)
        x_max,_=torch.max(torch.abs(e1),dim=3)  #[#bs, C(64), F(23)]
        
        x_gat1=self.GAT_layer1(x_max.transpose(1,2))  #(#bs,#node(F),feat_dim(C)) --> [#bs, 23, 32]
        
        x_pool1=self.pool1(x_gat1)
        out1=self.proj1(x_pool1.transpose(1,3))
        out1=out1.view(out1.shape[0],out1.shape[1],out1.shape[3]) #(#bs,feat_dim,#node) --> [#bs, 32, 12]
        


        # encoder structure for temporal GAT
        e2=self.encoder2(x)   #[#bs, C(64), F(23), T(29)]
        
        x_max2,_=torch.max(torch.abs(e2),dim=2) # max along frequency  #[#bs, C(64), T(29)]
        
        
        x_gat2=self.GAT_layer2(x_max2.transpose(1,2)) #(#bs,#node(T),feat_dim(C)) --> #[#bs, 29, 32]
       
        
        x_pool2=self.pool2(x_gat2)
        out2=self.proj2(x_pool2.transpose(1,3))
        out2=out2.view(out2.shape[0],out2.shape[1],out2.shape[3]) #(#bs,feat_dim,#node)  #[#bs, 32, 12]
        

        # To fuse both spectral (out1) and temporal (out2) graphs using element-wise multiplication  (graph combination)
        out_gat=torch.mul(out1,out2)  #(#bs,feat_dim,#node) -->  #[#bs, 32, 12]
        
        # Give fuse GAT output (out_gat) to Spectro-temporal GAT layer
        x_gat3=self.GAT_layer3(out_gat.transpose(1,2))  #(#bs,#node,feat_out_dim) --> #[#bs, 12, 16]
        
        x_pool3=self.pool3(x_gat3)
        
        out_proj=self.proj(x_pool3).flatten(1)  #(#bs,#nodes) --> [#bs, 7]
        
        output=self.proj_node(out_proj)  #(#bs, output node(no. of classes)) ---> [#bs,2]
        
        return output

        


    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
				first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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

    print("[Test 1/5] GraphAttentionLayer._pairwise_mul_nodes")
    try:
        layer = GraphAttentionLayer(4, 3)
        x = torch.randn(2, 5, 4)
        out = layer._pairwise_mul_nodes(x)
        check("Pairwise output not None", out is not None)
        if out is not None:
            check("Pairwise output shape", tuple(out.shape) == (2, 5, 5, 4), str(tuple(out.shape)))
            check("Pairwise finite output", torch.isfinite(out).all().item())
            check("Pairwise diagonal squares nodes", torch.allclose(out[:, torch.arange(5), torch.arange(5), :], x * x))
            check("Pairwise symmetric products", torch.allclose(out, out.transpose(1, 2)))
        else:
            skip_checks(4, "pairwise output was None")
    except Exception as exc:
        skip_checks(5, f"GraphAttentionLayer._pairwise_mul_nodes raised {type(exc).__name__}: {exc}")

    print("[Test 2/5] GraphAttentionLayer._derive_att_map")
    try:
        layer = GraphAttentionLayer(4, 3)
        layer.eval()
        x = torch.randn(2, 5, 4)
        att_map = layer._derive_att_map(x)
        check("Attention map not None", att_map is not None)
        if att_map is not None:
            check("Attention map shape", tuple(att_map.shape) == (2, 5, 5, 1), str(tuple(att_map.shape)))
            check("Attention map finite", torch.isfinite(att_map).all().item())
            check("Attention map normalized over source nodes", torch.allclose(att_map.sum(dim=-2), torch.ones(2, 5, 1), atol=1e-5))
            check("Attention map nonnegative", (att_map >= 0).all().item())
        else:
            skip_checks(4, "attention map was None")
    except Exception as exc:
        skip_checks(5, f"GraphAttentionLayer._derive_att_map raised {type(exc).__name__}: {exc}")

    print("[Test 3/5] Pool.top_k_graph")
    try:
        pool = Pool(0.5, 3, 0.0)
        scores = torch.tensor([[[0.1], [0.9], [0.2], [0.8]], [[0.4], [0.3], [0.7], [0.6]]])
        h = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
        out = pool.top_k_graph(scores, h, 0.5)
        check("Pool output not None", out is not None)
        if out is not None:
            check("Pool output shape", tuple(out.shape) == (2, 2, 1, 3), str(tuple(out.shape)))
            check("Pool output finite", torch.isfinite(out).all().item())
            expected0 = h[0, torch.tensor([1, 3])] * scores[0, torch.tensor([1, 3])]
            check("Pool keeps top weighted nodes batch0", torch.allclose(out[0, :, 0, :], expected0))
            check("Pool keeps at least two nodes", out.size(1) >= 2)
        else:
            skip_checks(4, "pool output was None")
    except Exception as exc:
        skip_checks(5, f"Pool.top_k_graph raised {type(exc).__name__}: {exc}")

    print("[Test 4/5] CONV.forward")
    try:
        conv = CONV(device="cpu", out_channels=4, kernel_size=8, in_channels=1, sample_rate=16000)
        x = torch.randn(2, 1, 64)
        out = conv(x, mask=False)
        check("CONV output not None", out is not None)
        if out is not None:
            check("CONV output shape", tuple(out.shape) == (2, 4, 56), str(tuple(out.shape)))
            check("CONV finite output", torch.isfinite(out).all().item())
            check("CONV odd kernel enforced", conv.kernel_size == 9, str(conv.kernel_size))
            check("CONV filter shape", tuple(conv.filters.shape) == (4, 1, 9), str(tuple(conv.filters.shape)))
        else:
            skip_checks(4, "CONV output was None")
    except Exception as exc:
        skip_checks(5, f"CONV.forward raised {type(exc).__name__}: {exc}")

    print("[Test 5/5] RawGAT_ST.forward")
    try:
        d_args = {
            "out_channels": 70,
            "first_conv": 128,
            "in_channels": 1,
            "filts": [32, [32, 32], [32, 64], [64, 64]],
        }
        model = RawGAT_ST(d_args, device="cpu")
        model.eval()
        x = torch.randn(2, 64600)
        with torch.no_grad():
            out = model(x, Freq_aug=False)
        check("RawGAT_ST output not None", out is not None)
        if out is not None:
            check("RawGAT_ST output shape", tuple(out.shape) == (2, 2), str(tuple(out.shape)))
            check("RawGAT_ST finite logits", torch.isfinite(out).all().item())
            check("RawGAT_ST class dimension", out.size(-1) == 2, str(out.size(-1)))
            check("RawGAT_ST uses multiplicative fusion width", model.GAT_layer3.att_proj.in_features == 32, str(model.GAT_layer3.att_proj.in_features))
        else:
            skip_checks(4, "RawGAT_ST.forward returned None")
    except Exception as exc:
        skip_checks(5, f"RawGAT_ST.forward raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
