from xxlimited import new
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class single_scale(nn.Module):

    def __init__(self, d_model, out_dim, patchsize):
        super(single_scale, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=out_dim, kernel_size=(1,patchsize), stride=(1,patchsize), bias=False),
            # nn.GELU(),
            # nn.Conv2d(in_channels=256, out_channels=out_dim, kernel_size=(1,1), stride=(1,1), bias=False),
        )
    
    def forward(self, x):

        '''
        input
        x: (B,N,T = 1 week,F)

        return 
        (B,N,T/patchsize,F)
        '''

        # apply convolution
        x = x.permute(0,3,1,2)    # (B, F, N, T)
        x = self.conv(x).permute(0,2,3,1)    # (B,N,T,out_dim)

        return x

class full_attns(nn.Module):

    def __init__(self, in_dim1, in_dim2, adj, hidden=4, num_head=4):
        super(full_attns, self).__init__()

        assert hidden % num_head == 0, 'hidden should be the multiplier of num_head'
        self.W_mapping1 = nn.Linear(in_dim2, 1, bias=False)
        self.W_mapping2 = nn.Linear(in_dim1, hidden, bias=False)
        self.A_mapping = nn.Linear(2*int(hidden/num_head)+1, 1, bias=False)

        self.num_heads = num_head
        self.adj = adj.unsqueeze(-1).unsqueeze(-1).repeat(1,1,self.num_heads,1)    # (N,N,H,1)
        self.activate = nn.GELU()


    def forward(self, x):
        """
        [TODO] Compute adjacency-aware multi-head spatial attention scores.

        Input:
            x: (batch, nodes, features, timesteps) - node time-series features.

        Output:
            (batch, num_heads, nodes, nodes) - unnormalized spatial attention
            scores for every node pair and attention head.

"""
        pass

class GCN(nn.Module):

    def __init__(self, adj, out_len, configs):
        super(GCN, self).__init__()

        feature_dim = configs.d_model
        num_head = configs.n_heads

        assert feature_dim % num_head == 0, 'feature dimension should be multiplier of number of heads'
        self.attn_cal = full_attns(in_dim1=feature_dim, in_dim2=out_len, adj=adj, hidden=32, num_head=num_head)
        self.W = nn.Linear(feature_dim, int(feature_dim/num_head), bias=False)
        
        self.num_head = num_head
        self.adj = adj.unsqueeze(0).repeat(self.num_head,1,1)    # (H,N,N)
        self.zero_vec = -9e15 * torch.ones_like(self.adj)   # (H,N,N)

        self.out_mapping = nn.Linear(int(feature_dim/num_head), feature_dim)

        self.activate = nn.GELU()
    
    def forward(self, x):
        """
        [TODO] Apply masked spatial graph aggregation with explainable attention.

        Input:
            x: (batch, nodes, timesteps, features) - traffic node features.

        Output:
            x_new: (batch, nodes, timesteps, features / num_heads) - aggregated
                node features.
            adj2: (batch, num_heads, nodes, nodes) - normalized spatial
                attention weights.

"""
        pass

class AutoCorrelation(nn.Module):
    """
    
    """
    def __init__(self, device, out_len, patch, query_len, configs):
        super(AutoCorrelation, self).__init__()

        # define query length
        self.query_length = query_len

        # define the output len
        self.out_len = out_len

        # define predict length
        self.pred_len = configs.pred_len

        # define time steps
        self.time_steps = configs.time_steps

        # assert self.pred_len % self.out_len == 0, 'prediction length must be the multiplier of output length'

        # define the device
        self.device = device

        # defien patch size
        self.patch = patch
    
        # define top_k
        self.topk = configs.topk

        # define number of head
        self.num_head = configs.n_heads

        # define feature dimension
        feature_dim = configs.d_model

        # define a zeros tensor
        self.zeros = torch.zeros(configs.num_nodes, configs.time_steps, configs.d_model).to(self.device)

        # define module for "cal_QKV"
        self.Q_mapping = single_scale(d_model=feature_dim+1, out_dim=self.num_head, patchsize=patch)
        self.K_mapping = single_scale(d_model=feature_dim+1, out_dim=self.num_head, patchsize=patch)
        self.V_mapping = single_scale(d_model=feature_dim+1, out_dim=int(feature_dim/self.num_head), patchsize=1)
        self.out_mapping = nn.Linear(int(feature_dim/self.num_head), feature_dim)

        # define initial index
        self.init_index = torch.arange(self.out_len).to(self.device).unsqueeze(0).unsqueeze(0)    # (1,1,out_len)
        self.init_index = self.init_index.repeat(configs.num_nodes, int(feature_dim/self.num_head), 1)    # (N,F,out_len)

    def time_delay_agg_full(self, values, corr):
        """
        [TODO] Aggregate values using top-k auto-correlation time delays.

        Input:
            values: (batch, nodes, timesteps, feature_per_head) - value series.
            corr: (batch, nodes, timesteps, num_heads) - delay correlations.

        Output:
            output: (batch, nodes, feature_per_head, out_len) - delayed pattern
                aggregation.
            delay: (batch, nodes, num_heads, topk) - selected delays.
            tmp_corr: (batch, nodes, num_heads, topk) - normalized delay scores.

"""
        pass

    def cal_QKV(self, Q_in, K_in, V_in, t_stamp):
        """
        [TODO] Construct temporal query/key/value tensors and FFT correlations.

        Input:
            Q_in: (batch, nodes, timesteps, features) - query sequence.
            K_in: (batch, nodes, timesteps, features) - key sequence.
            V_in: (batch, nodes, timesteps, features) - value sequence.
            t_stamp: (batch, long_timesteps) - timestamp signal.

        Output:
            values: (batch, nodes, timesteps, features) - value tensor.
            corr: (batch, nodes, timesteps, num_heads) - temporal correlations.

"""
        pass

    def forward(self, Q_in, K_in, V_in, t):
        """
        [TODO] Run one AutoCorrelation forecasting cell.

        Input:
            Q_in: (batch, nodes, timesteps, features) - query sequence.
            K_in: (batch, nodes, timesteps, features) - key sequence.
            V_in: (batch, nodes, timesteps, features) - value sequence.
            t: (batch, long_timesteps) - timestamp sequence.

        Output:
            output: (batch, nodes, out_len, features) - aggregated forecast.
            delay: (batch, nodes, num_heads, topk) - selected delays.
            delay_score: (batch, nodes, num_heads, topk) - delay weights.

"""
        pass

class patch_atten(nn.Module):

    def __init__(self, DEVICE, adj, configs):
        super(patch_atten, self).__init__()

        self.patch1 = 3
        self.patch2 = 4
        self.patch3 = 3
        self.patch4 = 4
        feature_dim = configs.d_model
        self.pred_len = configs.pred_len
        
        self.CAM1_mapping = nn.Sequential(nn.Linear(feature_dim * self.patch1, 32), nn.GELU(), nn.Linear(32,self.patch1))
        self.CAM2_mapping = nn.Sequential(nn.Linear(feature_dim * self.patch2, 32), nn.GELU(), nn.Linear(32,self.patch2))
        self.CAM3_mapping = nn.Sequential(nn.Linear(feature_dim * self.patch3, 32), nn.GELU(), nn.Linear(32,self.patch3))
        self.CAM4_mapping = nn.Sequential(nn.Linear(feature_dim * self.patch4, 32), nn.GELU(), nn.Linear(32,self.patch4))

        # Autocorrelation
        self.cell1 = AutoCorrelation(device=DEVICE, out_len=self.pred_len,  patch=1, query_len=int(configs.time_steps/2), configs=configs)
        self.cell2 = AutoCorrelation(device=DEVICE, out_len=int(self.pred_len/3),  patch=1, query_len=int(configs.time_steps/6), configs=configs)
        self.cell3 = AutoCorrelation(device=DEVICE, out_len=int(self.pred_len/12), patch=1, query_len=int(configs.time_steps/24), configs=configs)
        self.cell4 = AutoCorrelation(device=DEVICE, out_len=int(self.pred_len/36), patch=1, query_len=int(configs.time_steps/72), configs=configs)
        self.cell5 = AutoCorrelation(device=DEVICE, out_len=int(self.pred_len/144), patch=1, query_len=int(configs.time_steps/288), configs=configs)

        
    def patch_agg(self, x, mapping_function, patch_size):
        """
        [TODO] Aggregate non-overlapping temporal patches with CAM weights.

        Input:
            x: (batch, nodes, timesteps, features) - temporal features.
            mapping_function: module producing one score per patch position.
            patch_size: int - number of timesteps per patch.

        Output:
            (batch, nodes, timesteps / patch_size, features) - weighted patch
            summaries.

"""
        pass

    def patch_forward(self, x):
        """
        [TODO] Build the five-level temporal patch pyramid.

        Input:
            x: (batch, nodes, timesteps, features) - fine-scale sequence.

        Output:
            List of five tensors: original sequence followed by four coarser
            patch-aggregated scales.

"""
        pass

    def forward(self, x, t):
        """
        [TODO] Forecast each pyramid scale and align all outputs to pred_len.

        Input:
            x: (batch, nodes, timesteps, features) - temporal feature sequence.
            t: (batch, long_timesteps) - timestamp sequence.

        Output:
            outputs: list of five (batch, nodes, pred_len, features) tensors.
            delays: list of five selected-delay tensors.
            delay_scores: list of five selected-delay weight tensors.

"""
        pass
        




class Autoformer(nn.Module):
    
    def __init__(self, DEVICE, adj, configs):
        super(Autoformer, self).__init__()

        # used in the forward function
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.input_lenght = configs.time_steps

        # used in define the layers
        self.out_len_list = configs.out_len
        patch_list = configs.patch_list

        # define a zeros tensor
        self.zeros = torch.zeros(configs.num_nodes, self.pred_len, configs.d_model).to(DEVICE)

        # spatial encoder  
        self.GCN_agg1 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)
        self.GCN_agg2 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)
        self.GCN_agg3 = GCN(adj=adj, out_len=configs.time_steps, configs=configs)

        # patch attention
        self.patch_att1 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att2 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att3 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)
        self.patch_att4 = patch_atten(DEVICE=DEVICE, adj=adj, configs=configs)

        self.relu_fcnn = nn.Sequential(nn.Linear(configs.d_model * 21, 512), nn.Tanh(), nn.Linear(512,512), nn.Tanh(), nn.Linear(512,21))

    def forward(self, x_enc, t):
        """
        [TODO] Run the explainable graph pyramid Autoformer backbone.

        Input:
            x_enc: (batch, nodes, timesteps, features) - encoded traffic
                history.
            t: (batch, long_timesteps) - timestamp sequence.

        Output:
            output: (batch, nodes, pred_len, features) - fused multiscale
                forecast features.
            explanations: list containing fusion scores, spatial attentions,
                temporal delays, and temporal delay scores.

"""
        pass


class Model(nn.Module):
    
    def __init__(self, adj, configs, DEVICE):
        super(Model, self).__init__()

        # define model parameters
        self.pred_len = configs.pred_len
        self.d_model = configs.d_model
        self.out_len = configs.out_len
        self.patch_list = configs.patch_list
        self.topk = configs.topk
        self.encoder_length = configs.time_steps
        self.n_head = configs.n_heads
        self.time_steps = configs.time_steps
        self.num_nodes = configs.num_nodes

        # define t_embedding neural network
        self.encoder = nn.Sequential(nn.Linear(1, 256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256, configs.d_model))

        # define the Autoformer
        self.A1 = Autoformer(DEVICE=DEVICE, adj=adj, configs=configs)

        # out projection
        self.decoder = nn.Sequential(nn.Linear(configs.d_model, 512), nn.Tanh(), nn.Linear(512,256), nn.Tanh(), nn.Linear(256,256), nn.Tanh(), nn.Linear(256, 1))
        
    def forward(self, x_enc, t):
        """
        [TODO] Run the top-level traffic forecasting model.

        Input:
            x_enc: (batch, timesteps, nodes) - raw traffic history.
            t: (batch, long_timesteps) - timestamp sequence.

        Output:
            out: (batch, pred_len, nodes) - traffic forecast.
            explains: explanation tensors returned by the Graph Autoformer.

"""
        pass


if __name__ == "__main__":
    from types import SimpleNamespace

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
    print("Explainable Graph Pyramid Autoformer benchmark")
    print("=" * 70)

    device = torch.device("cpu")

    def make_cfg(pred_len=12, time_steps=24, d_model=4, n_heads=1, num_nodes=3, topk=2):
        return SimpleNamespace(
            pred_len=pred_len,
            time_steps=time_steps,
            d_model=d_model,
            n_heads=n_heads,
            num_nodes=num_nodes,
            topk=topk,
            out_len=[time_steps, time_steps, pred_len],
            patch_list=[12, 12, 3, 1],
        )

    adj = torch.tensor(
        [[1.0, 1.0, 0.0],
         [1.0, 1.0, 1.0],
         [0.0, 1.0, 1.0]],
        device=device
    )

    # Test group 1: full_attns spatial score construction
    try:
        attn = full_attns(in_dim1=4, in_dim2=12, adj=adj, hidden=4, num_head=1).to(device)
        x = torch.randn(2, 3, 4, 12, device=device)
        scores = attn(x)
        check("full_attns output not None", scores is not None)
        if scores is not None:
            check("full_attns score shape", tuple(scores.shape) == (2, 1, 3, 3), f"got {tuple(scores.shape)}")
            check("full_attns scores finite", torch.isfinite(scores).all().item())
            check("full_attns adjacency buffer shape", tuple(attn.adj.shape) == (3, 3, 1, 1))
            check("full_attns has learnable mappings", sum(p.numel() for p in attn.parameters()) > 0)
        else:
            skip_checks(4, "full_attns returned None")
    except Exception as exc:
        skip_checks(5, f"full_attns raised {type(exc).__name__}: {exc}")

    # Test group 2: GCN masked spatial aggregation
    try:
        cfg = make_cfg(pred_len=12, time_steps=24, d_model=4, n_heads=1)
        gcn = GCN(adj=adj, out_len=24, configs=cfg).to(device)
        x = torch.randn(2, 3, 24, 4, device=device)
        out, adj2 = gcn(x)
        check("GCN output not None", out is not None and adj2 is not None)
        if out is not None and adj2 is not None:
            check("GCN feature shape", tuple(out.shape) == (2, 3, 24, 4), f"got {tuple(out.shape)}")
            check("GCN attention shape", tuple(adj2.shape) == (2, 1, 3, 3), f"got {tuple(adj2.shape)}")
            check("GCN output finite", torch.isfinite(out).all().item())
            check("GCN attention row stochastic", torch.allclose(adj2.sum(dim=-1), torch.ones_like(adj2.sum(dim=-1)), atol=1e-5))
            masked_mass = adj2[:, :, adj == 0]
            check("GCN masks absent edges", torch.all(masked_mass < 1e-6).item())
        else:
            skip_checks(5, "GCN returned None")
    except Exception as exc:
        skip_checks(6, f"GCN raised {type(exc).__name__}: {exc}")

    # Test group 3: AutoCorrelation QKV and time-delay aggregation
    try:
        cfg = make_cfg(pred_len=12, time_steps=24, d_model=4, n_heads=1, topk=2)
        autocorr = AutoCorrelation(device=device, out_len=12, patch=1, query_len=6, configs=cfg).to(device)
        q = torch.randn(2, 3, 24, 4, device=device)
        k = torch.randn(2, 3, 24, 4, device=device)
        v = torch.randn(2, 3, 24, 4, device=device)
        t = torch.linspace(0, 1, 24, device=device).unsqueeze(0).repeat(2, 1)
        values, corr = autocorr.cal_QKV(q, k, v, t)
        check("AutoCorrelation cal_QKV not None", values is not None and corr is not None)
        if values is not None and corr is not None:
            check("AutoCorrelation values shape", tuple(values.shape) == (2, 3, 24, 4), f"got {tuple(values.shape)}")
            check("AutoCorrelation corr shape", tuple(corr.shape) == (2, 3, 24, 4), f"got {tuple(corr.shape)}")
            check("AutoCorrelation corr finite", torch.isfinite(corr).all().item())
            agg, delay, delay_score = autocorr.time_delay_agg_full(values, corr)
            check("AutoCorrelation agg shape", tuple(agg.shape) == (2, 3, 4, 12), f"got {tuple(agg.shape)}")
            check("AutoCorrelation delay shape", tuple(delay.shape) == (2, 3, 4, 2), f"got {tuple(delay.shape)}")
            check("AutoCorrelation delay weights sum", torch.allclose(delay_score.sum(dim=-1), torch.ones_like(delay_score.sum(dim=-1)), atol=1e-5))
            y, d, ds = autocorr(q, k, v, t)
            check("AutoCorrelation forward shape", tuple(y.shape) == (2, 3, 12, 4), f"got {tuple(y.shape)}")
            check("AutoCorrelation forward finite", torch.isfinite(y).all().item())
        else:
            skip_checks(8, "AutoCorrelation cal_QKV returned None")
    except Exception as exc:
        skip_checks(9, f"AutoCorrelation raised {type(exc).__name__}: {exc}")

    # Test group 4: patch pyramid attention
    try:
        cfg = make_cfg(pred_len=144, time_steps=288, d_model=4, n_heads=1, topk=2)
        patcher = patch_atten(DEVICE=device, adj=adj, configs=cfg).to(device)
        x = torch.randn(1, 3, 288, 4, device=device)
        t = torch.linspace(0, 1, 288, device=device).unsqueeze(0)
        pyramid = patcher.patch_forward(x)
        check("patch_forward returns five scales", isinstance(pyramid, list) and len(pyramid) == 5)
        if isinstance(pyramid, list) and len(pyramid) == 5:
            check("patch scale 0 shape", tuple(pyramid[0].shape) == (1, 3, 288, 4), f"got {tuple(pyramid[0].shape)}")
            check("patch scale 4 shape", tuple(pyramid[4].shape) == (1, 3, 2, 4), f"got {tuple(pyramid[4].shape)}")
            outputs, delays, delay_scores = patcher(x, t)
            check("patch_atten output count", len(outputs) == 5 and len(delays) == 5 and len(delay_scores) == 5)
            check("patch_atten first output shape", tuple(outputs[0].shape) == (1, 3, 144, 4), f"got {tuple(outputs[0].shape)}")
            check("patch_atten repeated output shape", tuple(outputs[-1].shape) == (1, 3, 144, 4), f"got {tuple(outputs[-1].shape)}")
            check("patch_atten delay score finite", all(torch.isfinite(item).all().item() for item in delay_scores))
        else:
            skip_checks(6, "patch_forward did not return five scales")
    except Exception as exc:
        skip_checks(7, f"patch_atten raised {type(exc).__name__}: {exc}")

    # Test group 5: Autoformer fusion and top-level Model
    try:
        cfg = make_cfg(pred_len=144, time_steps=288, d_model=4, n_heads=1, topk=2)
        autoformer = Autoformer(DEVICE=device, adj=adj, configs=cfg).to(device)
        x_enc = torch.randn(1, 3, 288, 4, device=device)
        t = torch.linspace(0, 1, 288, device=device).unsqueeze(0)
        auto_out, explains = autoformer(x_enc, t)
        check("Autoformer output not None", auto_out is not None and explains is not None)
        if auto_out is not None and explains is not None:
            check("Autoformer output shape", tuple(auto_out.shape) == (1, 3, 144, 4), f"got {tuple(auto_out.shape)}")
            check("Autoformer output finite", torch.isfinite(auto_out).all().item())
            check("Autoformer explain groups", len(explains) == 4)
            check("Autoformer spatial attention count", len(explains[1]) == 3)
            check("Autoformer temporal delay count", len(explains[2]) == 20 and len(explains[3]) == 20)
        else:
            skip_checks(5, "Autoformer returned None")

        model = Model(adj=adj, configs=cfg, DEVICE=device).to(device)
        raw_x = torch.randn(1, 288, 3, device=device)
        model_out, model_explains = model(raw_x, t)
        check("Model output not None", model_out is not None and model_explains is not None)
        if model_out is not None and model_explains is not None:
            check("Model output shape", tuple(model_out.shape) == (1, 144, 3), f"got {tuple(model_out.shape)}")
            check("Model output finite", torch.isfinite(model_out).all().item())
            check("Model exposes fusion weights", tuple(model_explains[0].shape) == (1, 3, 144, 21))
        else:
            skip_checks(3, "Model returned None")
    except Exception as exc:
        skip_checks(10, f"Autoformer/Model raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The benchmark implementation is complete.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
