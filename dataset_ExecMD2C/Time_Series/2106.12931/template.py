# ============================================================
# ground_truth.py - STGODE Core Model Components (Self-contained)
# Source: odegcn.py, model.py
#
# Contains only the spatial-temporal ODE graph model and direct dependencies.
# Training, data loading, adjacency preprocessing, and evaluation are excluded.
# ============================================================

import torch
from torch import nn
import torch.nn.functional as F
import math

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# --- [Original file: odegcn.py] ---
class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        """
        [TODO] Compute the STGODE derivative from graph, feature, temporal, and initial-state terms.

        Input:
            t: scalar ODE integration time.
            x: (batch, num_nodes, num_timesteps, feature_dim) - current continuous state.

        Output: (batch, num_nodes, num_timesteps, feature_dim) - time derivative.

"""
        pass


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)


# --- [Original file: model.py] ---
class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                   num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], 12, A_hat, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        """
        [TODO] Apply STGODE's temporal-ODE-temporal graph block.

        Input:
            X: (batch, num_nodes, num_timesteps, in_channels) - input graph signals.

        Output: (batch, num_nodes, num_timesteps, out_channels[-1]) - normalized block features.

"""
        pass


class ODEGCN(nn.Module):
    """ the overall network framework """
    def __init__(self, num_nodes, num_features, num_timesteps_input,
                 num_timesteps_output, A_sp_hat, A_se_hat):
        """
        Args:
            num_nodes : number of nodes in the graph
            num_features : number of features at each node in each time step
            num_timesteps_input : number of past time steps fed into the network
            num_timesteps_output : desired number of future time steps output by the network
            A_sp_hat : nomarlized adjacency spatial matrix
            A_se_hat : nomarlized adjacency semantic matrix
        """

        super(ODEGCN, self).__init__()
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_sp_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_sp_hat)) for _ in range(3)
            ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
                STGCNBlock(in_channels=num_features, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_se_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=num_nodes, A_hat=A_se_hat)) for _ in range(3)
            ])

        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input * 64, num_timesteps_output * 32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output * 32, num_timesteps_output)
        )

    def forward(self, x):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """
        """
        [TODO] Fuse spatial and semantic STGODE branches to forecast future traffic values.

        Input:
            x: (batch, num_nodes, num_timesteps_input, num_features) - historical observations.

        Output: (batch, num_nodes, num_timesteps_output) - future predictions.

"""
        pass


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
    print("STGODE: Spatial-Temporal Graph ODE Networks")
    print("Automated Test Suite - 3 ablated targets")
    print("=" * 70)

    device = torch.device("cpu")

    print("-" * 70)
    print("[Test 1/3] ODEFunc.forward - coupled graph and temporal dynamics")
    try:
        adj = torch.eye(3, device=device)
        module = ODEFunc(feature_dim=2, temporal_dim=4, adj=adj).to(device)
        x = torch.randn(2, 3, 4, 2, device=device, requires_grad=True)
        module.x0 = torch.zeros_like(x)
        output = module(torch.tensor(0.0, device=device), x)
        check("ODEFunc output not None", output is not None)
        if output is not None:
            check("ODEFunc output shape", tuple(output.shape) == (2, 3, 4, 2),
                  f"expected (2, 3, 4, 2), got {tuple(output.shape)}")
            check("ODEFunc output finite", torch.isfinite(output).all().item())
            module.adj = torch.zeros_like(adj)
            no_graph_output = module(torch.tensor(0.0, device=device), x)
            check("ODEFunc adjacency affects dynamics", not torch.allclose(output, no_graph_output))
            output.sum().backward()
            check("ODEFunc feature and temporal gradients",
                  module.w.grad is not None and module.w.grad.abs().sum().item() > 0 and
                  module.w2.grad is not None and module.w2.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "ODEFunc.forward returned None")
    except Exception as exc:
        print(f"  [ODEFunc.forward] ERROR - {exc}")
        skip_checks(5, "ODEFunc.forward raised an exception")

    print("-" * 70)
    print("[Test 2/3] STGCNBlock.forward - temporal-ODE-temporal composition")
    try:
        block = STGCNBlock(in_channels=2, out_channels=[4, 3, 4], num_nodes=3,
                           A_hat=torch.eye(3, device=device)).to(device)
        block.eval()
        x = torch.randn(2, 3, 12, 2, device=device)
        output = block(x)
        check("STGCNBlock output not None", output is not None)
        if output is not None:
            check("STGCNBlock output shape", tuple(output.shape) == (2, 3, 12, 4),
                  f"expected (2, 3, 12, 4), got {tuple(output.shape)}")
            check("STGCNBlock output finite", torch.isfinite(output).all().item())
            output.sum().backward()
            check("STGCNBlock ODE parameters receive gradients",
                  block.odeg.odeblock.odefunc.w.grad is not None and
                  block.odeg.odeblock.odefunc.w.grad.abs().sum().item() > 0)
        else:
            skip_checks(3, "STGCNBlock.forward returned None")
    except Exception as exc:
        print(f"  [STGCNBlock.forward] ERROR - {exc}")
        skip_checks(4, "STGCNBlock.forward raised an exception")

    print("-" * 70)
    print("[Test 3/3] ODEGCN.forward - spatial and semantic graph fusion")
    spatial_calls = []
    semantic_calls = []
    hooks = []
    try:
        model = ODEGCN(num_nodes=3, num_features=2, num_timesteps_input=12,
                       num_timesteps_output=2, A_sp_hat=torch.eye(3, device=device),
                       A_se_hat=torch.ones(3, 3, device=device) / 3).to(device)
        model.eval()
        for block in model.sp_blocks:
            hooks.append(block.register_forward_hook(lambda *_: spatial_calls.append(True)))
        for block in model.se_blocks:
            hooks.append(block.register_forward_hook(lambda *_: semantic_calls.append(True)))
        x = torch.randn(2, 3, 12, 2, device=device)
        output = model(x)
        check("ODEGCN output not None", output is not None)
        if output is not None:
            check("ODEGCN output shape", tuple(output.shape) == (2, 3, 2),
                  f"expected (2, 3, 2), got {tuple(output.shape)}")
            check("ODEGCN output finite", torch.isfinite(output).all().item())
            check("ODEGCN runs all graph branches", len(spatial_calls) == 3 and len(semantic_calls) == 3)
        else:
            skip_checks(3, "ODEGCN.forward returned None")
    except Exception as exc:
        print(f"  [ODEGCN.forward] ERROR - {exc}")
        skip_checks(4, "ODEGCN.forward raised an exception")
    finally:
        for hook in hooks:
            hook.remove()

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
