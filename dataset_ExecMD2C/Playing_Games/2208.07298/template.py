
from numpy.core.fromnumeric import shape
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class TMixer(nn.Module):
    def __init__(self, scheme, input_shape, args):
        super(TMixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.scheme = scheme
        
        self.input_shape = input_shape
        self.state_dim = int(np.prod(args.state_shape))
        self.n_actions = args.n_actions
        self.action_dim = args.n_agents * self.n_actions
        self.state_action_dim = self.state_dim + self.action_dim + 1
        self.max_seq_len = args.batch_size * args.eps_limit
        self.embed_dim = args.embed_dim
        self.obs_dim = self.n_agents * self.scheme['obs']['vshape']
        self.hist_dim = self.args.n_agents * self.args.rnn_hidden_dim
        dim = args.embed_dim * 3
        
        self.state_transform = nn.Linear(self.state_dim, args.embed_dim)        
        self.aqs_transform = nn.Linear(self.n_agents, args.embed_dim)        
        self.hist_transform = nn.Linear(self.hist_dim, args.embed_dim)            
        
        self.enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=args.heads, dim_feedforward=args.ff, dropout=0.4, 
                                                    activation=F.relu, batch_first=True, device='cuda')
        self.mixer = nn.TransformerEncoder(self.enc_layer, num_layers=args.t_depth)
        self.bottleneck = nn.Linear(dim, dim)
        self.out = nn.Sequential( 
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, dim // 4),
            nn.GELU(),
            nn.Linear(dim // 4, 1)
            
        )
    def create_noise(self, states, mean=0, stddev=0.05):
        noise = th.as_tensor(states, dtype=th.float).normal_(mean, stddev).cuda()
        return noise
    
    def calc_v(self, agent_qs):
        v_tot = th.sum(agent_qs, dim=-1, keepdim=True)
        return v_tot

    def forward(self, agent_qs, hist, states, b_max=0):  
        """TODO: Reproduce TransMix transformer value decomposition.

        Input:
            agent_qs: tensor shaped (batch, episode_len, n_agents) containing
                per-agent chosen-action Q values.
            hist: tensor shaped (batch, episode_len, n_agents, rnn_hidden_dim)
                containing recurrent agent histories from the controller.
            states: tensor shaped (batch, episode_len, state_dim) containing the
                global state sequence.
            b_max: unused compatibility argument.
        Output:
            q_tot: tensor shaped (batch, episode_len, embed_dim * 3) under the
                source implementation's broadcasted residual formulation.

"""
        pass


if __name__ == "__main__":
    torch_device = th.device("cuda" if th.cuda.is_available() else "cpu")
    th.manual_seed(42)

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

    class Args:
        ...

    def make_args(is_noise=False):
        args = Args()
        args.n_agents = 3
        args.n_actions = 5
        args.state_shape = (7,)
        args.batch_size = 2
        args.eps_limit = 4
        args.embed_dim = 12
        args.rnn_hidden_dim = 6
        args.heads = 3
        args.ff = 48
        args.t_depth = 1
        args.is_noise = is_noise
        return args

    def make_inputs(args):
        scheme = {"obs": {"vshape": 4}}
        agent_qs = th.tensor(
            [[[0.3, -0.2, 0.5], [0.1, 0.4, -0.3], [0.2, 0.2, 0.2], [-0.4, 0.6, 0.1]],
             [[-0.1, 0.2, 0.7], [0.5, -0.5, 0.0], [0.3, 0.1, -0.2], [0.8, -0.1, 0.4]]],
            dtype=th.float,
            device=torch_device,
        )
        hist = th.randn(args.batch_size, args.eps_limit, args.n_agents, args.rnn_hidden_dim, device=torch_device)
        states = th.randn(args.batch_size, args.eps_limit, int(np.prod(args.state_shape)), device=torch_device)
        return scheme, agent_qs, hist, states

    def force_module_device(module, device):
        module.to(device)
        return module

    print("=" * 70)
    print("TransMix benchmark: transformer value decomposition mixer")
    print("=" * 70)

    try:
        args = make_args(is_noise=False)
        scheme, agent_qs, hist, states = make_inputs(args)
        mixer = force_module_device(TMixer(scheme, input_shape=args.eps_limit, args=args), torch_device)
        output = mixer(agent_qs, hist, states)
        check("TMixer forward output not None", output is not None)
        if output is not None:
            check("TMixer forward output shape", output.shape == (2, 4, 36), f"got {tuple(output.shape)}")
            check("TMixer forward finite", th.isfinite(output).all().item())
            v = mixer.calc_v(agent_qs)
            check("TMixer forward V residual broadcast", v.shape == (2, 4, 1), f"got {tuple(v.shape)}")
            loss = output.sum()
            loss.backward()
            grad = mixer.state_transform.weight.grad
            check("TMixer forward gradient", grad is not None and th.isfinite(grad).all().item())
        else:
            skip_checks(4, "TMixer.forward returned None")
    except Exception as exc:
        skip_checks(5, f"TMixer.forward raised {type(exc).__name__}: {exc}")

    try:
        args = make_args(is_noise=False)
        scheme, agent_qs, hist, states = make_inputs(args)
        mixer = force_module_device(TMixer(scheme, input_shape=args.eps_limit, args=args), torch_device)
        v = mixer.calc_v(agent_qs)
        expected = agent_qs.sum(dim=-1, keepdim=True)
        check("calc_v output not None", v is not None)
        if v is not None:
            check("calc_v shape", v.shape == (2, 4, 1), f"got {tuple(v.shape)}")
            check("calc_v equals agent sum", th.allclose(v, expected))
            check("calc_v finite", th.isfinite(v).all().item())
        else:
            skip_checks(3, "TMixer.calc_v returned None")
    except Exception as exc:
        skip_checks(4, f"TMixer.calc_v raised {type(exc).__name__}: {exc}")

    try:
        args = make_args(is_noise=True)
        scheme, agent_qs, hist, states = make_inputs(args)
        mixer = force_module_device(TMixer(scheme, input_shape=args.eps_limit, args=args), torch_device)
        output = mixer(agent_qs, hist, states)
        check("TMixer noise forward output not None", output is not None)
        if output is not None:
            check("TMixer noise forward output shape", output.shape == (2, 4, 36), f"got {tuple(output.shape)}")
            check("TMixer noise forward finite", th.isfinite(output).all().item())
            output.sum().backward()
            grad = mixer.hist_transform.weight.grad
            check("TMixer noise forward gradient", grad is not None and th.isfinite(grad).all().item())
        else:
            skip_checks(3, "TMixer.forward noise branch returned None")
    except Exception as exc:
        skip_checks(4, f"TMixer.forward noise branch raised {type(exc).__name__}: {exc}")

    try:
        args = make_args(is_noise=False)
        scheme, agent_qs, hist, states = make_inputs(args)
        mixer = force_module_device(TMixer(scheme, input_shape=args.eps_limit, args=args), torch_device)
        noise = mixer.create_noise(states, mean=0, stddev=0.05)
        check("create_noise output not None", noise is not None)
        if noise is not None:
            check("create_noise shape", noise.shape == states.shape, f"got {tuple(noise.shape)}")
            check("create_noise finite", th.isfinite(noise).all().item())
            check("create_noise device", noise.device == states.device, f"got {noise.device}, expected {states.device}")
        else:
            skip_checks(3, "TMixer.create_noise returned None")
    except Exception as exc:
        skip_checks(4, f"TMixer.create_noise raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        print(f"{failed} check(s) FAILED")
        raise SystemExit(1)
    print("All tests PASSED")
