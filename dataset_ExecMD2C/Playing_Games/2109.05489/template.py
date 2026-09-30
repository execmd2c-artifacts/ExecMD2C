# ============================================================
# ground_truth.py - control-pcgrl Core RL Model Components
# Source: control_pcgrl/rl/models.py
#
# Contains ONLY selected model architecture definitions.
# No training, inference pipeline, environment, dataset, or CLI code.
# ============================================================

from einops import rearrange
import numpy as np
import torch as th
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils import override
from torch import nn
from torch.nn import Conv2d


# --- [Original file: control_pcgrl/rl/models.py] ---


class SeqNCA(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 **kwargs,
                #  n_aux_chan=0,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        
        custom_model_config = model_config["custom_model_config"]
        # HACK: Because rllib silently squashes our multi-agent observation somewhere along the way??? :D
        obs_space = custom_model_config['dummy_env_obs_space']

        conv_filters = custom_model_config['conv_filters']
        fc_size = custom_model_config['fc_size']

        # self.n_aux_chan = n_aux_chan
        self.conv_filters = conv_filters
        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        # obs_shape = (32, 32, 3)
        obs_shape = obs_space.shape
        self.obs_shape = obs_shape
        dim = len(obs_shape[:-1])
        self.is_3D = dim == 3

        # orig_obs_space = model_config['custom_model_config']['orig_obs_space']
        # obs_shape = orig_obs_space['map'].shape
        # metrics_size = orig_obs_space['ctrl_metrics'].shape \
            # if 'ctrl_metrics' in orig_obs_space.spaces else (0,)
        # assert len(metrics_size) == 1
        # metrics_size = metrics_size[0]
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters + metrics_size

        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * conv_filters

        self.fc_size = fc_size

        # TODO: use more convolutions here? Change and check that we can still overfit on binary problem.
        # self.conv_1 = nn.Conv2d(obs_shape[-1] + n_aux_chan, out_channels=conv_filters + n_aux_chan, kernel_size=3, stride=1, padding=0)
        if self.is_3D:
            self.conv_1 = nn.Conv3d(obs_shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=1)
        else:
            self.conv_1 = nn.Conv2d(obs_shape[-1], out_channels=conv_filters, kernel_size=3, stride=1, padding=1)

        # Calculate the size of the flattened feature vector after conv_1 by feeding in a dummy tensor
        # with the correct shape.
        dummy_input = th.zeros((1, *obs_shape))
        dummy_input = dummy_input.permute(0, 3, 1, 2)
        dummy_input = self.conv_1(dummy_input.float())
        dummy_input = dummy_input.reshape(dummy_input.size(0), -1)
        self.pre_fc_size = dummy_input.shape[1]

        # self.pre_fc_size = math.prod([obs_shape[-2-i] for i in range(dim)]) * conv_filters

        self.patch_width = model_config['custom_model_config']['patch_width']

        if self.patch_width is None:
            # Default to 3x3 patches.
            pw = 3
        elif self.patch_width == -1:
            # Do not carve out any patch (basically the default model)
            pw = self.obs_shape[0]  # NOTE: Assuming the observation is square/cube.
        else:
            pw = self.patch_width

        self.fc_1 = SlimFC(self.pre_fc_size, self.fc_size)

        self.action_branch = nn.Sequential(
            # SlimFC(3 * 3 * conv_filters + metrics_size, self.fc_size),
            SlimFC( (pw ** dim) * conv_filters, self.fc_size),
            nn.ReLU(),
            SlimFC(self.fc_size, num_outputs),)

        self.value_branch = nn.Sequential(
            self.fc_1,
            nn.ReLU(),
            SlimFC(self.fc_size, 1),
        )   
        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        """
        [TODO] Convert channel-last observations into NCA features, use the
        centered local patch for action logits, and store full-map features for
        the value function.

        Input:
            input_dict["obs"]: (batch, height, width, channels) for 2D inputs,
                or (batch, height, width, length, channels) for 3D inputs.
            state: recurrent state list expected by RLlib, unused here.
            seq_lens: sequence lengths expected by RLlib, unused here.

        Output: tuple(action_logits, []), where action_logits has shape
        (batch, num_outputs).

"""
        pass


class WideModel3D(TorchModelV2, nn.Module):
    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 n_hid_filters=64,  # number of "hidden" filters in convolutional layers
                # fc_size=128,
                 ):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # How many possible actions can the agent take *at a given coordinate*.
        num_output_actions = num_outputs // np.prod(obs_space.shape[:-1])

        # self.obs_size = get_preprocessor(obs_space)(obs_space).size
        obs_shape = obs_space.shape

        # Determine size of activation after convolutional layers so that we can initialize the fully-connected layer 
        # with the correct number of weights.
        # TODO: figure this out properly, independent of map size. Here we just assume width/height/length of 
        # (padded) observation is 14
        # self.pre_fc_size = (obs_shape[-2] - 2) * (obs_shape[-3] - 2) * 32
        # self.pre_fc_size = 128 * 2 * 2 * 2

        # Size of activation after flattening, after convolutional layers and before the value branch.
        pre_val_size = (obs_shape[-2]) * (obs_shape[-3]) * (obs_shape[-4]) * num_output_actions

        # Convolutinal layers.
        self.conv_1 = nn.Conv3d(obs_space.shape[-1], out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7   
        self.conv_2 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7
        self.conv_3 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7
#       self.conv_4 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7
#       self.conv_5 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=3, padding=1)  # 64 * 7 * 7 * 7
#       self.conv_6 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=3, padding=1)  # 64 * 7 * 7 * 7
#       self.conv_7 = nn.Conv3d(n_hid_filters, out_channels=n_hid_filters, kernel_size=3, padding=1)  # 64 * 7 * 7 * 7
        self.conv_8 = nn.Conv3d(n_hid_filters, out_channels=num_output_actions, kernel_size=5, padding=2)  # 64 * 7 * 7 * 7 

        # Fully connected layer.
        # self.fc_1 = SlimFC(self.pre_fc_size, fc_size)

        # Fully connected action and value heads.
        # self.action_branch = SlimFC(fc_size, num_outputs)
        self.value_branch = SlimFC(pre_val_size, 1)

        # Holds the current "base" output (before logits layer).
        self._features = None

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return th.reshape(self.value_branch(self._features), [-1])

    def forward(self, input_dict, state, seq_lens):
        """
        [TODO] Produce a full 3D action-logit field from a channel-last 3D
        observation.

        Input:
            input_dict["obs"]: (batch, height, width, length, channels) -
                channel-last 3D map observations.
            state: recurrent state list expected by RLlib, unused here.
            seq_lens: sequence lengths expected by RLlib, unused here.

        Output: tuple(action_logits, []), where action_logits has shape
        (batch, height * width * length * actions_per_cell).

"""
        pass


class WideModel3DSkip(WideModel3D, nn.Module):
    def forward(self, input_dict, state, seq_lens):
        """
        [TODO] Produce a full 3D action-logit field using the residual wide
        model variant.

        Input:
            input_dict["obs"]: (batch, height, width, length, channels) -
                channel-last 3D map observations.
            state: recurrent state list expected by RLlib, unused here.
            seq_lens: sequence lengths expected by RLlib, unused here.

        Output: tuple(action_logits, []), where action_logits has shape
        (batch, height * width * length * actions_per_cell).

"""
        pass


def init_weights(m):
    if type(m) == th.nn.Linear:
        th.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

    if type(m) == th.nn.Conv2d:
        th.nn.init.orthogonal_(m.weight)



class NCA(TorchModelV2, nn.Module):
    """ A neural cellular automata-type NN to generate levels or wide-representation action distributions."""

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        conv_filters = model_config.get('custom_model_config').get('conv_filters', 128)
        n_hid_1 = n_hid_2 = conv_filters
        # n_hid_1 = 128
        # n_hid_2 = 128
        n_in_chans = obs_space.shape[-1]
        # TODO: have these supplied to `__init__`
        # n_out_chans = n_in_chans - 1  # assuming we observa path
        n_out_chans = n_in_chans
        w_out = obs_space.shape[0]  # assuming no observable border
        h_out = obs_space.shape[1]

        self.l1 = Conv2d(n_in_chans + 2, n_hid_1, 3, 1, 1, bias=True)  # +2 for x, y coordinates at each tile
        self.l2 = Conv2d(n_hid_1, n_hid_1, 1, 1, 0, bias=True)
        self.l3 = Conv2d(n_hid_1, n_out_chans, 1, 1, 0, bias=True)
        # self.l_vars = nn.Conv2d(n_hid_1, n_out_chans, 1, 1, 0, bias=True)
        # self.l3 = Conv2d(n_hid_1, n_out_chans * 2, 1, 1, 0, bias=True)
        self.value_branch = SlimFC(n_out_chans * w_out * h_out, 1)
        # self.value_branch = SlimFC(n_out_chans * w_out * h_out * 2, 1)
        # self.layers = [self.l1, self.l2, self.l3]
        with th.no_grad():
            self.indices = (th.Tensor(np.indices((w_out, h_out)))[None,...] / max(w_out, h_out)) * 2 - 1
            print('indices max:', self.indices.max(), 'indices min:', self.indices.min())
        self.apply(init_weights)

    def forward(self, input_dict, state, seq_lens):
        """
        [TODO] Run the coordinate-augmented 2D NCA policy and flatten its full
        output field.

        Input:
            input_dict["obs"]: (batch, height, width, channels) - channel-last
                2D map observations.
            state: recurrent state list expected by RLlib, unused here.
            seq_lens: sequence lengths expected by RLlib, unused here.

        Output: tuple(action_logits, []), where action_logits has shape
        (batch, height * width * channels).

"""
        pass

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        vals = th.reshape(self.value_branch(self._features), [-1])
        return vals


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    torch_manual_seed = getattr(th, "manual_seed")
    torch_manual_seed(42)

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

    class DummySpace:
        def __init__(self, shape):
            self.shape = shape

    print("=" * 70)
    print("control-pcgrl: core RL model component benchmark")
    print("Automated Test Suite - 4 ablated functions")
    print("=" * 70)
    print()

    device = th.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/4: SeqNCA.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/4] SeqNCA.forward - centered patch action branch")
    try:
        obs_space = DummySpace((9, 9, 3))
        model_config = {
            "custom_model_config": {
                "dummy_env_obs_space": obs_space,
                "conv_filters": 4,
                "fc_size": 8,
                "patch_width": 3,
            }
        }
        model = SeqNCA(obs_space, DummySpace(()), 7, model_config, "seq_nca").to(device)
        obs = th.randn(2, 9, 9, 3, device=device)
        output, state = model({"obs": obs}, [], None)
        check("SeqNCA output not None", output is not None)
        if output is not None:
            check("SeqNCA output shape", output.shape == (2, 7),
                  f"expected (2, 7), got {tuple(output.shape)}")
            check("SeqNCA output finite", th.isfinite(output).all().item())
            check("SeqNCA recurrent state empty", state == [])
            check("SeqNCA stores full-map value features",
                  model._features is not None and model._features.shape == (2, 9 * 9 * 4))
            check("SeqNCA patch head input size",
                  model.action_branch[0]._model[0].in_features == 3 * 3 * 4)
            value = model.value_function()
            check("SeqNCA value shape", value.shape == (2,),
                  f"expected (2,), got {tuple(value.shape)}")
        else:
            skip_checks(6, "SeqNCA.forward returned None")
    except Exception as exc:
        skip_checks(7, f"SeqNCA.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/4: WideModel3D.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 2/4] WideModel3D.forward - full 3D action field")
    try:
        obs_space = DummySpace((5, 5, 5, 2))
        num_outputs = 5 * 5 * 5 * 3
        model = WideModel3D(
            obs_space, DummySpace(()), num_outputs,
            {"custom_model_config": {}}, "wide3d", n_hid_filters=4).to(device)
        obs = th.randn(2, 5, 5, 5, 2, device=device)
        output, state = model({"obs": obs}, [], None)
        check("WideModel3D output not None", output is not None)
        if output is not None:
            check("WideModel3D output shape", output.shape == (2, 375),
                  f"expected (2, 375), got {tuple(output.shape)}")
            check("WideModel3D output finite", th.isfinite(output).all().item())
            check("WideModel3D recurrent state empty", state == [])
            check("WideModel3D features are action logits",
                  model._features is not None and th.equal(model._features, output))
            value = model.value_function()
            check("WideModel3D value shape", value.shape == (2,),
                  f"expected (2,), got {tuple(value.shape)}")
        else:
            skip_checks(5, "WideModel3D.forward returned None")
    except Exception as exc:
        skip_checks(6, f"WideModel3D.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/4: WideModel3DSkip.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 3/4] WideModel3DSkip.forward - residual 3D wide model")
    try:
        obs_space = DummySpace((5, 5, 5, 2))
        num_outputs = 5 * 5 * 5 * 3
        model = WideModel3DSkip(
            obs_space, DummySpace(()), num_outputs,
            {"custom_model_config": {}}, "wide3d_skip", n_hid_filters=4).to(device)
        with th.no_grad():
            model.conv_3.weight.zero_()
            model.conv_3.bias.zero_()
            model.conv_8.weight.fill_(1.0)
            model.conv_8.bias.zero_()
        obs = th.ones(2, 5, 5, 5, 2, device=device)
        output, state = model({"obs": obs}, [], None)
        check("WideModel3DSkip output not None", output is not None)
        if output is not None:
            check("WideModel3DSkip output shape", output.shape == (2, 375),
                  f"expected (2, 375), got {tuple(output.shape)}")
            check("WideModel3DSkip output finite", th.isfinite(output).all().item())
            check("WideModel3DSkip recurrent state empty", state == [])
            check("WideModel3DSkip preserves conv2 path through skip",
                  output.abs().sum().item() > 0)
            value = model.value_function()
            check("WideModel3DSkip value shape", value.shape == (2,),
                  f"expected (2,), got {tuple(value.shape)}")
        else:
            skip_checks(5, "WideModel3DSkip.forward returned None")
    except Exception as exc:
        skip_checks(6, f"WideModel3DSkip.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/4: NCA.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 4/4] NCA.forward - coordinate-augmented 2D generator")
    try:
        obs_space = DummySpace((6, 6, 3))
        model = NCA(
            obs_space, DummySpace(()), 6 * 6 * 3,
            {"custom_model_config": {"conv_filters": 5}}, "coord_nca").to(device)
        obs = th.zeros(2, 6, 6, 3, device=device)
        output, state = model({"obs": obs}, [], None)
        check("NCA output not None", output is not None)
        if output is not None:
            check("NCA output shape", output.shape == (2, 108),
                  f"expected (2, 108), got {tuple(output.shape)}")
            check("NCA output finite", th.isfinite(output).all().item())
            check("NCA recurrent state empty", state == [])
            check("NCA first layer includes coordinate channels",
                  model.l1.in_channels == obs_space.shape[-1] + 2)
            check("NCA stores flattened channel-last features",
                  model._features is not None and model._features.shape == (2, 108))
            value = model.value_function()
            check("NCA value shape", value.shape == (2,),
                  f"expected (2,), got {tuple(value.shape)}")
        else:
            skip_checks(6, "NCA.forward returned None")
    except Exception as exc:
        skip_checks(7, f"NCA.forward raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
