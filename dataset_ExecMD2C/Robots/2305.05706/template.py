# ============================================================
# ground_truth.py - DexArt Core Model Components
# Source: D:\OneDrive\desktop\MDCdataset\Robots\dexart-release-main
#
# Contains ONLY the point-cloud model architecture components used by
# DexArt's PPO policy feature extractor. No training, environment,
# dataset, checkpoint, or simulation code is included.
# ============================================================

from typing import Dict, List, Optional, Type, Union

import gym
import torch
import torch as th
import torch.nn as nn

TensorDict = Dict[Union[str, int], th.Tensor]


# --- [Original file: stable_baselines3/common/torch_layers.py] ---
class BaseFeaturesExtractor(nn.Module):
    """
    Base class that represents a features extractor.

    :param observation_space:
    :param features_dim: Number of features extracted.
    """

    def __init__(self, observation_space: gym.Space, features_dim: int = 0):
        super().__init__()
        assert features_dim > 0
        self._observation_space = observation_space
        self._features_dim = features_dim

    @property
    def features_dim(self) -> int:
        return self._features_dim

    def forward(self, observations: th.Tensor) -> th.Tensor:
        raise NotImplementedError()


# --- [Original file: stable_baselines3/common/torch_layers.py] ---
def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


# --- [Original file: stable_baselines3/networks/pretrain_nets.py] ---
class PointNet(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNet, self).__init__()

        print(f'PointNetSmall')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        [TODO] Encode an unordered point cloud into one global 256-dimensional feature vector per batch item.

        Input:
            x: (batch, num_points, 3) - point coordinates for each observed point cloud.

        Output: (batch, 256) - permutation-invariant global point-cloud features.

"""
        pass


# --- [Original file: stable_baselines3/networks/pretrain_nets.py] ---
class PointNetMedium(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetMedium, self).__init__()

        print(f'PointNetMedium')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, mlp_out_dim),
        )
        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        [TODO] Encode an unordered point cloud with the medium PointNet local encoder.

        Input:
            x: (batch, num_points, 3) - point coordinates for each observed point cloud.

        Output: (batch, 256) - permutation-invariant global point-cloud features.

"""
        pass


# --- [Original file: stable_baselines3/networks/pretrain_nets.py] ---
class PointNetLarge(nn.Module):  # actually pointnet
    def __init__(self, point_channel=3, output_dim=256):
        # NOTE: we require the output dim to be 256, in order to match the pretrained weights
        super(PointNetLarge, self).__init__()

        print(f'PointNetLarge')

        in_channel = point_channel
        mlp_out_dim = 256
        self.local_mlp = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, mlp_out_dim),
        )

        self.reset_parameters_()

    def reset_parameters_(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        [TODO] Encode an unordered point cloud with the large PointNet local encoder.

        Input:
            x: (batch, num_points, 3) - point coordinates for each observed point cloud.

        Output: (batch, 256) - permutation-invariant global point-cloud features.

"""
        pass


# --- [Original file: stable_baselines3/common/torch_layers.py] ---
class PointNetImaginationExtractorGP(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, pc_key: str, feat_key: Optional[str] = None,
                 out_channel=256, extractor_name="smallpn",
                 gt_key: Optional[str] = None, imagination_keys=("imagination_robot",), state_key="state",
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU, *kwargs):
        self.imagination_key = imagination_keys
        # Init state representation
        self.use_state = state_key is not None
        self.state_key = state_key

        print(f"extractor use state = {self.use_state}")
        if self.use_state:
            if state_key not in observation_space.spaces.keys():
                raise RuntimeError(f"State key {state_key} not in observation space: {observation_space}")
            self.state_space = observation_space[self.state_key]
        if feat_key is not None:
            if feat_key not in list(observation_space.keys()):
                raise RuntimeError(f"Feature key {feat_key} not in observation space.")
        if pc_key not in list(observation_space.keys()):
            raise RuntimeError(f"Point cloud key {pc_key} not in observation space.")

        super().__init__(observation_space, out_channel)
        # Point cloud input should have size (n, 3), spec size (n, 3), feat size (n, m)
        self.pc_key = pc_key
        self.has_feat = feat_key is not None
        self.feat_key = feat_key
        self.gt_key = gt_key

        if extractor_name == "smallpn":
            self.extractor = PointNet()
        elif extractor_name == "mediumpn":
            self.extractor = PointNetMedium()
        elif extractor_name == "largepn":
            self.extractor = PointNetLarge()
        else:
            raise NotImplementedError(f"Extractor {extractor_name} not implemented. Available:\
             smallpn, mediumpn, largepn")

        # self.n_input_channels = n_input_channels
        self.n_output_channels = out_channel
        assert self.n_output_channels == 256

        if self.use_state:
            self.state_dim = self.state_space.shape[0]
            if len(state_mlp_size) == 0:
                raise RuntimeError(f"State mlp size is empty")
            elif len(state_mlp_size) == 1:
                net_arch = []
            else:
                net_arch = state_mlp_size[:-1]
            output_dim = state_mlp_size[-1]

            self.n_output_channels = out_channel + output_dim
            self._features_dim = self.n_output_channels
            self.state_mlp = nn.Sequential(*create_mlp(self.state_dim, output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: TensorDict) -> th.Tensor:
        """
        [TODO] Fuse observed point clouds, imagination point clouds, and optional robot state features.

        Input:
            observations: mapping with:
                self.pc_key: (batch, num_points, 3) - observed point cloud coordinates.
                each imagination key: (batch, num_imagination_points, channels) or
                    (num_imagination_points, channels) - imagined geometry where the first
                    three channels are point coordinates.
                self.state_key when enabled: (batch, state_dim) - low-dimensional robot state.

        Output:
            (batch, 256) when state is disabled, otherwise (batch, 256 + state_mlp_output_dim).

"""
        pass


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    import numpy as np

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
    print("DexArt PointNet Imagination Feature Extractor")
    print("Automated Test Suite - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: PointNet.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] PointNet.forward - local point encoding and global pooling")
    try:
        module = PointNet().to(device)
        x = torch.randn(2, 11, 3, device=device)
        y = module(x)
        check("PointNet output not None", y is not None)
        if y is not None:
            check("PointNet output shape", tuple(y.shape) == (2, 256), f"expected (2, 256), got {tuple(y.shape)}")
            check("PointNet output finite", torch.isfinite(y).all().item())
            permuted = module(x[:, torch.tensor([3, 1, 9, 0, 2, 10, 8, 4, 6, 5, 7], device=device), :])
            check("PointNet point-order invariant", torch.allclose(y, permuted, atol=1e-6))
            y.sum().backward()
            grad = module.local_mlp[0].weight.grad
            check("PointNet local MLP receives gradient", grad is not None and grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "PointNet.forward returned None")
    except Exception as exc:
        print(f"  [PointNet.forward] ERROR: {exc}")
        skip_checks(5, "PointNet.forward raised an exception")
    print()

    # ==============================================================
    # Test 2/4: PointNetMedium.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] PointNetMedium.forward - deeper local point encoding and pooling")
    try:
        module = PointNetMedium().to(device)
        x = torch.randn(2, 13, 3, device=device)
        y = module(x)
        check("PointNetMedium output not None", y is not None)
        if y is not None:
            check("PointNetMedium output shape", tuple(y.shape) == (2, 256), f"expected (2, 256), got {tuple(y.shape)}")
            check("PointNetMedium output finite", torch.isfinite(y).all().item())
            permuted = module(x[:, torch.tensor([8, 2, 12, 0, 5, 11, 1, 9, 6, 4, 10, 7, 3], device=device), :])
            check("PointNetMedium point-order invariant", torch.allclose(y, permuted, atol=1e-6))
            y.sum().backward()
            grad = module.local_mlp[0].weight.grad
            check("PointNetMedium local MLP receives gradient", grad is not None and grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "PointNetMedium.forward returned None")
    except Exception as exc:
        print(f"  [PointNetMedium.forward] ERROR: {exc}")
        skip_checks(5, "PointNetMedium.forward raised an exception")
    print()

    # ==============================================================
    # Test 3/4: PointNetLarge.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] PointNetLarge.forward - largest local point encoder and pooling")
    try:
        module = PointNetLarge().to(device)
        x = torch.randn(2, 17, 3, device=device)
        y = module(x)
        check("PointNetLarge output not None", y is not None)
        if y is not None:
            check("PointNetLarge output shape", tuple(y.shape) == (2, 256), f"expected (2, 256), got {tuple(y.shape)}")
            check("PointNetLarge output finite", torch.isfinite(y).all().item())
            permuted = module(x[:, torch.tensor([16, 0, 8, 4, 12, 1, 9, 5, 13, 2, 10, 6, 14, 3, 11, 7, 15], device=device), :])
            check("PointNetLarge point-order invariant", torch.allclose(y, permuted, atol=1e-6))
            y.sum().backward()
            grad = module.local_mlp[0].weight.grad
            check("PointNetLarge local MLP receives gradient", grad is not None and grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "PointNetLarge.forward returned None")
    except Exception as exc:
        print(f"  [PointNetLarge.forward] ERROR: {exc}")
        skip_checks(5, "PointNetLarge.forward raised an exception")
    print()

    # ==============================================================
    # Test 4/4: PointNetImaginationExtractorGP.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] PointNetImaginationExtractorGP.forward - point, imagination, and state fusion")
    try:
        observation_space = gym.spaces.Dict({
            "instance_1-point_cloud": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 3), dtype=np.float32),
            "imagination_robot": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4, 7), dtype=np.float32),
            "state": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32),
        })
        extractor = PointNetImaginationExtractorGP(
            observation_space=observation_space,
            pc_key="instance_1-point_cloud",
            imagination_keys=("imagination_robot",),
            state_key="state",
            state_mlp_size=(16, 64),
        ).to(device)
        observations = {
            "instance_1-point_cloud": torch.randn(2, 5, 3, device=device),
            "imagination_robot": torch.randn(2, 4, 7, device=device),
            "state": torch.randn(2, 8, device=device),
        }
        y = extractor(observations)
        check("Extractor output not None", y is not None)
        if y is not None:
            check("Extractor output shape", tuple(y.shape) == (2, 320), f"expected (2, 320), got {tuple(y.shape)}")
            check("Extractor output finite", torch.isfinite(y).all().item())
            changed_imagination = {key: value.clone() for key, value in observations.items()}
            changed_imagination["imagination_robot"] = changed_imagination["imagination_robot"] + 8.0
            y_changed_imagination = extractor(changed_imagination)
            check("Extractor imagination branch affects point features", not torch.allclose(y[:, :256], y_changed_imagination[:, :256], atol=1e-6))
            changed_state = {key: value.clone() for key, value in observations.items()}
            changed_state["state"] = changed_state["state"] + 3.0
            y_changed_state = extractor(changed_state)
            check("Extractor state branch affects state features", not torch.allclose(y[:, 256:], y_changed_state[:, 256:], atol=1e-6))
        else:
            skip_checks(4, "PointNetImaginationExtractorGP.forward returned None")
    except Exception as exc:
        print(f"  [PointNetImaginationExtractorGP.forward] ERROR: {exc}")
        skip_checks(5, "PointNetImaginationExtractorGP.forward raised an exception")
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
