# ============================================================
# ground_truth.py - Avalanche RL Core Model Components
# Source: avalanche_rl/models/dqn.py and avalanche_rl/models/actor_critic.py
#
# Contains ONLY model architecture definitions.
# No training, inference pipeline, environment, dataset, or strategy code.
# ============================================================

from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from avalanche.models.simple_mlp import SimpleMLP
from torch.distributions import Categorical


# --- [Original file: avalanche_rl/models/dqn.py] ---


class DQNModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x: torch.Tensor, task_label=None):
        raise NotImplementedError()

    @torch.no_grad()
    def get_action(self, observation: torch.Tensor, task_label=None):
        q_values = self(observation, task_label=task_label)
        return torch.argmax(q_values, dim=1).cpu().int().numpy()


class MLPDeepQN(DQNModel):
    """
    Simple Action-Value MLP for DQN.
    """
    def __init__(
            self, input_size: int, hidden_size: int, n_actions: int,
            hidden_layers: int = 1):
        super().__init__()
        # disable dropout by default
        self.dqn = SimpleMLP(
            num_classes=n_actions, input_size=input_size,
            hidden_size=hidden_size, hidden_layers=hidden_layers, drop_rate=0.)

    def forward(self, x: torch.Tensor, task_label=None):
        return self.dqn(x)


class ConvDeepQN(DQNModel):
    # network architecture from Mnih et al 2015
    # "Human-level Control Through Deep Reinforcement Learning"
    def __init__(self, input_channels, image_shape, n_actions,
                 batch_norm=False):
        super(ConvDeepQN, self).__init__()
        # 4x84x84 input in original paper
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        self.fc = nn.Sequential(
            nn.Linear(
                self._compute_flattened_shape(
                    (input_channels, image_shape[0],
                     image_shape[1])),
                512),
            nn.ReLU(),
            nn.Linear(512, n_actions))

    def forward(self, x, task_label=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # feed to linear layer
        x = x.flatten(1)
        return self.fc(x)

    def _compute_flattened_shape(self, input_shape):
        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
        print("Size of flattened input to fully connected layer:",
              x.flatten().shape)
        return x.squeeze(0).flatten().shape[0]


class EWCConvDeepQN(DQNModel):
    """Model used in the original EWC paper https://arxiv.org/abs/1612.00796.
        It is a variant of the original DQN with added task-specific biases
        and gains.
    """
    def __init__(self, input_channels, image_shape, n_actions, n_tasks,
                 bias=False):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4, bias=bias)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, bias=bias)
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, bias=bias)
        shapes = self._compute_shapes(
            (input_channels, image_shape[0], image_shape[1]))

        # bias/gain are game-specific and are initialized as in the paper
        for layer in range(1, 4):
            for task in range(n_tasks):
                setattr(self, f'bias{layer}_{task}', nn.parameter.Parameter(
                    torch.zeros(*shapes[layer-1])))
                setattr(self, f'gain{layer}_{task}', nn.parameter.Parameter(
                    torch.ones(*shapes[layer-1])))

        # fully connected part
        self.l1 = nn.Linear(shapes[-1], 1024, bias=bias)
        self.l2 = nn.Linear(1024, n_actions, bias=bias)

        # linear layers biases & gains
        fc_sizes = [1024, n_actions]
        for layer in range(1, 3):
            for task in range(n_tasks):
                setattr(self, f'bias_l{layer}_{task}', nn.parameter.Parameter(
                    torch.zeros(fc_sizes[layer-1],)))
                setattr(self, f'gain_l{layer}_{task}', nn.parameter.Parameter(
                    torch.ones(fc_sizes[layer-1])))

    def forward(self, x: torch.Tensor, task_label=None) -> torch.Tensor:
        # biases and gains are game-specific: select them using task label
        for i in range(1, 4):
            x = getattr(self, f'conv{i}')(x)
            task_bias = getattr(self, f'bias{i}_{task_label}')
            gain = getattr(self, f'gain{i}_{task_label}')
            # print('conv shape', x.shape, task_bias.shape)
            x += task_bias
            x *= gain
            # torch.add(x, bias, alpha=gains)?
            x = F.relu(x)

        # feed to fc layer
        x = x.flatten(1)

        x = self.l1(x)
        x += getattr(self, f'bias_l1_{task_label}')
        x *= getattr(self, f'gain_l1_{task_label}')
        x = F.relu(x)

        x = self.l2(x)
        x += getattr(self, f'bias_l2_{task_label}')
        x *= getattr(self, f'gain_l2_{task_label}')

        return x

    def _compute_shapes(self, input_shape):
        # returns activation maps sizes at each layer for adding biases & gains
        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(x)
            s1 = x.shape[2:]
            x = self.conv2(x)
            s2 = x.shape[2:]
            x = self.conv3(x)
            s3 = x.shape[2:]
        return s1, s2, s3, x.squeeze(0).flatten().shape[0]


# --- [Original file: avalanche_rl/models/actor_critic.py] ---


class A2CModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(state: torch.Tensor, compute_policy=True, compute_value=True,
                task_label=None):
        raise NotImplementedError()

    @torch.no_grad()
    def get_action(self, observation: torch.Tensor, task_label=None):
        _, policy_logits = self(
            observation, compute_value=False, task_label=task_label)
        return Categorical(logits=policy_logits).sample()


class ActorCriticMLP(A2CModel):
    def __init__(
            self, num_inputs, num_actions,
            actor_hidden_sizes: Union[int, List[int]] = [64, 64],
            critic_hidden_sizes: Union[int, List[int]] = [64, 64],
            activation_type: str = 'relu'):
        super(ActorCriticMLP, self).__init__()
        # these are actually 2 models in one
        if type(actor_hidden_sizes) is int:
            actor_hidden_sizes = [actor_hidden_sizes]
        if type(critic_hidden_sizes) is int:
            critic_hidden_sizes = [critic_hidden_sizes]
        assert len(critic_hidden_sizes) and len(actor_hidden_sizes)
        if activation_type == 'relu':
            act = nn.ReLU()
        elif activation_type == 'tanh':
            act = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type {activation_type}")

        critic = [nn.Linear(
                      critic_hidden_sizes[i],
                      critic_hidden_sizes[i + 1])
                  for i in range(len(critic_hidden_sizes) - 1)]
        actor = [
            nn.Linear(actor_hidden_sizes[i],
                      actor_hidden_sizes[i + 1])
            for i in range(len(actor_hidden_sizes) - 1)]

        # self.critic_linear2 = nn.Linear(hidden_size, 1)
        self.critic = []
        for layer in [nn.Linear(num_inputs, critic_hidden_sizes[0])]+critic:
            self.critic.append(layer)
            self.critic.append(act)
        self.critic.append(nn.Linear(critic_hidden_sizes[-1], num_actions))
        self.critic = nn.Sequential(*self.critic)

        self.actor = []
        for layer in [nn.Linear(num_inputs, actor_hidden_sizes[0])]+actor:
            self.actor.append(layer)
            self.actor.append(act)
        self.actor.append(nn.Linear(actor_hidden_sizes[-1], num_actions))
        self.actor = nn.Sequential(*self.actor)

    def forward(self, state: torch.Tensor, compute_policy=True,
                compute_value=True, task_label=None):
        value, policy_logits = None, None
        if compute_value:
            value = self.critic(state)
        if compute_policy:
            policy_logits = self.actor(state)

        return value, policy_logits


class ConvActorCritic(A2CModel):
    """
        Smaller version of the Convolutional DQN network introduced in
        Mnih et al 2013 (DQN paper), re-used for experiments in
        Mnih et al. 2016 (A3C paper).
    """
    def __init__(self, input_channels, image_shape, n_actions,
                 batch_norm=False):
        super(ConvActorCritic, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 16, 8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, 4, stride=2)
        # "We typically use a convolutional neural network
        # that has one softmax output for the policy and
        # one linear output for the value function, with all
        # non-output layers shared."
        self.fc = nn.Sequential(
            nn.Linear(
                self._compute_flattened_shape(
                    (input_channels, image_shape[0],
                     image_shape[1])),
                256),
            nn.ReLU())

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, n_actions)

    def forward(self, x, compute_policy=True, compute_value=True,
                task_label=None):
        value, policy_logits = None, None

        # shared backbone of the actor-critic network
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = self.fc(x.flatten(1))

        if compute_policy:
            # actor logits output head
            policy_logits = self.actor(x)
        if compute_value:
            # value output head
            value = self.critic(x)

        return value, policy_logits

    def _compute_flattened_shape(self, input_shape):
        x = torch.zeros(input_shape)
        x = x.unsqueeze(0)
        with torch.no_grad():
            x = self.conv1(x)
            x = self.conv2(x)
        return x.squeeze(0).flatten().shape[0]


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

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
    print("Avalanche RL: core model component benchmark")
    print("Automated Test Suite - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/4: ConvDeepQN.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/4] ConvDeepQN.forward - pixel observations to Q-values")
    try:
        model = ConvDeepQN(
            input_channels=4, image_shape=(84, 84), n_actions=6).to(device)
        observations = torch.randn(2, 4, 84, 84, device=device)
        output = model(observations)
        check("ConvDeepQN output not None", output is not None)
        if output is not None:
            check("ConvDeepQN output shape", output.shape == (2, 6),
                  f"expected (2, 6), got {tuple(output.shape)}")
            check("ConvDeepQN output finite", torch.isfinite(output).all().item())
            model.zero_grad(set_to_none=True)
            output.sum().backward()
            grad_ok = (
                model.conv1.weight.grad is not None
                and model.fc[-1].weight.grad is not None
                and model.conv1.weight.grad.abs().sum().item() > 0
                and model.fc[-1].weight.grad.abs().sum().item() > 0
            )
            check("ConvDeepQN gradients reach conv and Q head", grad_ok)
        else:
            skip_checks(3, "ConvDeepQN.forward returned None")
    except Exception as exc:
        skip_checks(4, f"ConvDeepQN.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/4: EWCConvDeepQN.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 2/4] EWCConvDeepQN.forward - task-specific bias/gain flow")
    try:
        model = EWCConvDeepQN(
            input_channels=4, image_shape=(84, 84), n_actions=5,
            n_tasks=2, bias=True).to(device)
        observations = torch.randn(2, 4, 84, 84, device=device)
        with torch.no_grad():
            getattr(model, "bias_l2_1").fill_(3.0)
        output_task0 = model(observations, task_label=0)
        output_task1 = model(observations, task_label=1)
        check("EWCConvDeepQN output not None", output_task1 is not None)
        if output_task1 is not None:
            check("EWCConvDeepQN output shape", output_task1.shape == (2, 5),
                  f"expected (2, 5), got {tuple(output_task1.shape)}")
            check("EWCConvDeepQN output finite",
                  torch.isfinite(output_task1).all().item())
            differs = (
                output_task0 is not None
                and not torch.allclose(output_task0, output_task1)
            )
            check("EWCConvDeepQN task label changes selected parameters", differs)
            model.zero_grad(set_to_none=True)
            output_task1.sum().backward()
            selected_grad = getattr(model, "bias_l2_1").grad
            other_grad = getattr(model, "bias_l2_0").grad
            grad_ok = (
                selected_grad is not None
                and selected_grad.abs().sum().item() > 0
                and other_grad is None
            )
            check("EWCConvDeepQN gradients stay on selected task parameters",
                  grad_ok)
        else:
            skip_checks(4, "EWCConvDeepQN.forward returned None")
    except Exception as exc:
        skip_checks(5, f"EWCConvDeepQN.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/4: ActorCriticMLP.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 3/4] ActorCriticMLP.forward - conditional actor/critic heads")
    try:
        model = ActorCriticMLP(
            num_inputs=4, num_actions=3, actor_hidden_sizes=16,
            critic_hidden_sizes=12).to(device)
        states = torch.randn(2, 4, device=device)
        value, policy_logits = model(states)
        check("ActorCriticMLP outputs not None",
              value is not None and policy_logits is not None)
        if value is not None and policy_logits is not None:
            check("ActorCriticMLP output shapes",
                  value.shape == (2, 3) and policy_logits.shape == (2, 3),
                  f"got {tuple(value.shape)} and {tuple(policy_logits.shape)}")
            finite = (
                torch.isfinite(value).all().item()
                and torch.isfinite(policy_logits).all().item()
            )
            check("ActorCriticMLP outputs finite", finite)
            value_only, policy_none = model(states, compute_policy=False)
            value_none, policy_only = model(states, compute_value=False)
            check("ActorCriticMLP compute_policy gate",
                  value_only is not None and policy_none is None)
            check("ActorCriticMLP compute_value gate",
                  value_none is None and policy_only is not None)
        else:
            skip_checks(4, "ActorCriticMLP.forward returned None output")
    except Exception as exc:
        skip_checks(5, f"ActorCriticMLP.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/4: ConvActorCritic.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 4/4] ConvActorCritic.forward - shared CNN with two heads")
    try:
        model = ConvActorCritic(
            input_channels=4, image_shape=(84, 84), n_actions=4).to(device)
        observations = torch.randn(2, 4, 84, 84, device=device)
        value, policy_logits = model(observations)
        check("ConvActorCritic outputs not None",
              value is not None and policy_logits is not None)
        if value is not None and policy_logits is not None:
            check("ConvActorCritic output shapes",
                  value.shape == (2, 4) and policy_logits.shape == (2, 4),
                  f"got {tuple(value.shape)} and {tuple(policy_logits.shape)}")
            finite = (
                torch.isfinite(value).all().item()
                and torch.isfinite(policy_logits).all().item()
            )
            check("ConvActorCritic outputs finite", finite)
            value_only, policy_none = model(observations, compute_policy=False)
            value_none, policy_only = model(observations, compute_value=False)
            check("ConvActorCritic compute_policy gate",
                  value_only is not None and policy_none is None)
            check("ConvActorCritic compute_value gate",
                  value_none is None and policy_only is not None)
        else:
            skip_checks(4, "ConvActorCritic.forward returned None output")
    except Exception as exc:
        skip_checks(5, f"ConvActorCritic.forward raised {type(exc).__name__}: {exc}")
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
