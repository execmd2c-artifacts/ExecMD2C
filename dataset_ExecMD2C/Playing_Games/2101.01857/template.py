# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""DQN agent network components and implementation."""

# pylint: disable=g-bad-import-order

import typing
from typing import Any, Callable, Tuple, Union

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

Network = hk.Transformed
Params = hk.Params
NetworkFn = Callable[..., Any]


class QNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray


class IqnInputs(typing.NamedTuple):
  state: jnp.ndarray
  taus: jnp.ndarray


class IqnOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


class QRNetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_dist: jnp.ndarray


class C51NetworkOutputs(typing.NamedTuple):
  q_values: jnp.ndarray
  q_logits: jnp.ndarray


def _dqn_default_initializer(
    num_input_units: int) -> hk.initializers.Initializer:
  """Default initialization scheme inherited from past implementations of DQN.

  This scheme was historically used to initialize all weights and biases
  in convolutional and linear layers of DQN-type agents' networks.
  It initializes each weight as an independent uniform sample from [`-c`, `c`],
  where `c = 1 / np.sqrt(num_input_units)`, and `num_input_units` is the number
  of input units affecting a single output unit in the given layer, i.e. the
  total number of inputs in the case of linear (dense) layers, and
  `num_input_channels * kernel_width * kernel_height` in the case of
  convolutional layers.

  Args:
    num_input_units: number of input units to a single output unit of the layer.

  Returns:
    Haiku weight initializer.
  """
  max_val = np.sqrt(1 / num_input_units)
  return hk.initializers.RandomUniform(-max_val, max_val)


def conv(
    num_features: int,
    kernel_shape: Union[int, Tuple[int, int]],
    stride: Union[int, Tuple[int, int]],
) -> NetworkFn:
  """Convolutional layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing conv layer with DQN's legacy initialization."""
    num_input_units = inputs.shape[-1] * kernel_shape[0] * kernel_shape[1]
    initializer = _dqn_default_initializer(num_input_units)
    layer = hk.Conv2D(
        num_features,
        kernel_shape=kernel_shape,
        stride=stride,
        w_init=initializer,
        b_init=initializer,
        padding='VALID')
    return layer(inputs)

  return net_fn


def linear(num_outputs: int, with_bias=True) -> NetworkFn:
  """Linear layer with DQN's legacy weight initialization scheme."""

  def net_fn(inputs):
    """Function representing linear layer with DQN's legacy initialization."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    layer = hk.Linear(
        num_outputs,
        with_bias=with_bias,
        w_init=initializer,
        b_init=initializer)
    return layer(inputs)

  return net_fn


def layer_norm(inputs):
  """Layer normalization over the final feature dimension."""
  return hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)(inputs)


def linear_with_shared_bias(num_outputs: int) -> NetworkFn:
  """Linear layer with single shared bias instead of one bias per output."""

  def layer_fn(inputs):
    """Function representing a linear layer with single shared bias."""
    initializer = _dqn_default_initializer(inputs.shape[-1])
    bias_free_linear = hk.Linear(
        num_outputs, with_bias=False, w_init=initializer)
    linear_output = bias_free_linear(inputs)
    bias = hk.get_parameter('b', [1], inputs.dtype, init=initializer)
    bias = jnp.broadcast_to(bias, linear_output.shape)
    return linear_output + bias

  return layer_fn


def noisy_linear(num_outputs: int,
                 weight_init_stddev: float,
                 with_bias: bool = True) -> NetworkFn:
  """Linear layer with weight randomization http://arxiv.org/abs/1706.10295."""
  """
  [TODO] Build a Haiku-compatible factorized NoisyNet linear layer.

  Input:
      num_outputs: scalar - output feature count.
      weight_init_stddev: scalar - initial learned-noise scale.
      with_bias: bool - whether the deterministic mean branch includes a bias.

  Output: callable layer function. When called with inputs of shape
  (batch, num_inputs), it returns an array of shape (batch, num_outputs).

"""
  pass


def dqn_torso() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        conv(32, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(64, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(64, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn

def dqn_torso_baseline() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        conv(16, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(32, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(32 * 6, kernel_shape=(3, 3), stride=(1, 1)),
        jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn

def dqn_torso_delta() -> NetworkFn:
  """DQN convolutional torso.

  Includes scaling from [`0`, `255`] (`uint8`) to [`0`, `1`] (`float32`)`.

  Returns:
    Network function that `haiku.transform` can be called on.
  """

  def net_fn(inputs):
    """Function representing convolutional torso for a DQN Q-network."""
    network = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        conv(16, kernel_shape=(8, 8), stride=(4, 4)),
        jax.nn.relu,
        conv(32, kernel_shape=(4, 4), stride=(2, 2)),
        jax.nn.relu,
        conv(32, kernel_shape=(3, 3), stride=(1, 1)),
        #jax.nn.relu,
        hk.Flatten(),
    ])
    return network(inputs)

  return net_fn

def dqn_value_head(num_actions: int, shared_bias: bool = False) -> NetworkFn:
  """Regular DQN Q-value head with single hidden layer."""

  last_layer = linear_with_shared_bias if shared_bias else linear

  def net_fn(inputs):
    """Function representing value head for a DQN Q-network."""
    network = hk.Sequential([
        linear(512),
        jax.nn.relu,
        last_layer(num_actions),
    ])
    return network(inputs)

  return net_fn


def rainbow_atari_network(
    num_actions: int,
    support: jnp.ndarray,
    noisy_weight_init: float,
) -> NetworkFn:
  """Rainbow network, expects `uint8` input."""
  """
  [TODO] Build the Rainbow Atari network with noisy dueling categorical heads.

  Input:
      num_actions: scalar - number of discrete actions.
      support: (num_atoms,) - fixed categorical value atoms.
      noisy_weight_init: scalar - initial scale for NoisyNet layers.

  Output: callable network function. When called with Atari observations of
  shape (batch, height, width, frames), it returns C51NetworkOutputs with:
      q_logits: (batch, num_actions, num_atoms)
      q_values: (batch, num_actions)

"""
  pass

def rainbow_flare_atari_network(
    num_actions: int,
    support: jnp.ndarray,
    noisy_weight_init: float,
) -> NetworkFn:
  """Rainbow Flare network, expects `uint8` input."""

  if support.ndim != 1:
    raise ValueError('support should be 1D.')
  num_atoms = len(support)
  support = support[None, None, :]

  def net_fn(inputs):
    """Function representing Rainbow Q-network."""
    inputs = jnp.transpose(inputs, [0, 3, 1, 2])
    inputs = dqn_torso_delta()(inputs.reshape(-1, 84, 84, 1))
    inputs = inputs.reshape(-1, 4, 1568)
    current = inputs[:, 1:, :]
    prev = jax.lax.stop_gradient(inputs[:, :-1, :])
    inputs = jax.numpy.concatenate([current, current - prev], axis=1)
    inputs = hk.Flatten()(inputs)
    inputs = linear(256)(inputs)
    inputs = layer_norm(inputs)

    # Advantage head.
    advantage = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    advantage = jax.nn.relu(advantage)
    advantage = noisy_linear(
        num_actions * num_atoms, noisy_weight_init, with_bias=False)(
            advantage)
    advantage = jnp.reshape(advantage, (-1, num_actions, num_atoms))

    # Value head.
    value = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    value = jax.nn.relu(value)
    value = noisy_linear(num_atoms, noisy_weight_init, with_bias=False)(value)
    value = jnp.reshape(value, (-1, 1, num_atoms))

    # Q-distribution and values.
    q_logits = value + advantage - jnp.mean(advantage, axis=-2, keepdims=True)
    assert q_logits.shape[1:] == (num_actions, num_atoms)
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn

def rainbow_baseline_atari_network(
    num_actions: int,
    support: jnp.ndarray,
    noisy_weight_init: float,
) -> NetworkFn:
  """Rainbow Baseline network, expects `uint8` input."""

  if support.ndim != 1:
    raise ValueError('support should be 1D.')
  num_atoms = len(support)
  support = support[None, None, :]

  def net_fn(inputs):
    """Function representing Rainbow Q-network."""
    inputs = dqn_torso_baseline()(inputs)
    inputs = hk.Flatten()(inputs)
    inputs = linear(256)(inputs)
    inputs = layer_norm(inputs)

    # Advantage head.
    advantage = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    advantage = jax.nn.relu(advantage)
    advantage = noisy_linear(
        num_actions * num_atoms, noisy_weight_init, with_bias=False)(
            advantage)
    advantage = jnp.reshape(advantage, (-1, num_actions, num_atoms))

    # Value head.
    value = noisy_linear(512, noisy_weight_init, with_bias=True)(inputs)
    value = jax.nn.relu(value)
    value = noisy_linear(num_atoms, noisy_weight_init, with_bias=False)(value)
    value = jnp.reshape(value, (-1, 1, num_atoms))

    # Q-distribution and values.
    q_logits = value + advantage - jnp.mean(advantage, axis=-2, keepdims=True)
    assert q_logits.shape[1:] == (num_actions, num_atoms)
    q_dist = jax.nn.softmax(q_logits)
    q_values = jnp.sum(q_dist * support, axis=2)
    q_values = jax.lax.stop_gradient(q_values)
    return C51NetworkOutputs(q_logits=q_logits, q_values=q_values)

  return net_fn

def iqn_atari_network(num_actions: int, latent_dim: int) -> NetworkFn:
  """IQN network, expects `uint8` input."""
  """
  [TODO] Build the IQN Atari network with quantile-conditioned value heads.

  Input:
      num_actions: scalar - number of discrete actions.
      latent_dim: scalar - number of cosine basis features for each tau.

  Output: callable network function. When called with IqnInputs:
      state: (batch, height, width, frames) Atari observations.
      taus: (batch, num_samples) quantile fractions.
  it returns IqnOutputs with:
      q_dist: (batch, num_samples, num_actions)
      q_values: (batch, num_actions)

"""
  pass


def qr_atari_network(num_actions: int, quantiles: jnp.ndarray) -> NetworkFn:
  """QR-DQN network, expects `uint8` input."""

  if quantiles.ndim != 1:
    raise ValueError('quantiles has to be 1D.')
  num_quantiles = len(quantiles)

  def net_fn(inputs):
    """Function representing QR-DQN Q-network."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_quantiles * num_actions),
    ])
    network_output = network(inputs)
    q_dist = jnp.reshape(network_output, (-1, num_quantiles, num_actions))
    q_values = jnp.mean(q_dist, axis=1)
    q_values = jax.lax.stop_gradient(q_values)
    return QRNetworkOutputs(q_dist=q_dist, q_values=q_values)

  return net_fn


def c51_atari_network(num_actions: int, support: jnp.ndarray) -> NetworkFn:
  """C51 network, expects `uint8` input."""
  """
  [TODO] Build the C51 Atari categorical value-distribution network.

  Input:
      num_actions: scalar - number of discrete actions.
      support: (num_atoms,) - fixed categorical value atoms.

  Output: callable network function. When called with Atari observations of
  shape (batch, height, width, frames), it returns C51NetworkOutputs with:
      q_logits: (batch, num_actions, num_atoms)
      q_values: (batch, num_actions)

"""
  pass


def double_dqn_atari_network(num_actions: int) -> NetworkFn:
  """DQN network with shared bias in final layer, expects `uint8` input."""

  def net_fn(inputs):
    """Function representing DQN Q-network with shared bias output layer."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions, shared_bias=True),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn


def dqn_atari_network(num_actions: int) -> NetworkFn:
  """DQN network, expects `uint8` input."""

  def net_fn(inputs):
    """Function representing DQN Q-network."""
    network = hk.Sequential([
        dqn_torso(),
        dqn_value_head(num_actions),
    ])
    return QNetworkOutputs(q_values=network(inputs))

  return net_fn


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
  rng_key = jax.random.PRNGKey(42)

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
  print("DQN Zoo: core network component benchmark")
  print("Automated Test Suite - 4 ablated functions")
  print("=" * 70)
  print()

  # ==========================================================
  # Test 1/4: noisy_linear
  # ==========================================================
  print("-" * 60)
  print("[Test 1/4] noisy_linear - factorized learned noise layer")
  try:
    def noisy_net(inputs):
      return noisy_linear(5, weight_init_stddev=0.5, with_bias=True)(inputs)

    network = hk.transform(noisy_net)
    inputs = jnp.ones((2, 7), dtype=jnp.float32)
    init_key, apply_key_1, apply_key_2 = jax.random.split(rng_key, 3)
    params = network.init(init_key, inputs)
    output_1 = network.apply(params, apply_key_1, inputs)
    output_2 = network.apply(params, apply_key_2, inputs)
    check("noisy_linear output not None", output_1 is not None)
    if output_1 is not None:
      check("noisy_linear output shape", output_1.shape == (2, 5),
            f"expected (2, 5), got {tuple(output_1.shape)}")
      check("noisy_linear output finite", jnp.isfinite(output_1).all().item())
      check("noisy_linear rng affects output",
            not jnp.allclose(output_1, output_2).item())
      flat_params = hk.data_structures.to_mutable_dict(params)
      has_mu = any("mu" in module_name for module_name in flat_params)
      has_sigma = any("sigma" in module_name for module_name in flat_params)
      check("noisy_linear has mu and sigma parameters", has_mu and has_sigma)
    else:
      skip_checks(4, "noisy_linear returned None")
  except Exception as exc:
    skip_checks(5, f"noisy_linear raised {type(exc).__name__}: {exc}")
  print()

  # ==========================================================
  # Test 2/4: rainbow_atari_network
  # ==========================================================
  print("-" * 60)
  print("[Test 2/4] rainbow_atari_network - dueling categorical noisy head")
  try:
    support = jnp.linspace(-2.0, 2.0, 5)
    network = hk.transform(rainbow_atari_network(
        num_actions=4, support=support, noisy_weight_init=0.1))
    observations = jnp.ones((2, 84, 84, 4), dtype=jnp.uint8)
    init_key, apply_key = jax.random.split(rng_key)
    params = network.init(init_key, observations)
    output = network.apply(params, apply_key, observations)
    check("Rainbow output not None", output is not None)
    if output is not None:
      check("Rainbow logits shape", output.q_logits.shape == (2, 4, 5),
            f"expected (2, 4, 5), got {tuple(output.q_logits.shape)}")
      check("Rainbow q_values shape", output.q_values.shape == (2, 4),
            f"expected (2, 4), got {tuple(output.q_values.shape)}")
      finite = (
          jnp.isfinite(output.q_logits).all()
          and jnp.isfinite(output.q_values).all())
      check("Rainbow outputs finite", finite.item())
      probs = jax.nn.softmax(output.q_logits, axis=-1)
      check("Rainbow categorical probabilities sum to one",
            jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5).item())
      expected_q = jnp.sum(probs * support[None, None, :], axis=2)
      check("Rainbow q_values are support expectation",
            jnp.allclose(output.q_values, expected_q, atol=1e-5).item())
    else:
      skip_checks(5, "rainbow_atari_network returned None")
  except Exception as exc:
    skip_checks(6, f"rainbow_atari_network raised {type(exc).__name__}: {exc}")
  print()

  # ==========================================================
  # Test 3/4: iqn_atari_network
  # ==========================================================
  print("-" * 60)
  print("[Test 3/4] iqn_atari_network - tau-conditioned quantile values")
  try:
    network = hk.transform(iqn_atari_network(num_actions=4, latent_dim=8))
    observations = jnp.ones((2, 84, 84, 4), dtype=jnp.uint8)
    taus = jnp.array([[0.2, 0.5, 0.8], [0.1, 0.4, 0.9]], dtype=jnp.float32)
    inputs = IqnInputs(state=observations, taus=taus)
    init_key, apply_key = jax.random.split(rng_key)
    params = network.init(init_key, inputs)
    output = network.apply(params, apply_key, inputs)
    check("IQN output not None", output is not None)
    if output is not None:
      check("IQN q_dist shape", output.q_dist.shape == (2, 3, 4),
            f"expected (2, 3, 4), got {tuple(output.q_dist.shape)}")
      check("IQN q_values shape", output.q_values.shape == (2, 4),
            f"expected (2, 4), got {tuple(output.q_values.shape)}")
      finite = (
          jnp.isfinite(output.q_dist).all()
          and jnp.isfinite(output.q_values).all())
      check("IQN outputs finite", finite.item())
      expected_q = jnp.mean(output.q_dist, axis=1)
      check("IQN q_values average quantile distribution",
            jnp.allclose(output.q_values, expected_q, atol=1e-5).item())
      shifted_inputs = IqnInputs(state=observations, taus=1.0 - taus)
      shifted_output = network.apply(params, apply_key, shifted_inputs)
      check("IQN taus affect quantile outputs",
            not jnp.allclose(output.q_dist, shifted_output.q_dist).item())
    else:
      skip_checks(5, "iqn_atari_network returned None")
  except Exception as exc:
    skip_checks(6, f"iqn_atari_network raised {type(exc).__name__}: {exc}")
  print()

  # ==========================================================
  # Test 4/4: c51_atari_network
  # ==========================================================
  print("-" * 60)
  print("[Test 4/4] c51_atari_network - categorical value distribution")
  try:
    support = jnp.linspace(-3.0, 3.0, 7)
    network = hk.transform(c51_atari_network(num_actions=3, support=support))
    observations = jnp.ones((2, 84, 84, 4), dtype=jnp.uint8)
    init_key, apply_key = jax.random.split(rng_key)
    params = network.init(init_key, observations)
    output = network.apply(params, apply_key, observations)
    check("C51 output not None", output is not None)
    if output is not None:
      check("C51 logits shape", output.q_logits.shape == (2, 3, 7),
            f"expected (2, 3, 7), got {tuple(output.q_logits.shape)}")
      check("C51 q_values shape", output.q_values.shape == (2, 3),
            f"expected (2, 3), got {tuple(output.q_values.shape)}")
      finite = (
          jnp.isfinite(output.q_logits).all()
          and jnp.isfinite(output.q_values).all())
      check("C51 outputs finite", finite.item())
      probs = jax.nn.softmax(output.q_logits, axis=-1)
      check("C51 categorical probabilities sum to one",
            jnp.allclose(probs.sum(axis=-1), 1.0, atol=1e-5).item())
      expected_q = jnp.sum(probs * support[None, None, :], axis=2)
      check("C51 q_values are support expectation",
            jnp.allclose(output.q_values, expected_q, atol=1e-5).item())
    else:
      skip_checks(5, "c51_atari_network returned None")
  except Exception as exc:
    skip_checks(6, f"c51_atari_network raised {type(exc).__name__}: {exc}")
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
