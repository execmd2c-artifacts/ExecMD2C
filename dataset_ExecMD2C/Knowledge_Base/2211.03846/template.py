# ============================================================
# ground_truth.py - FED-CD / ENCO Core Model Components
# Source: Knowledge_Base/fed-cdi-main/causal_discovery/multivariable_mlp.py
#        Knowledge_Base/fed-cdi-main/causal_discovery/graph_fitting.py
#        Knowledge_Base/fed-cdi-main/federated/federated_simulation.py
#
# Contains ONLY core model/algorithm components and direct dependencies.
# No training loops, datasets, simulations, checkpointing, or pipeline code.
# ============================================================

import math

import numpy as np
import torch
import torch.nn as nn


# --- [Original file: causal_discovery/multivariable_mlp.py] ---
class MultivarMLP(nn.Module):

    def __init__(self, input_dims, hidden_dims, output_dims, extra_dims, actfn, pre_layers=None):
        """
        Module for stacking N neural networks in parallel for more efficient evaluation. In the context
        of ENCO, we stack the neural networks of the conditional distributions for all N variables on top
        of each other to parallelize it on a GPU. 

        Parameters
        ----------
        input_dims : int
                     Input dimensionality for all networks (in ENCO, size of embedding)
        hidden_dims : list[int]
                      Hidden dimensionalities to use in the hidden layer. Length of list determines
                      the number of hidden layers to use.
        output_dims : int
                      Output dimensionality of all networks (in ENCO, max. number of categories)
        extra_dims : list[int]
                     Number of neural networks to have in parallel (in ENCO, number of variables).
                     Can have multiple dimensions if needed.
        actfn : function -> nn.Module
                Activation function to use in between hidden layers
        pre_layers : list[nn.Module]  / nn.Module
                     Any modules that should be applied before the actual MLP. This can include 
                     an embedding layer and/or masking operation.
        """
        super().__init__()
        self.extra_dims = extra_dims

        layers = []
        if pre_layers is not None:
            if not isinstance(pre_layers, list):
                layers += [pre_layers]
            else:
                layers += pre_layers
        hidden_dims = [input_dims] + hidden_dims
        for i in range(len(hidden_dims)-1):
            if not isinstance(layers[-1], EmbedLayer):  # After an embedding layer, we directly apply a non-linearity
                layers += [MultivarLinear(input_dims=hidden_dims[i],
                                          output_dims=hidden_dims[i+1],
                                          extra_dims=extra_dims)]
            layers += [actfn()]
        layers += [MultivarLinear(input_dims=hidden_dims[-1],
                                  output_dims=output_dims,
                                  extra_dims=extra_dims)]
        self.layers = nn.ModuleList(layers)

    def forward(self, x, mask=None):
        """
        [TODO] Run the stacked per-variable conditional distribution model.

        Input:
            x: tensor - categorical samples with shape `(batch, num_vars)` before
                embedding, or hidden states with shape `(batch, num_vars, width)`.
            mask: tensor or None - candidate parent mask with shape
                `(batch, num_vars, num_vars)` or broadcast-compatible variant.

        Output:
            tensor with shape `(batch, num_vars, output_dims)` containing one
            conditional prediction vector per target variable.

"""
        pass

    @property
    def device(self):
        return next(iter(self.parameters())).device


class MultivarLinear(nn.Module):

    def __init__(self, input_dims, output_dims, extra_dims):
        """
        Linear layer with the same properties as MultivarMLP. It effectively applies N independent
        linear layers in parallel.

        Parameters
        ----------
        input_dims : int
                     Number of input dimensions per network.
        output_dims : int
                      Number of output dimensions per network.
        extra_dims : list[int]
                     Number of networks to apply in parallel. Can have multiple dimensions if needed.
        """
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.extra_dims = extra_dims

        self.weight = nn.Parameter(torch.zeros(*extra_dims, output_dims, input_dims))
        self.bias = nn.Parameter(torch.zeros(*extra_dims, output_dims))

        nn.init.kaiming_uniform_(self.weight, nonlinearity='relu')

    def forward(self, x):
        """
        [TODO] Apply independent linear maps across the configured extra dimensions.

        Input:
            x: tensor with shape `(batch, *extra_dims, input_dims)` or
                `(batch, input_dims)` when extra dimensions should be broadcast.

        Output:
            tensor with shape `(batch, *extra_dims, output_dims)`.

"""
        pass

    def extra_repr(self):
        # For printing
        return 'input_dims={}, output_dims={}, extra_dims={}'.format(
            self.input_dims, self.output_dims, str(self.extra_dims)
        )


class InputMask(nn.Module):

    def __init__(self, input_mask, concat_mask=False):
        """
        Module for handling to mask the input. Needed to simulate different parent sets.

        Parameters
        ----------
        input_mask : torch.Tensor/None
                     If a tensor, it is assumed to be a fixed mask for all forward passes.
                     If None, it is required to pass the mask during every forward pass.
        concat_mask : bool
                      If True, the mask will additionally be concatenated with the input.
                      Recommended for inputs where zero is a valid value
        """
        super().__init__()
        if isinstance(input_mask, torch.Tensor):
            self.register_buffer('input_mask', input_mask.float(), persistent=False)
        else:
            self.input_mask = input_mask
        self.concat_mask = concat_mask

    def forward(self, x, mask=None, mask_val=0):
        """
        [TODO] Apply graph-structure masks to input features.

        Input:
            x: tensor - feature tensor whose trailing dimensions correspond to
                candidate parent variables or parent embeddings.
            mask: tensor or None - binary/float mask where 1 means the parent input
                is available and 0 means it is hidden. If omitted, use the fixed
                mask stored on the module.
            mask_val: float - replacement value for hidden entries.

        Output:
            masked tensor with the same broadcasted leading shape as `x`; if
            `concat_mask` is enabled, the feature width is doubled by appending the
            expanded mask.

"""
        pass


class EmbedLayer(nn.Module):

    def __init__(self, num_vars, num_categs, hidden_dim, input_mask, sparse_embeds=False):
        """
        Embedding layer to represent categorical inputs in continuous space. For efficiency, the embeddings
        of different inputs are summed in this layer instead of stacked. This is equivalent to stacking the
        embeddings and applying a linear layer, but is more efficient with slightly more parameter cost.
        Masked inputs are represented by a zero embedding tensor.

        Parameters
        ----------
        num_vars : int
                   Number of variables that are input to each neural network.
        num_categs : int
                     Max. number of categories that each variable can take.
        hidden_dim : int
                   Output dimensionality of the embedding layer.
        input_mask : InputMask
                     Input mask module to use for masking possible inputs.
        sparse_embeds : bool
                        If True, we sparsify the embedding tensors before summing them together in the
                        forward pass. This is more memory efficient and can give a considerable speedup
                        for networks with many variables, but can be slightly slower for small networks.
                        It is recommended to set it to True for graphs with more than 50 variables.
        """
        super().__init__()
        self.num_vars = num_vars
        self.hidden_dim = hidden_dim
        self.input_mask = input_mask
        self.sparse_embeds = sparse_embeds
        self.num_categs = num_categs
        # For each of the N networks, we have num_vars*num_categs possible embeddings to model.
        # Sharing embeddings across all N networks can limit the expressiveness of the networks.
        # Instead, we share them across 10-20 variables for large graphs to reduce memory.
        self.num_embeds = self.num_vars*self.num_vars*self.num_categs
        if self.num_embeds > 1e7:
            self.num_embeds = int(math.ceil(self.num_embeds / 20.0))
            self.shortend = True
        elif self.num_embeds > 1e6:
            for s in range(11, -1, -1):
                if self.num_vars % s == 0:
                    self.num_embeds = self.num_embeds // s
                    break
            self.shortend = True
        else:
            self.shortend = False
        self.embedding = nn.Embedding(num_embeddings=self.num_embeds,
                                      embedding_dim=hidden_dim)
        self.embedding.weight.data.mul_(2./math.sqrt(self.num_vars))
        self.bias = nn.Parameter(torch.zeros(num_vars, self.hidden_dim))
        # Tensor for mapping each input to its corresponding embedding range in self.embedding
        pos_trans = torch.arange(self.num_vars**2, dtype=torch.long) * self.num_categs
        self.register_buffer("pos_trans", pos_trans, persistent=False)

    def forward(self, x, mask):
        # For very large x tensors during graph fitting, it is more efficient to split it
        # into multiple sub-tensors before running the forward pass.
        num_chunks = int(math.ceil(np.prod(mask.shape) / 256e5))
        if self.training or num_chunks == 1:
            return self.embed_tensor(x, mask)
        else:
            x = x.chunk(num_chunks, dim=0)
            mask = mask.chunk(num_chunks, dim=0)
            x_out = []
            for x_l, mask_l in zip(x, mask):
                out_l = self.embed_tensor(x_l, mask_l)
                x_out.append(out_l)
            x_out = torch.cat(x_out, dim=0)
            return x_out

    def embed_tensor(self, x, mask):
        """
        [TODO] Embed categorical variables for each target variable under a parent mask.

        Input:
            x: integer tensor with shape `(batch, num_vars)` or
                `(batch, num_vars, num_vars)` containing categorical values.
            mask: tensor with shape `(batch, num_vars, num_vars)` where each row
                selects candidate parents for each target variable.

        Output:
            tensor with shape `(batch, num_vars, hidden_dim)` containing one
            summed parent embedding vector per target variable.

"""
        pass


def get_activation_function(actfn):
    """
    Returns an activation function based on a string description.
    """
    if actfn is None or actfn == 'leakyrelu':
        def create_actfn(): return nn.LeakyReLU(0.1, inplace=True)
    elif actfn == 'gelu':
        def create_actfn(): return nn.GELU()
    elif actfn == 'relu':
        def create_actfn(): return nn.ReLU()
    elif actfn == 'swish' or actfn == 'silu':
        def create_actfn(): return nn.SiLU()
    else:
        raise Exception('Unknown activation function ' + str(actfn))
    return create_actfn


def create_model(num_vars, num_categs, hidden_dims, actfn=None):
    """
    Method for creating a full multivariable MLP as used in ENCO.
    """
    num_outputs = max(1, num_categs)
    num_inputs = num_vars
    actfn = get_activation_function(actfn)

    mask = InputMask(None)
    if num_categs > 0:
        pre_layers = EmbedLayer(num_vars=num_vars,
                                num_categs=num_categs,
                                hidden_dim=hidden_dims[0],
                                input_mask=mask,
                                sparse_embeds=(num_vars >= 50))
        num_inputs = pre_layers.hidden_dim
        pre_layers = [pre_layers, actfn()]
    else:
        pre_layers = mask

    mlps = MultivarMLP(input_dims=num_inputs,
                       hidden_dims=hidden_dims,
                       output_dims=num_outputs,
                       extra_dims=[num_vars],
                       actfn=actfn,
                       pre_layers=pre_layers)
    return mlps


# --- [Original file: causal_discovery/graph_fitting.py] ---
class GraphFitting(object):

    @torch.no_grad()
    def gradient_estimator(self, adj_matrices, log_likelihoods, gamma, theta, var_idx):
        """
        [TODO] Estimate ENCO gradients for edge existence and orientation parameters.

        Input:
            adj_matrices: tensor `(num_graphs, num_vars, num_vars)` containing
                sampled graph structures.
            log_likelihoods: tensor `(num_graphs, num_vars)` containing average
                likelihood scores for each sampled graph and target variable.
            gamma: tensor/parameter `(num_vars, num_vars)` for edge existence logits.
            theta: tensor/parameter `(num_vars, num_vars)` for orientation logits.
            var_idx: int - variable that was intervened on for this estimate.

        Output:
            gamma_grads: `(num_vars, num_vars)` estimated gradients.
            theta_grads: `(num_vars, num_vars)` estimated orientation gradients.
            theta_mask: `(num_vars, num_vars)` optimizer mask for theta updates.

"""
        pass


# --- [Original file: federated/federated_simulation.py] ---
class FederatedSimulator:

    @staticmethod
    def get_binary_adjacency_mat(gamma: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Calculate the adjacency matrix based on gamma and theta matrices.

        Args:
            gamma (numpy.ndarray): Edge existence matrix.
            theta (numpy.ndarray): Edge orientation matrix.

        Returns:
            numpy.ndarray: Binary adjacency matrix.
        """

        return (((gamma > 0.0) * (theta > 0.0)) == 1).astype(int)

    @staticmethod
    def adjust_theta(prior_theta):
        """
        [TODO] Restore ENCO orientation antisymmetry after federated aggregation.

        Input:
            prior_theta: numpy array `(num_vars, num_vars)` containing aggregated
                orientation scores from clients.

        Output:
            numpy array `(num_vars, num_vars)` with each off-diagonal pair adjusted
            so opposite directions have equal magnitude and opposite signs.

"""
        pass


# ============================================================
# __main__: Automated test suite for 6 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)

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
    print("FED-CD / ENCO: Core Model Component Tests")
    print("Automated Test Suite - 6 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/6: InputMask.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/6] InputMask.forward - graph parent-set masking")
    input_mask_checks = 7
    try:
        mask_layer = InputMask(None, concat_mask=True)
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=device)
        mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]], device=device)
        out = mask_layer(x, mask=mask, mask_val=-1.0)
        check("InputMask output not None", out is not None)
        if out is not None:
            expected_shape = (2, 6)
            check("InputMask output shape", tuple(out.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(out.shape)}")
            check("InputMask output finite", torch.isfinite(out).all().item())
            check("InputMask applies replacement value",
                  torch.allclose(out[:, :3], torch.tensor([[1.0, -1.0, 3.0], [-1.0, 5.0, 6.0]], device=device)))
            check("InputMask concatenates expanded mask",
                  torch.allclose(out[:, 3:], mask))
            x_grad = x.clone().requires_grad_(True)
            masked_grad_out = InputMask(None)(x_grad, mask=mask)
            masked_grad_out.sum().backward()
            check("InputMask gradient follows unmasked entries",
                  torch.allclose(x_grad.grad, mask))
            fixed_layer = InputMask(mask)
            fixed_out = fixed_layer(x)
            check("InputMask fixed mask branch", torch.allclose(fixed_out, x * mask))
        else:
            skip_checks(input_mask_checks - 1, "InputMask returned None")
    except Exception as exc:
        skip_checks(input_mask_checks, f"InputMask.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/6: MultivarLinear.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 2/6] MultivarLinear.forward - parallel variable-specific linear maps")
    linear_checks = 6
    try:
        layer = MultivarLinear(input_dims=3, output_dims=2, extra_dims=[4]).to(device)
        x = torch.randn(5, 4, 3, device=device)
        y = layer(x)
        check("MultivarLinear output not None", y is not None)
        if y is not None:
            expected_shape = (5, 4, 2)
            check("MultivarLinear output shape", tuple(y.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(y.shape)}")
            check("MultivarLinear output finite", torch.isfinite(y).all().item())
            x_without_extra = torch.randn(5, 3, device=device)
            y_without_extra = layer(x_without_extra)
            check("MultivarLinear broadcasts missing extra dim",
                  tuple(y_without_extra.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(y_without_extra.shape)}")
            loss = y.sum() + y_without_extra.sum()
            loss.backward()
            check("MultivarLinear weight gradients exist",
                  layer.weight.grad is not None and layer.weight.grad.abs().sum().item() > 0)
            check("MultivarLinear bias gradients exist",
                  layer.bias.grad is not None and layer.bias.grad.abs().sum().item() > 0)
        else:
            skip_checks(linear_checks - 1, "MultivarLinear returned None")
    except Exception as exc:
        skip_checks(linear_checks, f"MultivarLinear.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/6: EmbedLayer.embed_tensor
    # ==========================================================
    print("-" * 60)
    print("[Test 3/6] EmbedLayer.embed_tensor - categorical parent embedding")
    embed_checks = 8
    try:
        input_mask = InputMask(None)
        embed_layer = EmbedLayer(num_vars=3, num_categs=4, hidden_dim=5, input_mask=input_mask).to(device)
        x = torch.tensor([[0, 1, 2], [2, 3, 1]], dtype=torch.long, device=device)
        mask_all = torch.ones(2, 3, 3, device=device)
        mask_none = torch.zeros(2, 3, 3, device=device)
        out_all = embed_layer.embed_tensor(x, mask_all)
        out_none = embed_layer.embed_tensor(x, mask_none)
        check("EmbedLayer output not None", out_all is not None)
        if out_all is not None:
            expected_shape = (2, 3, 5)
            check("EmbedLayer output shape", tuple(out_all.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(out_all.shape)}")
            check("EmbedLayer output finite", torch.isfinite(out_all).all().item())
            expected_bias = embed_layer.bias.view(1, 3, 5).expand(2, -1, -1)
            check("EmbedLayer zero mask returns bias only",
                  torch.allclose(out_none, expected_bias, atol=1e-6))
            check("EmbedLayer mask changes embeddings",
                  not torch.allclose(out_all, out_none, atol=1e-6))
            out_all.sum().backward()
            check("EmbedLayer embedding gradients exist",
                  embed_layer.embedding.weight.grad is not None
                  and embed_layer.embedding.weight.grad.abs().sum().item() > 0)
            check("EmbedLayer bias gradients exist",
                  embed_layer.bias.grad is not None and embed_layer.bias.grad.abs().sum().item() > 0)
            embed_layer.eval()
            forward_out = embed_layer(x, mask_all)
            check("EmbedLayer forward delegates same shape", tuple(forward_out.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(forward_out.shape)}")
        else:
            skip_checks(embed_checks - 1, "EmbedLayer returned None")
    except Exception as exc:
        skip_checks(embed_checks, f"EmbedLayer.embed_tensor raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/6: MultivarMLP.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 4/6] MultivarMLP.forward - masked conditional distribution model")
    mlp_checks = 7
    try:
        model = create_model(num_vars=3, num_categs=4, hidden_dims=[6], actfn="relu").to(device)
        x = torch.tensor([[0, 1, 2], [2, 3, 1]], dtype=torch.long, device=device)
        mask_all = torch.ones(2, 3, 3, device=device)
        mask_without_parents = torch.zeros(2, 3, 3, device=device)
        preds_all = model(x, mask=mask_all)
        preds_masked = model(x, mask=mask_without_parents)
        check("MultivarMLP output not None", preds_all is not None)
        if preds_all is not None:
            expected_shape = (2, 3, 4)
            check("MultivarMLP output shape", tuple(preds_all.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(preds_all.shape)}")
            check("MultivarMLP output finite", torch.isfinite(preds_all).all().item())
            check("MultivarMLP mask affects predictions",
                  not torch.allclose(preds_all, preds_masked, atol=1e-6))
            loss = preds_all.sum() + preds_masked.sum()
            loss.backward()
            first_param = next(iter(model.parameters()))
            check("MultivarMLP gradients exist",
                  first_param.grad is not None and torch.isfinite(first_param.grad).all().item())
            check("MultivarMLP device property", model.device == first_param.device)
            check("MultivarMLP has per-variable final layer",
                  isinstance(model.layers[-1], MultivarLinear) and model.layers[-1].extra_dims == [3])
        else:
            skip_checks(mlp_checks - 1, "MultivarMLP returned None")
    except Exception as exc:
        skip_checks(mlp_checks, f"MultivarMLP.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/6: GraphFitting.gradient_estimator
    # ==========================================================
    print("-" * 60)
    print("[Test 5/6] GraphFitting.gradient_estimator - ENCO graph parameter gradients")
    gradient_checks = 8
    try:
        graph_fit = GraphFitting()
        graph_fit.lambda_sparse = 0.1
        graph_fit.theta_grad_mask = torch.zeros(3, 3)
        adj_matrices = torch.tensor(
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
            device=device,
        )
        log_likelihoods = torch.tensor(
            [[0.2, -0.1, 0.3], [0.4, 0.0, -0.2], [-0.3, 0.5, 0.1], [0.1, -0.4, 0.2]],
            device=device,
        )
        gamma = nn.Parameter(torch.zeros(3, 3, device=device))
        theta = nn.Parameter(torch.zeros(3, 3, device=device))
        gamma_grads, theta_grads, theta_mask = graph_fit.gradient_estimator(
            adj_matrices, log_likelihoods, gamma, theta, var_idx=1
        )
        check("gradient_estimator outputs not None",
              gamma_grads is not None and theta_grads is not None and theta_mask is not None)
        if gamma_grads is not None and theta_grads is not None and theta_mask is not None:
            expected_shape = (3, 3)
            check("gradient_estimator gamma shape", tuple(gamma_grads.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(gamma_grads.shape)}")
            check("gradient_estimator theta shape", tuple(theta_grads.shape) == expected_shape,
                  f"expected {expected_shape}, got {tuple(theta_grads.shape)}")
            check("gradient_estimator finite outputs",
                  torch.isfinite(gamma_grads).all().item()
                  and torch.isfinite(theta_grads).all().item()
                  and torch.isfinite(theta_mask).all().item())
            check("gradient_estimator masks intervened incoming gamma",
                  torch.allclose(gamma_grads[:, 1], torch.zeros(3, device=device)))
            check("gradient_estimator masks gamma diagonal",
                  torch.allclose(torch.diag(gamma_grads), torch.zeros(3, device=device)))
            check("gradient_estimator theta antisymmetric",
                  torch.allclose(theta_grads + theta_grads.T, torch.zeros(3, 3, device=device), atol=1e-6))
            check("gradient_estimator theta mask covers intervened row and column",
                  torch.all(theta_mask[1, [0, 2]] == 1.0).item()
                  and torch.all(theta_mask[[0, 2], 1] == 1.0).item()
                  and theta_mask[1, 1].item() == 0.0)
        else:
            skip_checks(gradient_checks - 1, "gradient_estimator returned None")
    except Exception as exc:
        skip_checks(gradient_checks, f"GraphFitting.gradient_estimator raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6/6: FederatedSimulator.adjust_theta
    # ==========================================================
    print("-" * 60)
    print("[Test 6/6] FederatedSimulator.adjust_theta - orientation consistency after aggregation")
    theta_checks = 6
    try:
        theta = np.array(
            [
                [0.0, 0.2, -0.7],
                [0.4, 0.0, 0.1],
                [-0.1, -0.5, 0.0],
            ],
            dtype=float,
        )
        adjusted = FederatedSimulator.adjust_theta(theta.copy())
        gamma = np.ones_like(adjusted)
        binary_adj = FederatedSimulator.get_binary_adjacency_mat(gamma, adjusted)
        check("adjust_theta output not None", adjusted is not None)
        if adjusted is not None:
            expected_shape = (3, 3)
            check("adjust_theta output shape", adjusted.shape == expected_shape,
                  f"expected {expected_shape}, got {adjusted.shape}")
            check("adjust_theta finite", np.isfinite(adjusted).all())
            check("adjust_theta antisymmetric off diagonal",
                  np.allclose(adjusted + adjusted.T, np.zeros_like(adjusted), atol=1e-8))
            check("adjust_theta preserves strongest pair magnitude",
                  np.isclose(abs(adjusted[0, 1]), max(abs(theta[0, 1]), abs(theta[1, 0]))))
            check("get_binary_adjacency_mat shape and diagonal",
                  binary_adj.shape == expected_shape and np.all(np.diag(binary_adj) == 0))
        else:
            skip_checks(theta_checks - 1, "adjust_theta returned None")
    except Exception as exc:
        skip_checks(theta_checks, f"FederatedSimulator.adjust_theta raised {type(exc).__name__}: {exc}")
    print()

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
