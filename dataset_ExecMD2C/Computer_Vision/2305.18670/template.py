import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange


class SVDConv2d(nn.Conv2d) :
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            scale: float = 1.0,
            **kwargs
    ) :
        nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped.type(torch.FloatTensor), full_matrices=False)

        ## initialize to 0 for smooth tuning
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.delta[0: int(0.1*self.S.size()[0])].requires_grad = False
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float) :
        self.scale = scale
    def perform_svd(self) :
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped.type(torch.FloatTensor), full_matrices=False)
        self.done_svd = True

    def reset_parameters(self) :
        nn.Conv2d.reset_parameters(self)
        if hasattr(self, 'delta') :
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor) :
        """
        TODO: Reproduce the SAVE SVD-parameterized 2D convolution forward pass.

        Input:
            x: image feature tensor with shape (N, in_channels, H, W).

        Output:
            Tensor produced by a 2D convolution with the module's configured stride, padding,
            dilation, and groups.

        Required behavior:
            Ensure the SVD factors are refreshed after state-dict loading when needed. Reconstruct
            the convolution weight from the stored singular vectors and singular values plus the
            learnable spectral shift scaled by the module scale, constrain the shifted spectrum to
            non-negative values, reshape the reconstructed matrix back to convolution kernel layout,
            and apply the convolution using the original bias and convolution hyperparameters. The
            base weight must remain frozen while gradients flow to the spectral shift parameter.
        """
        pass


class SVDConv1d(nn.Conv1d) :
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            scale: float = 1.0,
            **kwargs
    ) :
        nn.Conv1d.__init__(self, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped.type(torch.FloatTensor), full_matrices=False)

        print(self.S)
        ## initialize to 0 for smooth tuning
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float) :
        self.scale = scale

    def perform_svd(self) :
        # shape
        weight_reshaped = rearrange(self.weight, 'co cin h w -> co (cin h w)')
        self.U, self.S, self.Vh = torch.linalg.svd(weight_reshaped.type(torch.FloatTensor), full_matrices=False)
        self.done_svd = True

    def reset_parameters(self) :
        nn.Conv1d.reset_parameters(self)
        if hasattr(self, 'delta') :
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor) :
        if not self.done_svd :
            # this happens after loading the state dict
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(
            F.relu(self.S.to(x.device, dtype=x.dtype) + self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = rearrange(weight_updated, 'co (cin h w) -> co cin h w', cin=self.weight.size(1),
                                   h=self.weight.size(2), w=self.weight.size(3))
        return F.conv1d(x, weight_updated, self.bias, self.stride, self.padding, self.dilation, self.groups)


class SVDLinear(nn.Linear) :
    def __init__(
            self,
            in_features: int,
            out_features: int,
            scale: float = 1.0,
            **kwargs
    ) :
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.type(torch.FloatTensor), full_matrices=False)
        # initialize to 0 for smooth tuning
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float) :
        self.scale = scale

    def perform_svd(self) :
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.type(torch.FloatTensor), full_matrices=False)
        self.done_svd = True

    def reset_parameters(self) :
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'delta') :
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor) :
        """
        TODO: Reproduce the SAVE SVD-parameterized linear forward pass.

        Input:
            x: feature tensor whose last dimension equals in_features.

        Output:
            Tensor whose last dimension equals out_features.

        Required behavior:
            Refresh SVD factors on first use after external weight loading, reconstruct the effective
            dense weight from singular vectors and shifted singular values, apply the module scale to
            the learnable spectral shift, clamp the shifted spectrum through the same non-negative
            activation used by the source implementation, and call the linear projection with the
            original bias. Keep the original weight frozen and make the output differentiable with
            respect to delta.
        """
        pass


class SVDEmbedding(nn.Embedding) :
    # LoRA implemented in a dense layer
    def __init__(
            self,
            num_embeddings: int,
            embedding_dim: int,
            scale: float = 1.0,
            **kwargs
    ) :
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.type(torch.FloatTensor), full_matrices=False)
        # initialize to 0 for smooth tuning
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float) :
        self.scale = scale

    def perform_svd(self) :
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.type(torch.FloatTensor), full_matrices=False)
        self.done_svd = True

    def reset_parameters(self) :
        nn.Embedding.reset_parameters(self)
        if hasattr(self, 'delta') :
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor) :
        """
        TODO: Reproduce the SAVE SVD-parameterized embedding lookup.

        Input:
            x: integer token/index tensor of arbitrary leading shape.

        Output:
            Embedding tensor with input leading shape plus embedding_dim.

        Required behavior:
            Refresh SVD factors on first use after external weight loading, reconstruct the embedding
            matrix from singular vectors and shifted singular values, apply the module scale to the
            learnable spectral shift with the source non-negative spectrum constraint, and perform an
            embedding lookup preserving padding, max-norm, norm-type, frequency-scaling, and sparse
            options. The base embedding weight must remain frozen while gradients can reach delta.
        """
        pass


# 1-D
class SVDLayerNorm(nn.LayerNorm) :
    def __init__(
            self,
            normalized_shape: int,
            scale: float = 1.0,
            **kwargs
    ) :
        nn.LayerNorm.__init__(self, normalized_shape=normalized_shape, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0).type(torch.FloatTensor),
                                                   full_matrices=False)
        # initialize to 0 for smooth tuning
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float) :
        self.scale = scale

    def perform_svd(self) :
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0).type(torch.FloatTensor),
                                                   full_matrices=False)
        self.done_svd = True

    def reset_parameters(self) :
        nn.LayerNorm.reset_parameters(self)
        if hasattr(self, 'delta') :
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor) :
        if not self.done_svd :
            # this happens after loading the state dict
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(
            F.relu(self.S.to(x.device, dtype=x.dtype) + self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = weight_updated.squeeze(0)
        return F.layer_norm(x, normalized_shape=self.normalized_shape, weight=weight_updated, bias=self.bias,
                            eps=self.eps)


class SVDGroupNorm(nn.GroupNorm) :
    def __init__(
            self,
            num_groups: int,
            num_channels: int,
            scale: float = 1.0,
            **kwargs
    ) :
        nn.GroupNorm.__init__(self, num_groups, num_channels, **kwargs)
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0).type(torch.FloatTensor),
                                                   full_matrices=False)
        # initialize to 0 for smooth tuning
        self.delta = nn.Parameter(torch.zeros_like(self.S))
        self.weight.requires_grad = False
        self.done_svd = False
        self.scale = scale
        self.reset_parameters()

    def set_scale(self, scale: float) :
        self.scale = scale

    def perform_svd(self) :
        self.U, self.S, self.Vh = torch.linalg.svd(self.weight.unsqueeze(0).type(torch.FloatTensor),
                                                   full_matrices=False)
        self.done_svd = True

    def reset_parameters(self) :
        nn.GroupNorm.reset_parameters(self)
        if hasattr(self, 'delta') :
            nn.init.zeros_(self.delta)

    def forward(self, x: torch.Tensor) :
        if not self.done_svd :
            # this happens after loading the state dict
            self.perform_svd()
        weight_updated = self.U.to(x.device, dtype=x.dtype) @ torch.diag(
            F.relu(self.S.to(x.device, dtype=x.dtype) + self.scale * self.delta)) @ self.Vh.to(x.device, dtype=x.dtype)
        weight_updated = weight_updated.squeeze(0)
        return F.group_norm(x, num_groups=self.num_groups, weight=weight_updated, bias=self.bias, eps=self.eps)


if __name__ == "__main__":
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

    def make_svd_conv2d_for_test(in_channels, out_channels, kernel_size, scale=1.0, **kwargs):
        conv = SVDConv2d.__new__(SVDConv2d)
        nn.Conv2d.__init__(conv, in_channels, out_channels, kernel_size, **kwargs)
        assert type(kernel_size) is int
        weight_reshaped = rearrange(conv.weight, 'co cin h w -> co (cin h w)')
        conv.U, conv.S, conv.Vh = torch.linalg.svd(weight_reshaped.type(torch.FloatTensor), full_matrices=False)
        conv.delta = nn.Parameter(torch.zeros_like(conv.S))
        conv.weight.requires_grad = False
        conv.done_svd = True
        conv.scale = scale
        return conv

    print("Running SAVE benchmark checks...")
    torch.manual_seed(17)

    try:
        layer = SVDLinear(6, 4, scale=0.5)
        x = torch.randn(5, 6, requires_grad=True)
        y = layer(x)
        check("svd_linear_functionality", y is not None)
        if y is None:
            skip_checks(3, "SVDLinear returned None")
        else:
            check("svd_linear_shape", tuple(y.shape) == (5, 4), str(tuple(y.shape)))
            check("svd_linear_finite", torch.isfinite(y).all().item())
            y.sum().backward()
            grad_ok = layer.delta.grad is not None and torch.isfinite(layer.delta.grad).all().item()
            frozen_ok = layer.weight.requires_grad is False
            check("svd_linear_delta_gradient_and_frozen_weight", grad_ok and frozen_ok)
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"SVDLinear checks raised {type(exc).__name__}: {exc}")

    try:
        conv = make_svd_conv2d_for_test(3, 5, 3, scale=0.25, padding=1)
        image = torch.randn(2, 3, 8, 8, requires_grad=True)
        z = conv(image)
        check("svd_conv2d_functionality", z is not None)
        if z is None:
            skip_checks(3, "SVDConv2d returned None")
        else:
            check("svd_conv2d_shape", tuple(z.shape) == (2, 5, 8, 8), str(tuple(z.shape)))
            check("svd_conv2d_finite", torch.isfinite(z).all().item())
            z.mean().backward()
            grad_ok = conv.delta.grad is not None and torch.isfinite(conv.delta.grad).all().item()
            frozen_ok = conv.weight.requires_grad is False
            check("svd_conv2d_delta_gradient_and_frozen_weight", grad_ok and frozen_ok)
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"SVDConv2d checks raised {type(exc).__name__}: {exc}")

    try:
        emb = SVDEmbedding(10, 4, scale=1.5)
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
        e = emb(ids)
        check("svd_embedding_functionality", e is not None)
        if e is None:
            skip_checks(3, "SVDEmbedding returned None")
        else:
            check("svd_embedding_shape", tuple(e.shape) == (2, 3, 4), str(tuple(e.shape)))
            check("svd_embedding_finite", torch.isfinite(e).all().item())
            e.sum().backward()
            grad_ok = emb.delta.grad is not None and torch.isfinite(emb.delta.grad).all().item()
            frozen_ok = emb.weight.requires_grad is False
            check("svd_embedding_delta_gradient_and_frozen_weight", grad_ok and frozen_ok)
    except Exception as exc:
        traceback.print_exc()
        skip_checks(4, f"SVDEmbedding checks raised {type(exc).__name__}: {exc}")

    print(f"Checks passed: {passed}")
    print(f"Checks failed: {failed}")
    if failed != 0:
        raise SystemExit(1)
