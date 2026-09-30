# ============================================================
# ground_truth.py - AraSpot Core Model Components
# Source:
#   Speech/AraSpot-main/layers.py
#   Speech/AraSpot-main/models.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import math
import torch
import torch.nn as nn
from typing import List
from torch import Tensor


# --- [Original file: layers.py] ---
class FeedForwardModule(nn.Module):
    """Implements the feed forward module in the conformer block
    where the module consists of the below
    1. Layer Norm
    2. Linear Layer
    3. Swish Activation
    4. Dropout
    5. Linear Layer
    6. Dropout
    Args:
        enc_dim (int): The encoder dimensionality
        scaling_factor (int): The scaling factor of the linear layer
        p_dropout (float): The dropout probability
        residual_scaler (float, optional): The residual scaling.
        Defaults to 0.5.
    """
    def __init__(
            self,
            enc_dim: int,
            scaling_factor: int,
            p_dropout: float,
            residual_scaler=0.5
            ) -> None:
        super().__init__()
        self.residual_scaler = residual_scaler
        scaled_dim = scaling_factor * enc_dim
        self.lnorm = nn.LayerNorm(enc_dim)
        self.fc1 = nn.Linear(
            in_features=enc_dim,
            out_features=scaled_dim
        )
        self.fc2 = nn.Linear(
            in_features=scaled_dim,
            out_features=enc_dim
        )
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)

    def forward(self, inp: Tensor) -> Tensor:
        """Passes the given inp through the feed forward
        module
        Args:
            inp (Tensor): the input to the feed forward module
            with shape [B, M, N] where B is the batch size, M
            is the maximum length, and N is the encoder dim
        Returns:
            Tensor: The result of the forward module
        """
        out = self.lnorm(inp)
        out = self.fc1(out)
        out = self.swish(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return self.residual_scaler * inp + out


class ConvModule(nn.Module):
    """Implements the convolution module
    where it contains the following layers
    1. Layernorm
    2. Pointwise Conv
    3. Gate Linear unit
    4. 1D Depthwise conv
    5. BatchNorm
    6. Swish Activation
    7. Pointwise Conv
    8. Dropout
    Args:
        enc_dim (int): The encoder dimensionality.
        scaling_factor (int): The scaling factor of the conv layer.
        kernel_size (int): The convolution kernel size.
        p_dropout (float): The dropout probability.
    """
    def __init__(
            self,
            enc_dim: int,
            scaling_factor: int,
            kernel_size: int,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.lnorm = nn.LayerNorm(enc_dim)
        n_scaled_channels = enc_dim * scaling_factor
        assert (kernel_size - 1) % 2 == 0, 'kernel_size - 1 \
            must be divisable by 2 -odd'
        padding_size = (kernel_size - 1) // 2
        self.pwise_conv1 = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=n_scaled_channels,
            kernel_size=1
        )
        self.glu = nn.GLU(dim=1)
        self.dwise_conv = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=kernel_size,
            padding=padding_size,
            groups=enc_dim
        )
        self.bnorm = nn.BatchNorm1d(enc_dim)
        self.swish = nn.SiLU()
        self.dropout = nn.Dropout(p_dropout)
        self.pwise_conv2 = nn.Conv1d(
            in_channels=enc_dim,
            out_channels=enc_dim,
            kernel_size=1
        )

    def forward(self, inp: Tensor) -> Tensor:
        out = self.lnorm(inp)
        out = out.permute(0, 2, 1)
        out = self.pwise_conv1(out)
        out = self.glu(out)
        out = self.dwise_conv(out)
        out = self.bnorm(out)
        out = self.swish(out)
        out = self.pwise_conv2(out)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)
        return out + inp


class MHSA(nn.Module):
    def __init__(
            self,
            enc_dim: int,
            h: int,
            p_dropout: float,
            device: str
            ) -> None:
        super().__init__()
        assert enc_dim % h == 0, 'enc_dim is not divisible by h'
        self.fc_key = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.fc_query = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.fc_value = nn.Linear(
            in_features=enc_dim,
            out_features=enc_dim,
        )
        self.proj_fc = nn.Linear(
            in_features=2 * enc_dim,
            out_features=enc_dim,
        )
        self.lnorm = nn.LayerNorm(enc_dim)
        self.dropout = nn.Dropout(p_dropout)
        self.enc_dim = enc_dim
        self.h = h
        self.dk = enc_dim // h
        self.sqrt_dk = math.sqrt(self.dk)
        self.softmax = nn.Softmax(dim=-1)
        self.device = device

    def _get_scaled_att(
            self,
            Q: Tensor,
            K: Tensor
            ) -> Tensor:
        """Calculates the scaled attention map
        by calculating softmax(matmul(Q, K.T)/sqrt(dk))
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
        Returns:
            Tensor: The scaled attention weights of shape
            [B * h, Tq, Tk]
        """
        result = torch.matmul(Q, K)
        result = result / self.sqrt_dk
        return self.softmax(result)

    def perform_att(
            self,
            Q: Tensor,
            K: Tensor,
            V: Tensor
            ) -> Tensor:
        """Performs multi-head scaled attention
        by calculating softmax(matmul(Q, K.T)/sqrt(dk)).V
        Args:
            Q (Tensor): The Query tensor of shape [h * B, Tq, dk]
            K (Tensor): The Key tensor of shape [h * B, dk, Tk]
            V (Tensor): The Value tensor of shape [h * B, Tk, dk]
        Returns:
            Tuple[Tensor, Tensor]: The attention matrix of shape
            [B * h, Tq, Tk] and the scaled attention value of
            shape [B * h, Tq, dk].
        """
        att = self._get_scaled_att(Q, K)
        result = torch.matmul(att, V)
        return att, result

    def _reshape(self, *args) -> List[Tensor]:
        """Reshabes all the given list of tensor
        from [B, T, N] to [B, T, h, dk]
        Returns:
            List[Tensor]: list of all reshaped tensors
        """
        return [
            item.contiguous().view(-1, item.shape[1], self.h, self.dk)
            for item in args
        ]

    def _pre_permute(self, *args) -> List[Tensor]:
        """Permutes all the given list of tensors
        from [B, T, h, dk] to become [h, B, T, dk].
        Returns:
            List[Tensor]: List of all permuted tensors.
        """
        return [
            item.permute(2, 0, 1, 3)
            for item in args
        ]

    def _change_dim(self, *args) -> List[Tensor]:
        """Changes the dimensionality of all passed tensores
        from [B, T, N] to [B * h, T, dk]
        Returns:from functools import lru_cache
            List[Tensor]: List of the modified tensors.
        """
        result = self._reshape(*args)  # [B, T, h, dk]
        result = self._pre_permute(*result)  # [h, B, T, dk]
        return [
            item.permute(1, 0, 2, 3).contiguous().view(
                -1, item.shape[2], item.shape[3]
                )
            for item in result
        ]

    def forward(self, inp: Tensor) -> Tensor:
        """Passes the input into multi-head attention
        Args:
            inp (Tensor): The input tensor
        Returns:
            Tensor: The result after adding it to positionals
            and passing it through multi-head self-attention
        """
        out = self.lnorm(inp)
        [b, s, _] = inp.shape
        K = self.fc_key(inp)
        Q = self.fc_query(inp)
        V = self.fc_value(inp)
        (Q, K, V) = self._change_dim(Q, K, V)  # [h * B, T, dk]
        K = K.permute(0, 2, 1)  # [h, T, B, dk]
        _, result = self.perform_att(Q, K, V)
        result = result.view(b, self.h, s, self.dk)
        result = result.permute(0, 2, 1, 3)
        result = result.contiguous().view(b, s, -1)
        result = torch.cat([inp, result], dim=-1)
        result = self.proj_fc(result)
        out = self.dropout(result)
        return inp + out


class ConformerBlock(nn.Module):
    def __init__(
            self,
            enc_dim: int,
            h: int,
            kernel_size: int,
            scaling_factor: int,
            residual_scaler: float,
            device: str,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.ff1 = FeedForwardModule(
            enc_dim=enc_dim,
            scaling_factor=scaling_factor,
            p_dropout=p_dropout,
            residual_scaler=residual_scaler
        )
        self.mhsa = MHSA(
            enc_dim=enc_dim,
            h=h, p_dropout=p_dropout, device=device
        )
        self.conv = ConvModule(
            enc_dim=enc_dim,
            scaling_factor=scaling_factor,
            kernel_size=kernel_size,
            p_dropout=p_dropout
            )
        self.ff2 = FeedForwardModule(
            enc_dim=enc_dim,
            scaling_factor=scaling_factor,
            p_dropout=p_dropout,
            residual_scaler=residual_scaler
        )
        self.lnorm = nn.LayerNorm(enc_dim)

    def forward(self, inp: Tensor):
        out = self.ff1(inp)
        out = self.mhsa(out)
        out = self.conv(out)
        out = self.ff2(out)
        out = self.lnorm(out)
        return out


class Conformer(nn.Module):
    def __init__(
            self,
            n_layers: int,
            enc_dim: int,
            h: int,
            kernel_size: int,
            scaling_factor: int,
            residual_scaler: float,
            device: str,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                enc_dim=enc_dim,
                h=h,
                kernel_size=kernel_size,
                scaling_factor=scaling_factor,
                residual_scaler=residual_scaler,
                device=device,
                p_dropout=p_dropout
            )
            for _ in range(n_layers)
        ])

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# --- [Original file: models.py] ---
class ConformerGRU(nn.Module):
    def __init__(
            self,
            feat_size: int,
            n_layers: int,
            enc_dim: int,
            h: int,
            kernel_size: int,
            scaling_factor: int,
            residual_scaler: float,
            bidirectional: bool,
            n_classes: int,
            device: str,
            p_dropout: float
            ) -> None:
        super().__init__()
        self.conf = Conformer(
            n_layers=n_layers,
            enc_dim=enc_dim,
            h=h,
            kernel_size=kernel_size,
            scaling_factor=scaling_factor,
            residual_scaler=residual_scaler,
            device=device,
            p_dropout=p_dropout
        )
        self.gru = nn.GRU(
            input_size=enc_dim,
            hidden_size=enc_dim,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc0 = nn.Linear(
            in_features=2 * enc_dim if bidirectional else enc_dim,
            out_features=4 * enc_dim if bidirectional else 2 * enc_dim
        )
        self.pred_fc = nn.Linear(
            in_features=4 * enc_dim if bidirectional else 2 * enc_dim,
            out_features=n_classes
        )
        self.fc = nn.Linear(
            in_features=feat_size, out_features=enc_dim
            )

    def forward(self, x: Tensor) -> Tensor:
        out = self.fc(x)
        out = self.conf(out)
        out, h = self.gru(out)
        h = h.permute(1, 0, 2).contiguous().view(x.shape[0], -1)
        h = self.fc0(h)
        return self.pred_fc(h)


# ============================================================
# __main__: Automated test suite for 5 ablated functions
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

    def build_block_kwargs():
        return {
            "enc_dim": 8,
            "h": 2,
            "kernel_size": 3,
            "scaling_factor": 2,
            "residual_scaler": 0.5,
            "device": "cpu",
            "p_dropout": 0.0,
        }

    def build_model():
        return ConformerGRU(
            feat_size=6,
            n_layers=2,
            enc_dim=8,
            h=2,
            kernel_size=3,
            scaling_factor=2,
            residual_scaler=0.5,
            bidirectional=True,
            n_classes=4,
            device="cpu",
            p_dropout=0.0,
        )

    print("=" * 70)
    print("AraSpot: ConformerGRU Arabic Spoken Command Spotting")
    print("Automated reproduction benchmark - 5 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/5: FeedForwardModule.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] FeedForwardModule.forward - macaron feed-forward residual branch")
    try:
        module = FeedForwardModule(enc_dim=8, scaling_factor=2, p_dropout=0.0, residual_scaler=0.5).to(device)
        x = torch.randn(2, 5, 8, device=device, requires_grad=True)
        y = module(x)
        check("FeedForward output not None", y is not None)
        if y is not None:
            check("FeedForward output shape", tuple(y.shape) == (2, 5, 8), f"expected (2, 5, 8), got {tuple(y.shape)}")
            check("FeedForward output finite", torch.isfinite(y).all().item())
            zero_module = FeedForwardModule(enc_dim=8, scaling_factor=2, p_dropout=0.0, residual_scaler=0.5).to(device)
            with torch.no_grad():
                zero_module.fc1.weight.zero_()
                zero_module.fc1.bias.zero_()
                zero_module.fc2.weight.zero_()
                zero_module.fc2.bias.zero_()
            zero_y = zero_module(x.detach())
            check("FeedForward preserves residual scaling convention", torch.allclose(zero_y, 0.5 * x.detach(), atol=1e-6))
            y.sum().backward()
            check("FeedForward routes gradient to input", x.grad is not None and x.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "FeedForwardModule.forward returned None")
    except Exception as e:
        skip_checks(5, f"FeedForwardModule.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/5: ConvModule.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] ConvModule.forward - Conformer convolution branch")
    try:
        module = ConvModule(enc_dim=8, scaling_factor=2, kernel_size=3, p_dropout=0.0).to(device)
        module.eval()
        x = torch.randn(2, 7, 8, device=device, requires_grad=True)
        y = module(x)
        check("ConvModule output not None", y is not None)
        if y is not None:
            check("ConvModule output shape", tuple(y.shape) == (2, 7, 8), f"expected (2, 7, 8), got {tuple(y.shape)}")
            check("ConvModule output finite", torch.isfinite(y).all().item())
            zero_module = ConvModule(enc_dim=8, scaling_factor=2, kernel_size=3, p_dropout=0.0).to(device)
            zero_module.eval()
            with torch.no_grad():
                zero_module.pwise_conv2.weight.zero_()
                zero_module.pwise_conv2.bias.zero_()
            zero_y = zero_module(x.detach())
            check("ConvModule preserves residual identity when conv branch is zeroed", torch.allclose(zero_y, x.detach(), atol=1e-6))
            y.sum().backward()
            check("ConvModule routes gradient through depthwise branch", module.dwise_conv.weight.grad is not None and module.dwise_conv.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "ConvModule.forward returned None")
    except Exception as e:
        skip_checks(5, f"ConvModule.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/5: MHSA.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] MHSA.forward - multi-head self-attention with residual projection")
    try:
        module = MHSA(enc_dim=8, h=2, p_dropout=0.0, device="cpu").to(device)
        x = torch.randn(2, 6, 8, device=device, requires_grad=True)
        y = module(x)
        check("MHSA output not None", y is not None)
        if y is not None:
            check("MHSA output shape", tuple(y.shape) == (2, 6, 8), f"expected (2, 6, 8), got {tuple(y.shape)}")
            check("MHSA output finite", torch.isfinite(y).all().item())
            zero_module = MHSA(enc_dim=8, h=2, p_dropout=0.0, device="cpu").to(device)
            with torch.no_grad():
                zero_module.proj_fc.weight.zero_()
                zero_module.proj_fc.bias.zero_()
            zero_y = zero_module(x.detach())
            check("MHSA residual path is identity when projection is zeroed", torch.allclose(zero_y, x.detach(), atol=1e-6))
            y.sum().backward()
            check("MHSA routes gradient to projection weights", module.proj_fc.weight.grad is not None and module.proj_fc.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "MHSA.forward returned None")
    except Exception as e:
        skip_checks(5, f"MHSA.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/5: ConformerBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] ConformerBlock.forward - FFN/MHSA/Conv/FFN block assembly")
    try:
        module = ConformerBlock(**build_block_kwargs()).to(device)
        x = torch.randn(2, 6, 8, device=device, requires_grad=True)
        y = module(x)
        check("ConformerBlock output not None", y is not None)
        if y is not None:
            check("ConformerBlock output shape", tuple(y.shape) == (2, 6, 8), f"expected (2, 6, 8), got {tuple(y.shape)}")
            check("ConformerBlock output finite", torch.isfinite(y).all().item())
            token_means = y.mean(dim=-1)
            check("ConformerBlock final layer norm centers token features", token_means.abs().max().item() < 1e-5)
            (y ** 2).sum().backward()
            check("ConformerBlock routes gradient to first FFN", module.ff1.fc1.weight.grad is not None and module.ff1.fc1.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "ConformerBlock.forward returned None")
    except Exception as e:
        skip_checks(5, f"ConformerBlock.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 5/5: ConformerGRU.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] ConformerGRU.forward - top-level AraSpot classifier data flow")
    try:
        model = build_model().to(device)
        x = torch.randn(2, 9, 6, device=device, requires_grad=True)
        y = model(x)
        check("ConformerGRU output not None", y is not None)
        if y is not None:
            check("ConformerGRU output shape", tuple(y.shape) == (2, 4), f"expected (2, 4), got {tuple(y.shape)}")
            check("ConformerGRU output finite", torch.isfinite(y).all().item())
            check("ConformerGRU bidirectional hidden flatten feeds fc0", model.fc0.in_features == 16 and model.pred_fc.out_features == 4)
            y.sum().backward()
            check("ConformerGRU routes gradient to feature projection", model.fc.weight.grad is not None and model.fc.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "ConformerGRU.forward returned None")
    except Exception as e:
        skip_checks(5, f"ConformerGRU.forward raised {type(e).__name__}: {e}")
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
        print(f"{failed} check(s) FAILED - some ablated functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
