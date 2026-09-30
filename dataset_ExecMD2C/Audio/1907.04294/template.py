# -*- coding: utf-8 -*-
"""Core attention model components extracted from AttentionMIC.

This file contains the decision-level attention architecture from Attention.py
and the direct parameter-count helper from model.py. Baseline-only models,
experiment orchestration, corpus handling, logging, and file-save routines are
intentionally excluded.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def count_parameters(model):
    total_param = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            num_param = np.prod(param.size())
            # if param.dim() > 1:
            #     print(name, ':', 'x'.join(str(x) for x in list(param.size())), '=', num_param)
            # else:
            #     print(name, ':', num_param)
            total_param += num_param
    # print('Total Parameters: {}'.format(total_param))
    return total_param


def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    bn.weight.data.fill_(1.)


class Attention(nn.Module):
    def __init__(self, n_in, n_out):
        super(Attention, self).__init__()

        self.att = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.cla = nn.Conv2d(
            in_channels=n_in, out_channels=n_out, kernel_size=(
                1, 1), stride=(
                1, 1), padding=(
                0, 0), bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.att,)
        init_layer(self.cla)

    def forward(self, x):
        """
        TODO: Compute decision-level attention pooling over time.

        Input:
            x: Tensor of shape (samples_num, feature_channels, time_steps, 1).

        Output:
            Tensor of shape (samples_num, classes_num), with one bounded score per
            class.

"""
        pass


class EmbeddingLayers(nn.Module):

    def __init__(self, freq_bins, emb_layers, hidden_units, drop_rate):
        super(EmbeddingLayers, self).__init__()

        self.freq_bins = freq_bins
        self.hidden_units = hidden_units
        self.drop_rate = drop_rate

        self.conv1x1 = nn.ModuleList()
        self.batchnorm = nn.ModuleList()

        for i in range(emb_layers):
            in_channels = freq_bins if i == 0 else hidden_units
            conv = nn.Conv2d(
                in_channels=in_channels, out_channels=hidden_units,
                kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
            self.conv1x1.append(conv)
            self.batchnorm.append(nn.BatchNorm2d(in_channels))

        # Append last batch-norm layer
        self.batchnorm.append(nn.BatchNorm2d(hidden_units))

        self.init_weights()

    def init_weights(self):

        for conv in self.conv1x1:
            init_layer(conv)

        for bn in self.batchnorm:
            init_bn(bn)

    def forward(self, input, return_layers=False):
        """
        TODO: Transform frame-level frequency features with residual 1x1 embeddings.

        Input:
            input: Tensor of shape (samples_num, time_steps, freq_bins).
            return_layers: Boolean controlling whether to return intermediate layer
                activations instead of the final residual output.

        Output:
            If return_layers is False, a tensor of shape
            (samples_num, hidden_units, time_steps, 1). If return_layers is True, a
            list containing the initial normalized tensor followed by each embedding
            layer output.

"""
        pass


class DecisionLevelSingleAttention(nn.Module):

    def __init__(self, freq_bins, classes_num, emb_layers, hidden_units, drop_rate):

        super(DecisionLevelSingleAttention, self).__init__()

        self.emb = EmbeddingLayers(
            freq_bins=freq_bins,
            emb_layers=emb_layers,
            hidden_units=hidden_units,
            drop_rate=drop_rate)

        self.attention = Attention(
            n_in=hidden_units,
            n_out=classes_num)
            
        self.param_count = count_parameters(self)
        print(self.param_count)

    def init_weights(self):
        pass

    def forward(self, input):
        """
        TODO: Run the complete decision-level attention model.

        Input:
            input: Tensor of shape (samples_num, time_steps, freq_bins).

        Output:
            Tensor of shape (samples_num, classes_num) containing bounded
            multi-label instrument scores.

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
    print("AttentionMIC: decision-level attention for musical instrument recognition")
    print("=" * 70)

    print("-" * 70)
    print("[Test 1/4] initialization helpers and parameter counting")
    try:
        conv = nn.Conv2d(4, 3, kernel_size=(1, 1), bias=True)
        init_layer(conv)
        bn = nn.BatchNorm2d(4)
        init_bn(bn)
        model = Attention(n_in=4, n_out=3)
        expected_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
        check("init_layer keeps finite weights", torch.isfinite(conv.weight).all().item())
        check("init_layer zeroes bias", torch.allclose(conv.bias, torch.zeros_like(conv.bias)))
        check("init_bn fills weights with one", torch.allclose(bn.weight, torch.ones_like(bn.weight)))
        check("count_parameters matches manual count", count_parameters(model) == expected_params, f"{count_parameters(model)} vs {expected_params}")
    except Exception as exc:
        skip_checks(4, f"initialization helpers raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 2/4] Attention.forward")
    try:
        attention = Attention(n_in=4, n_out=3)
        attention.eval()
        with torch.no_grad():
            attention.att.weight.zero_()
            attention.att.bias.zero_()
            attention.cla.weight.zero_()
            attention.cla.bias.copy_(torch.tensor([-2.0, 0.0, 2.0]))
        x = torch.randn(2, 4, 5, 1, requires_grad=True)
        y = attention(x)
        expected = torch.sigmoid(torch.tensor([-2.0, 0.0, 2.0])).repeat(2, 1)
        check("Attention output not None", y is not None)
        if y is not None:
            check("Attention output shape", y.shape == (2, 3), str(tuple(y.shape)))
            check("Attention output finite", torch.isfinite(y).all().item())
            check("Attention output bounded", ((y >= 0.0) & (y <= 1.0)).all().item())
            check("Attention uniform weights average class branch", torch.allclose(y, expected, atol=1e-6), str(y))
            y.sum().backward()
            check("Attention class branch receives gradients", attention.cla.bias.grad is not None and torch.isfinite(attention.cla.bias.grad).all().item())
        else:
            skip_checks(5, "Attention returned None")
    except Exception as exc:
        skip_checks(6, f"Attention forward raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 3/4] EmbeddingLayers.forward")
    try:
        emb = EmbeddingLayers(freq_bins=8, emb_layers=2, hidden_units=8, drop_rate=0.0)
        emb.eval()
        x = torch.randn(2, 6, 8)
        y = emb(x)
        layers = emb(x, return_layers=True)
        with torch.no_grad():
            for conv in emb.conv1x1:
                conv.weight.zero_()
        residual_y = emb(x)
        expected_residual = x.transpose(1, 2)[:, :, :, None].contiguous()
        check("EmbeddingLayers output not None", y is not None)
        if y is not None:
            check("EmbeddingLayers output shape", y.shape == (2, 8, 6, 1), str(tuple(y.shape)))
            check("EmbeddingLayers output finite", torch.isfinite(y).all().item())
            check("EmbeddingLayers return_layers length", isinstance(layers, list) and len(layers) == 3, str(type(layers)))
            check("EmbeddingLayers return_layers first shape", layers[0].shape == (2, 8, 6, 1), str(tuple(layers[0].shape)))
            check("EmbeddingLayers residual path is preserved", torch.allclose(residual_y, expected_residual, atol=1e-5), str((residual_y - expected_residual).abs().max().item()))
        else:
            skip_checks(5, "EmbeddingLayers returned None")
    except Exception as exc:
        skip_checks(6, f"EmbeddingLayers forward raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 4/4] DecisionLevelSingleAttention.forward")
    try:
        model = DecisionLevelSingleAttention(
            freq_bins=8,
            classes_num=4,
            emb_layers=2,
            hidden_units=8,
            drop_rate=0.0,
        )
        model.eval()
        x = torch.randn(2, 7, 8, requires_grad=True)
        y = model(x)
        check("DecisionLevelSingleAttention output not None", y is not None)
        if y is not None:
            check("DecisionLevelSingleAttention output shape", y.shape == (2, 4), str(tuple(y.shape)))
            check("DecisionLevelSingleAttention output finite", torch.isfinite(y).all().item())
            check("DecisionLevelSingleAttention output bounded", ((y >= 0.0) & (y <= 1.0)).all().item())
            check("DecisionLevelSingleAttention stores parameter count", model.param_count == count_parameters(model), f"{model.param_count} vs {count_parameters(model)}")
            y.sum().backward()
            grad_ok = any(param.grad is not None for param in model.parameters() if param.requires_grad)
            check("DecisionLevelSingleAttention supports gradient flow", grad_ok)
        else:
            skip_checks(5, "DecisionLevelSingleAttention returned None")
    except Exception as exc:
        skip_checks(6, f"DecisionLevelSingleAttention forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
