# -*- coding: utf-8 -*-
"""Core model components extracted from heartheflow.

This file contains only the Hear The Flow model architecture from model.py.
Lightning modules, corpus handling, testing scripts, weight-file loading, and
metric-orchestration code are intentionally excluded.
"""

import torch
import torch.nn.functional as F
from torchvision.models import resnet18
from torch import nn
import math


class HearTheFlowVSSLModel(nn.Module):
    def __init__(self, args):
        super(HearTheFlowVSSLModel, self).__init__()
        self.args = args
        self.tau = self.args.tau
        self.flowtype = self.args.flowtype
        self.freeze_vision = self.args.freeze_vision
        self.trimap = self.args.trimap
        self.pretrain_flow = True if self.args.pretrain_flow else False
        self.pretrain_vision = True if self.args.pretrain_vision else False
        self.logit_temperature = self.args.logit_temperature

        # Vision model
        self.imgnet = resnet18(pretrained=self.pretrain_vision)
        self.imgnet.avgpool = nn.Identity()
        self.imgnet.fc = nn.Identity()

        # Audio model
        self.audnet = resnet18()
        # Fix first layer channel
        self.audnet.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.audnet.avgpool = nn.AdaptiveMaxPool2d((1, 1))
        self.audnet.fc = nn.Identity()

        # Flow model
        if self.flowtype == 'cnn':
            self.flownet = resnet18(pretrained=self.pretrain_flow)
            # Fix first layer channel
            self.flownet.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.flownet.avgpool = nn.Identity()
            self.flownet.fc = nn.Identity()
            self.flowatt = Self_Attn(512, 512)
        elif self.flowtype == 'maxpool':
            self.flownet = nn.AdaptiveMaxPool2d((7,7))
            self.flowatt = Self_Attn(512, 2)

        self.m = nn.Sigmoid()
        self.epsilon = self.args.epsilon
        self.epsilon2 = self.args.epsilon - self.args.epsilon_margin

        for m in self.audnet.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def unfreeze_vision(self, grad):
        for param in self.imgnet.parameters():
            param.requires_grad = grad

    def lvs_loss(self, img, aud):
        """
        TODO: Compute the self-supervised localization-via-similarity loss.

        Input:
            img: Tensor of shape (batch, channels, height, width), containing
                normalized attended visual feature maps.
            aud: Tensor of shape (batch, channels), containing normalized audio
                embeddings for the same batch.

        Output:
            Tuple:
            - loss: scalar tensor used to separate matched audio-visual pairs from
              mismatched and low-response regions.
            - localization: Tensor of shape (batch, height, width), containing the
              matched audio-visual similarity map.

"""
        pass

    def forward(self, image, flow, audio):
        """
        TODO: Run the Hear The Flow audiovisual localization model.

        Input:
            image: Tensor of shape (batch, 3, image_height, image_width).
            flow: Tensor of shape (batch, 2, flow_height, flow_width).
            audio: Tensor of shape (batch, 1, spectrogram_height, spectrogram_width).

        Output:
            Tuple:
            - loss: scalar localization-via-similarity loss.
            - localization: Tensor of shape (batch, 7, 7), containing spatial
              audio-visual localization logits.

"""
        pass


class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self,in_dim, key_in_dim):
        super(Self_Attn,self).__init__()
        self.chanel_in = in_dim
        
        self.query_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = key_in_dim , out_channels = in_dim//8 , kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim , out_channels = in_dim , kernel_size= 1)

        self.softmax  = nn.Softmax(dim=-1)

    def forward(self, x, v):
        """
        TODO: Compute cross-attention from visual features and flow features.

        Input:
            x: Tensor of shape (batch, visual_channels, width, height), used for
                query and value projections.
            v: Tensor of shape (batch, flow_channels, width, height), used for key
                projection.

        Output:
            Tuple:
            - out: Tensor of shape (batch, visual_channels, width, height), containing
              attention-weighted visual features.
            - attention: Tensor of shape (batch, width * height, width * height),
              containing spatial attention weights.

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
    print("heartheflow: optical-flow-guided audiovisual localization")
    print("=" * 70)

    class Args:
        tau = 0.03
        flowtype = "cnn"
        freeze_vision = 0
        trimap = 1
        pretrain_flow = 0
        pretrain_vision = 0
        logit_temperature = 0.07
        epsilon = 0.65
        epsilon_margin = 0.05

    class TinyResNet(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=1)
            self.avgpool = nn.Identity()
            self.fc = nn.Identity()
            self.image_proj = nn.Linear(3, 512 * 7 * 7)
            self.flow_proj = nn.Linear(2, 512 * 7 * 7)
            self.audio_proj = nn.Linear(1, 512)

        def forward(self, x):
            pooled = x.mean(dim=(2, 3))
            if x.size(1) == 1:
                return self.audio_proj(pooled)
            if x.size(1) == 2:
                return self.flow_proj(pooled)
            return self.image_proj(pooled)

    original_resnet18 = resnet18
    globals()["resnet18"] = TinyResNet

    try:
        print("-" * 70)
        print("[Test 1/4] Self_Attn.forward")
        try:
            attn = Self_Attn(in_dim=8, key_in_dim=2)
            x = torch.randn(2, 8, 4, 4, requires_grad=True)
            v = torch.randn(2, 2, 4, 4)
            out, attention = attn(x, v)
            check("Self_Attn output not None", out is not None and attention is not None)
            if out is not None and attention is not None:
                check("Self_Attn output shape", out.shape == (2, 8, 4, 4), str(tuple(out.shape)))
                check("Self_Attn attention shape", attention.shape == (2, 16, 16), str(tuple(attention.shape)))
                check("Self_Attn outputs finite", torch.isfinite(out).all().item() and torch.isfinite(attention).all().item())
                check("Self_Attn attention rows sum to one", torch.allclose(attention.sum(dim=-1), torch.ones(2, 16), atol=1e-5), str(attention.sum(dim=-1)[0, :3]))
                out.sum().backward()
                check("Self_Attn value branch receives gradients", attn.value_conv.weight.grad is not None and torch.isfinite(attn.value_conv.weight.grad).all().item())
            else:
                skip_checks(5, "Self_Attn returned incomplete outputs")
        except Exception as exc:
            skip_checks(6, f"Self_Attn forward raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 2/4] HearTheFlowVSSLModel.lvs_loss")
        try:
            args = Args()
            args.flowtype = "maxpool"
            model = HearTheFlowVSSLModel(args)
            img = torch.randn(3, 512, 7, 7, requires_grad=True)
            aud = F.normalize(torch.randn(3, 512), dim=1)
            loss, localization = model.lvs_loss(img, aud)
            check("lvs_loss output not None", loss is not None and localization is not None)
            if loss is not None and localization is not None:
                check("lvs_loss scalar loss", loss.dim() == 0, str(tuple(loss.shape)))
                check("lvs_loss localization shape", localization.shape == (3, 7, 7), str(tuple(localization.shape)))
                check("lvs_loss outputs finite", torch.isfinite(loss).item() and torch.isfinite(localization).all().item())
                check("lvs_loss builds batch mask", hasattr(model, "mask") and model.mask.shape == (3, 3), str(getattr(model, "mask", None)))
                loss.backward()
                check("lvs_loss keeps gradients to image features", img.grad is not None and torch.isfinite(img.grad).all().item())
            else:
                skip_checks(5, "lvs_loss returned incomplete outputs")
        except Exception as exc:
            skip_checks(6, f"lvs_loss raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 3/4] HearTheFlowVSSLModel.forward with cnn flow")
        try:
            args = Args()
            args.flowtype = "cnn"
            model = HearTheFlowVSSLModel(args)
            image = torch.randn(2, 3, 32, 32)
            flow = torch.randn(2, 2, 32, 32)
            audio = torch.randn(2, 1, 32, 32)
            loss, localization = model(image, flow, audio)
            check("cnn forward output not None", loss is not None and localization is not None)
            if loss is not None and localization is not None:
                check("cnn forward scalar loss", loss.dim() == 0, str(tuple(loss.shape)))
                check("cnn forward localization shape", localization.shape == (2, 7, 7), str(tuple(localization.shape)))
                check("cnn forward outputs finite", torch.isfinite(loss).item() and torch.isfinite(localization).all().item())
                check("cnn forward uses 512-channel flow attention", model.flowatt.key_conv.in_channels == 512, str(model.flowatt.key_conv.in_channels))
                model.unfreeze_vision(False)
                check("unfreeze_vision disables image gradients", all(not param.requires_grad for param in model.imgnet.parameters()))
                model.unfreeze_vision(True)
                check("unfreeze_vision enables image gradients", all(param.requires_grad for param in model.imgnet.parameters()))
            else:
                skip_checks(6, "cnn forward returned incomplete outputs")
        except Exception as exc:
            skip_checks(7, f"cnn forward raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 4/4] HearTheFlowVSSLModel.forward with maxpool flow")
        try:
            args = Args()
            args.flowtype = "maxpool"
            model = HearTheFlowVSSLModel(args)
            image = torch.randn(2, 3, 32, 32)
            flow = torch.randn(2, 2, 32, 32)
            audio = torch.randn(2, 1, 32, 32)
            loss, localization = model(image, flow, audio)
            check("maxpool forward output not None", loss is not None and localization is not None)
            if loss is not None and localization is not None:
                check("maxpool forward scalar loss", loss.dim() == 0, str(tuple(loss.shape)))
                check("maxpool forward localization shape", localization.shape == (2, 7, 7), str(tuple(localization.shape)))
                check("maxpool forward outputs finite", torch.isfinite(loss).item() and torch.isfinite(localization).all().item())
                check("maxpool forward uses two-channel flow attention", model.flowatt.key_conv.in_channels == 2, str(model.flowatt.key_conv.in_channels))
                check("maxpool flow backbone returns 7x7 map", tuple(model.flownet(flow).shape[-2:]) == (7, 7), str(tuple(model.flownet(flow).shape)))
            else:
                skip_checks(5, "maxpool forward returned incomplete outputs")
        except Exception as exc:
            skip_checks(6, f"maxpool forward raised {type(exc).__name__}: {exc}")
    finally:
        globals()["resnet18"] = original_resnet18

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
