# ============================================================
# ground_truth.py - Lifelong-MonoDepth Core Model Components
# Source: Lifelong-MonoDepth-main
#
# Contains ONLY the shared encoder, multi-scale fusion, and
# domain-specific depth/uncertainty heads. No training loops,
# datasets, evaluation scripts, downloaded checkpoints, or CLI code.
# ============================================================

from collections import OrderedDict
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo


# --- [Original file: models/resnet.py] ---
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101',
    'resnet152': 'https://download.pytorch.org/models/resnet152',
}


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'], 'pretrained_model/encoder'))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], 'pretrained_model/encoder'))
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50'], 'pretrained_model/encoder'))
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model


# --- [Original file: models/modules.py] ---
class _UpProjection(nn.Sequential):

    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()
        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
 
    def forward(self, x, size):
        x = self.relu(self.bn1(self.conv1(x)))
        x = F.upsample(x, size=size, mode='bilinear', align_corners=True)

        return x


class E_resnet(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4



class MFF(nn.Module):
    def __init__(self, block_channel, num_features=64):
        super(MFF, self).__init__()

        self.att1 = self.attention(block_channel[0], block_channel[0] // 16)
        self.att2 = self.attention(block_channel[1], block_channel[1] // 16)
        self.att3 = self.attention(block_channel[2], block_channel[2] // 16)
        self.att4 = self.attention(block_channel[3], block_channel[3] // 16)

        self.up1 = _UpProjection(num_input_features=block_channel[
                                 0], num_output_features=16)
        self.up2 = _UpProjection(num_input_features=block_channel[
                                 1], num_output_features=16)
        self.up3 = _UpProjection(num_input_features=block_channel[
                                 2], num_output_features=16)
        self.up4 = _UpProjection(num_input_features=block_channel[
                                 3], num_output_features=16)

        self.conv = nn.Conv2d(num_features, num_features,
                              kernel_size=5, stride=1, padding=2, bias=False)
        self.bn = nn.BatchNorm2d(num_features)

    def attention(self, features1, features2):
        prior = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        conv1 = nn.Conv2d(features1, features2, kernel_size=1, bias=False)
        relu = nn.ReLU()
        conv2 = nn.Conv2d(features2, features1, kernel_size=1, bias=False)
        sigmoid = nn.Sigmoid()
        return nn.Sequential(prior, conv1, relu, conv2, sigmoid)

    def forward(self, x_block1, x_block2, x_block3, x_block4, size):
        """
        [TODO] Fuse four encoder feature scales into one shared depth representation.

        Input:
            x_block1: (batch, c1, h1, w1) - shallow encoder feature map.
            x_block2: (batch, c2, h2, w2) - second encoder feature map.
            x_block3: (batch, c3, h3, w3) - third encoder feature map.
            x_block4: (batch, c4, h4, w4) - deepest encoder feature map.
            size: [target_height, target_width] - spatial size for all projected features.

        Output:
            (batch, 64, target_height, target_width) - shared fused feature map.

"""
        pass



class R(nn.Module):
    def __init__(self, block_channel):
        super(R, self).__init__()
        num_features = 64
        self.conv0 = nn.Conv2d(num_features, num_features,kernel_size=5, stride=1, padding=2, bias=False)
        self.bn0 = nn.BatchNorm2d(num_features)
        self.conv2 = nn.Conv2d(num_features, 1, kernel_size=5, stride=1, padding=2, bias=True)

    def forward(self, x):
        x0 = self.conv0(x)
        x0 = self.bn0(x0)
        x0 = F.relu(x0)
        out = self.conv2(x0)

        return out



class Uncertainty_depth(nn.Module):
    def __init__(self, block_channel):
        super(Uncertainty_depth, self).__init__()
        self.depth = R(block_channel)
        self.uncertainty = R(block_channel)
        
    def forward(self, x):
        """
        [TODO] Predict both metric depth and uncertainty from the shared fused feature map.

        Input:
            x: (batch, 64, height, width) - shared multi-scale feature map.

        Output:
            tuple:
                depth: (batch, 1, height, width) - domain-specific depth prediction.
                uncertainty: (batch, 1, height, width) - domain-specific uncertainty prediction.

"""
        pass


# --- [Original file: models/net.py] ---
class backbone(nn.Module):
    def __init__(self, Encoder, num_features, block_channel):
        super(backbone, self).__init__()
        self.E = Encoder
        self.MFF = MFF(block_channel)

class model_ll(nn.Module):
    def __init__(self, backbone, num_tasks, block_channel, is_training=False):
        super(model_ll, self).__init__()
        self.E = backbone.E
        self.MFF = backbone.MFF
                
        self.tasks = {}
        self.num_tasks = num_tasks

        for i in range(self.num_tasks):            
            if is_training:
                if((self.num_tasks > 1) and (i != (self.num_tasks-1))):
                    self.tasks[i] = backbone.tasks[i].cuda()
                else:
                    self.tasks[i] = Uncertainty_depth(block_channel).cuda()   
                self.add_module('task'+ str(i),self.tasks[i])
            else:
                self.tasks[i] = Uncertainty_depth(block_channel).cuda()   
                self.add_module('task'+ str(i),self.tasks[i])

        
    def forward(self, x):
        """
        [TODO] Run shared encoding/fusion once and dispatch the result to all domain-specific heads.

        Input:
            x: (batch, 3, height, width) - RGB image tensor.

        Output:
            tuple:
                out: dictionary mapping each task index to a (depth, uncertainty) tuple,
                    each prediction shaped (batch, 1, fused_height, fused_width).
                x_mff: (batch, 64, fused_height, fused_width) - shared multi-scale feature map.

"""
        pass


# ============================================================
# __main__: Automated test suite for 3 ablated functions
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
    print("Lifelong-MonoDepth Multi-Domain Depth Model")
    print("Automated Test Suite - 3 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/3: MFF.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/3] MFF.forward - attention-weighted multi-scale feature fusion")
    try:
        mff = MFF([16, 32, 64, 128]).to(device)
        mff.eval()
        x1 = torch.randn(2, 16, 8, 8, device=device, requires_grad=True)
        x2 = torch.randn(2, 32, 4, 4, device=device, requires_grad=True)
        x3 = torch.randn(2, 64, 2, 2, device=device, requires_grad=True)
        x4 = torch.randn(2, 128, 1, 1, device=device, requires_grad=True)
        y = mff(x1, x2, x3, x4, [16, 16])
        check("MFF output not None", y is not None)
        if y is not None:
            check("MFF output shape", tuple(y.shape) == (2, 64, 16, 16), f"expected (2, 64, 16, 16), got {tuple(y.shape)}")
            check("MFF output finite", torch.isfinite(y).all().item())
            x4_changed = x4.detach() + 3.0
            y_changed = mff(x1.detach(), x2.detach(), x3.detach(), x4_changed, [16, 16])
            check("MFF deepest branch affects output", not torch.allclose(y.detach(), y_changed.detach(), atol=1e-6))
            y.mean().backward()
            check("MFF gradients reach shallow and deep inputs",
                  x1.grad is not None and x1.grad.abs().sum().item() > 0 and
                  x4.grad is not None and x4.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "MFF.forward returned None")
    except Exception as exc:
        print(f"  [MFF.forward] ERROR: {exc}")
        skip_checks(5, "MFF.forward raised an exception")
    print()

    # ==============================================================
    # Test 2/3: Uncertainty_depth.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/3] Uncertainty_depth.forward - depth and uncertainty heads")
    try:
        head = Uncertainty_depth([16, 32, 64, 128]).to(device)
        head.eval()
        x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
        result = head(x)
        check("Uncertainty_depth output not None", result is not None)
        if result is not None:
            check("Uncertainty_depth returns pair", isinstance(result, tuple) and len(result) == 2)
            if isinstance(result, tuple) and len(result) == 2:
                depth, uncertainty = result
                check("Uncertainty_depth depth shape", tuple(depth.shape) == (2, 1, 16, 16), f"expected (2, 1, 16, 16), got {tuple(depth.shape)}")
                check("Uncertainty_depth uncertainty shape", tuple(uncertainty.shape) == (2, 1, 16, 16), f"expected (2, 1, 16, 16), got {tuple(uncertainty.shape)}")
                check("Uncertainty_depth outputs finite", torch.isfinite(depth).all().item() and torch.isfinite(uncertainty).all().item())
                (depth.mean() + uncertainty.mean()).backward()
                check("Uncertainty_depth gradients reach both heads",
                      head.depth.conv2.weight.grad is not None and head.depth.conv2.weight.grad.abs().sum().item() > 0 and
                      head.uncertainty.conv2.weight.grad is not None and head.uncertainty.conv2.weight.grad.abs().sum().item() > 0)
            else:
                skip_checks(5, "Uncertainty_depth.forward did not return a two-item tuple")
        else:
            skip_checks(6, "Uncertainty_depth.forward returned None")
    except Exception as exc:
        print(f"  [Uncertainty_depth.forward] ERROR: {exc}")
        skip_checks(7, "Uncertainty_depth.forward raised an exception")
    print()

    # ==============================================================
    # Test 3/3: model_ll.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/3] model_ll.forward - shared encoder with multiple task heads")
    try:
        class FakeEncoder(nn.Module):
            def forward(self, x):
                batch = x.shape[0]
                return (
                    torch.randn(batch, 16, 8, 8, device=x.device),
                    torch.randn(batch, 32, 4, 4, device=x.device),
                    torch.randn(batch, 64, 2, 2, device=x.device),
                    torch.randn(batch, 128, 1, 1, device=x.device),
                )

        model = model_ll.__new__(model_ll)
        nn.Module.__init__(model)
        model.E = FakeEncoder()
        model.MFF = MFF([16, 32, 64, 128])
        model.tasks = {
            0: Uncertainty_depth([16, 32, 64, 128]),
            1: Uncertainty_depth([16, 32, 64, 128]),
            2: Uncertainty_depth([16, 32, 64, 128]),
        }
        model.num_tasks = 3
        for idx, task in model.tasks.items():
            model.add_module('task' + str(idx), task)
        model.to(device)
        model.eval()
        x = torch.randn(2, 3, 32, 32, device=device)
        output = model_ll.forward(model, x)
        check("model_ll output not None", output is not None)
        if output is not None:
            check("model_ll returns output and shared feature", isinstance(output, tuple) and len(output) == 2)
            if isinstance(output, tuple) and len(output) == 2:
                out, shared = output
                check("model_ll task dictionary keys", isinstance(out, dict) and sorted(out.keys()) == [0, 1, 2])
                check("model_ll shared feature shape", tuple(shared.shape) == (2, 64, 16, 16), f"expected (2, 64, 16, 16), got {tuple(shared.shape)}")
                task_shapes_ok = all(
                    isinstance(out[i], tuple) and len(out[i]) == 2 and
                    tuple(out[i][0].shape) == (2, 1, 16, 16) and
                    tuple(out[i][1].shape) == (2, 1, 16, 16)
                    for i in range(3)
                )
                check("model_ll all task heads produce depth uncertainty", task_shapes_ok)
                finite_ok = torch.isfinite(shared).all().item() and all(
                    torch.isfinite(out[i][0]).all().item() and torch.isfinite(out[i][1]).all().item()
                    for i in range(3)
                )
                check("model_ll outputs finite", finite_ok)
            else:
                skip_checks(5, "model_ll.forward did not return a two-item tuple")
        else:
            skip_checks(6, "model_ll.forward returned None")
    except Exception as exc:
        print(f"  [model_ll.forward] ERROR: {exc}")
        skip_checks(7, "model_ll.forward raised an exception")
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
