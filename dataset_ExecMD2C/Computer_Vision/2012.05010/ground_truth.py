# ============================================================
# ground_truth.py - DGTL-for-VT-ReID Core Model Components (Source-faithful)
# Source: DGTL-for-VT-ReID-main
#
# Contains ONLY model architecture definitions and direct dependencies.
# Model code above __main__ is copied from the source repository with only
# one-file consolidation import adjustments.
# No training, dataset, evaluation, CLI, or checkpoint orchestration code.
# ============================================================

import math
import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.autograd import Variable
from torchvision import models



# --- [Original file: resnet.py] ---
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
  'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
  'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
  'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
  'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
  'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
  """3x3 convolution with padding"""
  # original padding is 1; original dilation is 1
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(BasicBlock, self).__init__()
    self.conv1 = conv3x3(inplanes, planes, stride, dilation)
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

  def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
    super(Bottleneck, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    # original padding is 1; original dilation is 1
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=dilation, bias=False, dilation=dilation)
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

  def __init__(self, block, layers, last_conv_stride=2, last_conv_dilation=1):

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
    self.layer4 = self._make_layer(block, 512, layers[3], stride=last_conv_stride, dilation=last_conv_dilation)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
        nn.Conv2d(self.inplanes, planes * block.expansion,
                  kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
      )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample, dilation))
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

    return x


def remove_fc(state_dict):
  """Remove the fc layer parameters from state_dict."""
  # for key, value in state_dict.items():
  for key, value in list(state_dict.items()):
    if key.startswith('fc.'):
      del state_dict[key]
  return state_dict


def resnet18(pretrained=False, **kwargs):
  """Constructs a ResNet-18 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet18'])))
  return model


def resnet34(pretrained=False, **kwargs):
  """Constructs a ResNet-34 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet34'])))
  return model


def resnet50(pretrained=False, **kwargs):
  """Constructs a ResNet-50 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    # model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
    model.load_state_dict(remove_fc(model_zoo.load_url(model_urls['resnet50'])))
  return model


def resnet101(pretrained=False, **kwargs):
  """Constructs a ResNet-101 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet101'])))
  return model


def resnet152(pretrained=False, **kwargs):
  """Constructs a ResNet-152 model.
  Args:
      pretrained (bool): If True, returns a model pre-trained on ImageNet
  """
  model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
  if pretrained:
    model.load_state_dict(
      remove_fc(model_zoo.load_url(model_urls['resnet152'])))
  return model


# --- [Original file: attention.py] ---
"""
    PART of the code is from the following link
    https://github.com/Diego999/pyGAT/blob/master/layers.py
"""


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class AVG(nn.Module):
    def __init__(self, channel, fuse = 'sum', reduction=16):
        super(AVG, self).__init__()
        self.fuse = fuse
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(channel) if fuse == 'sum' else nn.BatchNorm1d(channel*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self, x, feat):
        b, c, _, _ = x.size()
        y_pool = self.avg_pool(x).view(b, c)

        if self.fuse == 'sum': 
            feats = y_pool + feat
        elif self.fuse == 'cat':
            feats = torch.cat([y_pool,feat],dim=1)
        feats_bn = self.bottleneck(feats)
        return feats, feats_bn

class MAX(nn.Module):
    def __init__(self, channel, fuse = 'sum', reduction=16):
        super(MAX, self).__init__()
        self.fuse = fuse
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(channel) if fuse == 'sum' else nn.BatchNorm1d(channel*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self, x, feat):
        b, c, _, _ = x.size()
        y_pool = self.max_pool(x).view(b, c)

        if self.fuse == 'sum': 
            feats = y_pool + feat
        elif self.fuse == 'cat':
            feats = torch.cat([y_pool,feat],dim=1)
        feats_bn = self.bottleneck(feats)
        return feats, feats_bn

class GEM(nn.Module):
    def __init__(self, channel, fuse = 'sum', reduction=16):
        super(GEM, self).__init__()
        self.fuse = fuse
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.bottleneck = nn.BatchNorm1d(channel) if fuse == 'sum' else nn.BatchNorm1d(channel*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

    def forward(self, x, feat):
        b, c, _, _ = x.size()
        x_pool = x.view(b, c, -1)
        p = 3.0    
        y_pool = (torch.mean(x_pool**p, dim=-1) + 1e-12)**(1/p)

        if self.fuse == 'sum': 
            feats = y_pool + feat
        elif self.fuse == 'cat':
            feats = torch.cat([y_pool,feat],dim=1)
        feats_bn = self.bottleneck(feats)
        return feats, feats_bn

class IWPA(nn.Module):
    """
    Part attention layer, "Dynamic Dual-Attentive Aggregation Learning for Visible-Infrared Person Re-Identification"
    """
    def __init__(self, in_channels, part = 3, fuse = 'sum', inter_channels=None, out_channels=None):
        super(IWPA, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.out_channels = out_channels
        self.l2norm = Normalize(2)
        self.fuse = fuse

        if self.inter_channels is None:
            self.inter_channels = in_channels

        if self.out_channels is None:
            self.out_channels = in_channels

        conv_nd = nn.Conv2d

        self.fc1 = nn.Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.fc2 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.fc3 = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.out_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.out_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.bottleneck = nn.BatchNorm1d(in_channels) if fuse == 'sum' else nn.BatchNorm1d(in_channels*2)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        nn.init.normal_(self.bottleneck.weight.data, 1.0, 0.01)
        nn.init.zeros_(self.bottleneck.bias.data)

        # weighting vector of the part features
        self.gate = nn.Parameter(torch.FloatTensor(part))
        nn.init.constant_(self.gate, 1/part)
    def forward(self, x, feat, t=None, part=0):
        bt, c, h, w = x.shape
        b = bt // t

        # get part features
        part_feat = F.adaptive_avg_pool2d(x, (part, 1))
        part_feat = part_feat.view(b, t, c, part)
        part_feat = part_feat.permute(0, 2, 1, 3) # B, C, T, Part

        part_feat1 = self.fc1(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        part_feat1 = part_feat1.permute(0, 2, 1)  # B, T*Part, C//r

        part_feat2 = self.fc2(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part

        part_feat3 = self.fc3(part_feat).view(b, self.inter_channels, -1)  # B, C//r, T*Part
        part_feat3 = part_feat3.permute(0, 2, 1)   # B, T*Part, C//r

        # get cross-part attention
        cpa_att = torch.matmul(part_feat1, part_feat2) # B, T*Part, T*Part
        cpa_att = F.softmax(cpa_att, dim=-1)

        # collect contextual information
        refined_part_feat = torch.matmul(cpa_att, part_feat3) # B, T*Part, C//r
        refined_part_feat = refined_part_feat.permute(0, 2, 1).contiguous() # B, C//r, T*Part
        refined_part_feat = refined_part_feat.view(b, self.inter_channels, part) # B, C//r, T, Part

        gate = F.softmax(self.gate, dim=-1)
        weight_part_feat = torch.matmul(refined_part_feat, gate)
        #x = F.adaptive_avg_pool2d(x, (1, 1))
        # weight_part_feat = weight_part_feat + x.view(x.size(0), x.size(1))

        if self.fuse == 'sum': 
            feats = weight_part_feat + feat
        elif self.fuse == 'cat':
            feats = torch.cat([weight_part_feat,feat],dim=1)
        feats_bn = self.bottleneck(feats)

        return feats, feats_bn


# --- [Original file: model_main.py] ---
class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out



# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x

class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        x = self.visible.layer1(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        x = self.thermal.layer1(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        #x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, drop=0.2, part = 3, arch='resnet50', cpool = 'no', bpool = 'avg', fuse = 'sum'):
        super(embed_net, self).__init__()

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        pool_dim = 2048
        pool_dim_att = 2048 if fuse == "sum" else 4096
        self.dropout = drop
        self.part = part
        self.cpool = cpool
        self.bpool = bpool
        self.fuse = fuse

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        if self.cpool == 'wpa':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = IWPA(pool_dim, part,fuse)
        if self.cpool == 'avg':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = AVG(pool_dim,fuse)
        if self.cpool == 'max':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = MAX(pool_dim,fuse)
        if self.cpool == 'gem':
            self.classifier_att = nn.Linear(pool_dim_att, class_num, bias=False)    
            self.classifier_att.apply(weights_init_classifier)
            self.cpool_layer = GEM(pool_dim,fuse)



    def forward(self, x1, x2, modal=0):
        # domain specific block
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        # shared four blocks
        x = self.base_resnet(x)

        if self.bpool == 'gem':
            b, c, _, _ = x.shape
            x_pool = x.view(b, c, -1)
            p = 3.0    
            x_pool = (torch.mean(x_pool**p, dim=-1) + 1e-12)**(1/p)
        elif self.bpool == 'avg':
            x_pool = F.adaptive_avg_pool2d(x,1)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        elif self.bpool == 'max':
            x_pool = F.adaptive_max_pool2d(x,1)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        else:
            print("wrong backbone pooling!!!")
            exit()

        feat  = self.bottleneck(x_pool)

        if self.cpool != 'no':
            # intra-modality weighted part attention
            if self.cpool == 'wpa':
                feat_att, feat_att_bn = self.cpool_layer(x, feat, 1, self.part)
            if self.cpool in ['avg', 'max', 'gem']:
                feat_att, feat_att_bn = self.cpool_layer(x, feat)

            if self.training:            
                return x_pool, self.classifier(feat), feat_att_bn, self.classifier_att(feat_att_bn) 
            else:
                return self.l2norm(feat), self.l2norm(feat_att_bn)
        else:
            if self.training:            
                return x_pool, self.classifier(feat)
            else:
                return self.l2norm(feat)


if __name__ == "__main__":
    torch.manual_seed(13)
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

    def fake_resnet50_state_dict():
        return resnet50(pretrained=False, last_conv_stride=1, last_conv_dilation=1).state_dict()

    print("DGTL-for-VT-ReID model reproduction benchmark checks")

    print("\nTest 1: AVG.forward")
    try:
        layer = AVG(channel=8, fuse="sum")
        layer.eval()
        x = torch.randn(2, 8, 4, 3)
        feat = torch.randn(2, 8)
        out = layer(x, feat)
        check("avg-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "AVG.forward returned None")
        else:
            feats, feats_bn = out
            check("avg-shape-finite", feats.shape == (2, 8) and feats_bn.shape == (2, 8) and torch.isfinite(feats).all() and torch.isfinite(feats_bn).all(), f"{feats.shape}, {feats_bn.shape}")
            expected = F.adaptive_avg_pool2d(x, 1).view(2, 8) + feat
            check("avg-fuses-global-feature", torch.allclose(feats, expected, atol=1e-5))
    except Exception as exc:
        skip_checks(3, f"AVG.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 2: MAX.forward")
    try:
        layer = MAX(channel=8, fuse="cat")
        layer.eval()
        x = torch.randn(2, 8, 4, 3)
        feat = torch.randn(2, 8)
        out = layer(x, feat)
        check("max-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "MAX.forward returned None")
        else:
            feats, feats_bn = out
            check("max-shape-finite", feats.shape == (2, 16) and feats_bn.shape == (2, 16) and torch.isfinite(feats).all() and torch.isfinite(feats_bn).all(), f"{feats.shape}, {feats_bn.shape}")
            expected_prefix = F.adaptive_max_pool2d(x, 1).view(2, 8)
            check("max-cat-order", torch.allclose(feats[:, :8], expected_prefix, atol=1e-5) and torch.allclose(feats[:, 8:], feat, atol=1e-5))
    except Exception as exc:
        skip_checks(3, f"MAX.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 3: GEM.forward")
    try:
        layer = GEM(channel=8, fuse="sum")
        layer.eval()
        x = torch.rand(2, 8, 4, 3) + 0.1
        feat = torch.randn(2, 8)
        out = layer(x, feat)
        check("gem-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "GEM.forward returned None")
        else:
            feats, feats_bn = out
            check("gem-shape-finite", feats.shape == (2, 8) and feats_bn.shape == (2, 8) and torch.isfinite(feats).all() and torch.isfinite(feats_bn).all(), f"{feats.shape}, {feats_bn.shape}")
            expected_pool = (torch.mean(x.view(2, 8, -1) ** 3.0, dim=-1) + 1e-12) ** (1 / 3.0)
            check("gem-power-mean-fusion", torch.allclose(feats, expected_pool + feat, atol=1e-5))
    except Exception as exc:
        skip_checks(3, f"GEM.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 4: IWPA.forward")
    try:
        layer = IWPA(in_channels=8, part=3, fuse="sum")
        layer.eval()
        x = torch.randn(2, 8, 6, 4)
        feat = torch.randn(2, 8)
        out = layer(x, feat, t=1, part=3)
        check("iwpa-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "IWPA.forward returned None")
        else:
            feats, feats_bn = out
            check("iwpa-shape-finite", feats.shape == (2, 8) and feats_bn.shape == (2, 8) and torch.isfinite(feats).all() and torch.isfinite(feats_bn).all(), f"{feats.shape}, {feats_bn.shape}")
            gate = F.softmax(layer.gate, dim=-1)
            check("iwpa-gate-simplex", torch.allclose(gate.sum(), torch.tensor(1.0), atol=1e-6) and bool((gate >= 0).all()))
    except Exception as exc:
        skip_checks(3, f"IWPA.forward raised {type(exc).__name__}: {exc}")

    print("\nTest 5: embed_net.forward")
    try:
        old_load_url = model_zoo.load_url
        model_zoo.load_url = lambda *args, **kwargs: fake_resnet50_state_dict()
        try:
            model = embed_net(class_num=5, drop=0.0, part=3, cpool="max", bpool="avg", fuse="sum")
        finally:
            model_zoo.load_url = old_load_url
        model.eval()
        x1 = torch.randn(1, 3, 64, 32)
        x2 = torch.randn(1, 3, 64, 32)
        out = model(x1, x2, modal=0)
        check("embed-output-not-none", out is not None)
        if out is None:
            skip_checks(2, "embed_net.forward returned None")
        else:
            feat, feat_att = out
            check("embed-shape-finite", feat.shape == (2, 2048) and feat_att.shape == (2, 2048) and torch.isfinite(feat).all() and torch.isfinite(feat_att).all(), f"{feat.shape}, {feat_att.shape}")
            check("embed-l2-normalized", torch.allclose(feat.norm(dim=1), torch.ones(2), atol=1e-4) and torch.allclose(feat_att.norm(dim=1), torch.ones(2), atol=1e-4))
    except Exception as exc:
        skip_checks(3, f"embed_net.forward raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"\nResult: {passed}/{total} checks passed")
    if failed != 0:
        raise SystemExit(1)
