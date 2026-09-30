from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.utils import load_state_dict_from_url


# ============================================================
# ground_truth.py - AdvCL Core Model Components (Self-contained)
# Source: Adversarial/AdvCL-main
#
# Contains ONLY the model/loss/view-generation components needed by
# the AdvCL adversarial contrastive pretraining mechanism.
# No training loop, dataset, evaluation, checkpointing, or logging code.
# ============================================================


# --- [Original file: fr_util.py] ---
def distance(i, j, imageSize, r):
    dis = np.sqrt((i - imageSize / 2) ** 2 + (j - imageSize / 2) ** 2)
    if dis < r:
        return 1.0
    else:
        return 0


def mask_radial(img, r):
    rows, cols = img.shape
    mask = torch.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            mask[i, j] = distance(i, j, imageSize=rows, r=r)
    return mask.cuda()


def generate_high(Images, r):
    # Image: bsxcxhxw, input batched images
    # r: int, radius
    mask = mask_radial(torch.zeros([Images.shape[2], Images.shape[3]]), r)
    bs, c, h, w = Images.shape
    x = Images.reshape([bs * c, h, w])
    fd = torch.fft.fftshift(torch.fft.fftn(x, dim=(-2, -1)))
    mask = mask.unsqueeze(0).repeat([bs * c, 1, 1])
    fd = fd * (1.-mask)
    fd = torch.fft.ifftn(torch.fft.ifftshift(fd), dim=(-2, -1))
    fd = torch.real(fd)
    fd = fd.reshape([bs, c, h, w])
    return fd


# --- [Original file: losses.py] ---
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


# --- [Original file: models/resnet_cifar_multibn_ensembleFC.py] ---
__all__ = ['ResNet', 'resnet18']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_names=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = batch_norm_multiple(norm_layer, planes, bn_names=bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = batch_norm_multiple(norm_layer, planes, bn_names=bn_names)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = x[0]
        bn_name = x[1]

        # debug
        # print("bn_name: {}".format(bn_name))

        out = self.conv1(out)
        out = self.bn1([out, bn_name])

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2([out, bn_name])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity[0]
        out = self.relu(out)

        return [out, bn_name]


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, bn_names=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = batch_norm_multiple(norm_layer, width, bn_names=bn_names)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = batch_norm_multiple(norm_layer, width, bn_names=bn_names)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = batch_norm_multiple(norm_layer, planes * self.expansion, bn_names=bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = x[0]
        bn_name = x[1]

        out = self.conv1(out)
        out = self.bn1([out, bn_name])

        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2([out, bn_name])

        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3([out, bn_name])

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity[0]
        out = self.relu(out)

        return [out, bn_name]


class Downsample_multiple(nn.Module):
    def __init__(self, inplanes, planes, expansion, stride, norm_layer, bn_names=None):
        super(Downsample_multiple, self).__init__()
        self.conv = conv1x1(inplanes, planes * expansion, stride)
        self.bn = batch_norm_multiple(norm_layer, planes * expansion, bn_names=bn_names)

    def forward(self, x):
        out = x[0]
        bn_name = x[1]
        # debug
        # print("adv attack: {}".format(flag_adv))
        # print("out is {}".format(out))

        out = self.conv(out)
        out = self.bn([out, bn_name])

        return [out, bn_name]


class batch_norm_multiple(nn.Module):
    def __init__(self, norm, inplanes, bn_names=None):
        super(batch_norm_multiple, self).__init__()

        # if no bn name input, by default use single bn
        self.bn_names = bn_names
        if self.bn_names is None:
            self.bn_list = norm(inplanes)
            return

        len_bn_names = len(bn_names)
        self.bn_list = nn.ModuleList([norm(inplanes) for _ in range(len_bn_names)])
        self.bn_names_dict = {bn_name: i for i, bn_name in enumerate(bn_names)}
        return

    def forward(self, x):
        out = x[0]
        name_bn = x[1]

        if name_bn is None:
            out = self.bn_list(out)
        else:
            bn_index = self.bn_names_dict[name_bn]
            out = self.bn_list[bn_index](out)

        return out


class proj_head(nn.Module):
    def __init__(self, ch, bn_names=None, twoLayerProj=False):
        super(proj_head, self).__init__()
        self.in_features = ch
        self.twoLayerProj = twoLayerProj

        self.fc1 = nn.Linear(ch, ch)
        self.bn1 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)
        self.fc2 = nn.Linear(ch, ch, bias=False)
        self.bn2 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

        if not twoLayerProj:
            self.fc3 = nn.Linear(ch, ch, bias=False)
            self.bn3 = batch_norm_multiple(nn.BatchNorm1d, ch, bn_names)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, bn_name):
        # debug
        # print("adv attack: {}".format(flag_adv))

        x = self.fc1(x)
        x = self.bn1([x, bn_name])

        x = self.relu(x)

        x = self.fc2(x)
        x = self.bn2([x, bn_name])

        if not self.twoLayerProj:
            x = self.relu(x)

            x = self.fc3(x)
            x = self.bn3([x, bn_name])

        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, bn_names, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        """
        :param bn_names: list, the name of bn that would be employed
        """

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.bn_names = bn_names

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, 3, 1, 1, bias=False)
        self.bn1 = batch_norm_multiple(norm_layer, self.inplanes, bn_names=self.bn_names)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.Identity()
        self.layer1 = self._make_layer(block, 64, layers[0], bn_names=self.bn_names)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], bn_names=self.bn_names,
                                       stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_002 = nn.Linear(512 * block.expansion, 2)
        self.fc_010 = nn.Linear(512 * block.expansion, 10)
        self.fc_050 = nn.Linear(512 * block.expansion, 50)
        self.fc_100 = nn.Linear(512 * block.expansion, 100)
        self.fc_500 = nn.Linear(512 * block.expansion, 500)

        dim_in = 512*block.expansion

        self.head_ce = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, num_classes)
        )

        self.head_proj = nn.Sequential(
            nn.Linear(dim_in, dim_in),
            nn.BatchNorm1d(dim_in),
            nn.ReLU(inplace=True),
            nn.Linear(dim_in, 128)
        )

        self.head_pred = nn.Sequential(
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, bn_names=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = Downsample_multiple(self.inplanes, planes, block.expansion, stride, norm_layer, bn_names=bn_names)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, bn_names=bn_names))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, bn_names=bn_names))

        return nn.Sequential(*layers)

    def _forward_impl(self, x, bn_name=None, contrast=False, return_feat=False, CF=False, return_logits=False, nonlinear=False):

        # debug
        # print("bn name: {}".format(bn_name))

        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1([x, bn_name])

        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1([x, bn_name])
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x[0])
        x = torch.flatten(x, 1)

        # if isinstance(self.fc, proj_head):
        #     x = self.fc(x, bn_name)
        # else:
        #     x = self.fc(x)
        out = x
        if return_feat:
            return out
        feat = out
        if contrast:
            # out = self.linear_contrast(out)
            proj = self.head_proj(out)
            pred = self.head_pred(proj)
            proj = F.normalize(proj, dim=1)
            pred = F.normalize(pred, dim=1)
            if CF:
                if return_logits:
                    if nonlinear:
                        return proj, pred, self.head_ce(feat)
                    else:
                        return proj, pred, \
                               (self.fc_002(feat),
                                self.fc_010(feat),
                                self.fc_050(feat),
                                self.fc_100(feat),
                                self.fc_500(feat),
                                )
                else:
                    return proj, pred, feat
            else:
                return proj, pred
        else:
            out = self.fc(out)
        return out

    def forward(self, x, bn_name=None, contrast=False, return_feat=False, CF=False, return_logits=False, nonlinear=False):
        return self._forward_impl(x, bn_name, contrast=contrast, return_feat=return_feat, CF=CF, return_logits=return_logits, nonlinear=nonlinear)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


# --- [Original file: pretraining_advCL.py] ---
class _AdvCLArgs:
    radius = 8
    ce_weight = 0.2


args = _AdvCLArgs()


class AttackPGD(nn.Module):
    def __init__(self, model, config):
        super(AttackPGD, self).__init__()
        self.model = model
        self.rand = config['random_start']
        self.step_size = config['step_size']
        self.epsilon = config['epsilon']
        self.num_steps = config['num_steps']
        assert config['loss_func'] == 'xent', 'Plz use xent for loss function.'

    def forward(self, images_t1, images_t2, images_org, targets, criterion):
        x1 = images_t1.clone().detach()
        x2 = images_t2.clone().detach()
        x_cl = images_org.clone().detach()
        x_ce = images_org.clone().detach()

        images_org_high = generate_high(x_cl.clone(), r=args.radius)
        x_HFC = images_org_high.clone().detach()

        if self.rand:
            x_cl = x_cl + torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)
            x_ce = x_ce + torch.zeros_like(x1).uniform_(-self.epsilon, self.epsilon)

        for i in range(self.num_steps):
            x_cl.requires_grad_()
            x_ce.requires_grad_()
            with torch.enable_grad():
                f_proj, f_pred = self.model(x_cl, bn_name='pgd', contrast=True)
                fce_proj, fce_pred, logits_ce = self.model(x_ce, bn_name='pgd_ce', contrast=True, CF=True, return_logits=True, nonlinear=False)
                f1_proj, f1_pred = self.model(x1, bn_name='normal', contrast=True)
                f2_proj, f2_pred = self.model(x2, bn_name='normal', contrast=True)
                f_high_proj, f_high_pred = self.model(x_HFC, bn_name='normal', contrast=True)
                features = torch.cat([f_proj.unsqueeze(1), f1_proj.unsqueeze(1), f2_proj.unsqueeze(1), f_high_proj.unsqueeze(1)], dim=1)
                loss_contrast = criterion(features)
                loss_ce = 0
                for label_idx in range(5):
                    tgt = targets[label_idx].long()
                    lgt = logits_ce[label_idx]
                    loss_ce += F.cross_entropy(lgt, tgt, size_average=False, ignore_index=-1) / 5.
                loss = loss_contrast + loss_ce * args.ce_weight
            grad_x_cl, grad_x_ce = torch.autograd.grad(loss, [x_cl, x_ce])
            x_cl = x_cl.detach() + self.step_size * torch.sign(grad_x_cl.detach())
            x_cl = torch.min(torch.max(x_cl, images_org - self.epsilon), images_org + self.epsilon)
            x_cl = torch.clamp(x_cl, 0, 1)
            x_ce = x_ce.detach() + self.step_size * torch.sign(grad_x_ce.detach())
            x_ce = torch.min(torch.max(x_ce, images_org - self.epsilon), images_org + self.epsilon)
            x_ce = torch.clamp(x_ce, 0, 1)
        return x1, x2, x_cl, x_ce, x_HFC


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

    print("=" * 70)
    print("AdvCL: adversarial contrastive pretraining benchmark")
    print("Automated Test Suite - 5 ablated targets")
    print("=" * 70)
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/5: batch_norm_multiple.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/5] batch_norm_multiple.forward - branch-specific BN routing")
    try:
        bn = batch_norm_multiple(nn.BatchNorm1d, 4, bn_names=["normal", "pgd"]).to(device)
        bn.eval()
        with torch.no_grad():
            bn.bn_list[0].weight.fill_(1.0)
            bn.bn_list[0].bias.fill_(0.0)
            bn.bn_list[1].weight.fill_(2.0)
            bn.bn_list[1].bias.fill_(0.5)
        x = torch.randn(3, 4, device=device)
        y_normal = bn([x, "normal"])
        y_pgd = bn([x, "pgd"])
        check("batch_norm output not None", y_normal is not None and y_pgd is not None)
        if y_normal is not None and y_pgd is not None:
            check("batch_norm output shape", tuple(y_normal.shape) == (3, 4),
                  f"expected (3, 4), got {tuple(y_normal.shape)}")
            check("batch_norm output finite", torch.isfinite(y_normal).all().item() and torch.isfinite(y_pgd).all().item())
            check("batch_norm routes selected branch", not torch.allclose(y_normal, y_pgd))
        else:
            skip_checks(3, "batch_norm_multiple.forward returned None")
    except Exception as e:
        skip_checks(4, f"batch_norm_multiple.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/5: ResNet._forward_impl
    # ==============================================================
    print("-" * 60)
    print("[Test 2/5] ResNet._forward_impl - contrast and pseudo-label heads")
    try:
        model = resnet18(bn_names=["normal", "pgd", "pgd_ce"]).to(device)
        model.train()
        x = torch.randn(2, 3, 32, 32, device=device)
        result = model(x, bn_name="pgd_ce", contrast=True, CF=True, return_logits=True, nonlinear=False)
        check("ResNet output not None", result is not None)
        if result is not None:
            proj, pred, logits = result
            check("ResNet projection shape", tuple(proj.shape) == (2, 128),
                  f"expected (2, 128), got {tuple(proj.shape)}")
            check("ResNet prediction shape", tuple(pred.shape) == (2, 128),
                  f"expected (2, 128), got {tuple(pred.shape)}")
            check("ResNet outputs finite", torch.isfinite(proj).all().item() and torch.isfinite(pred).all().item())
            check("ResNet projections normalized", torch.allclose(proj.norm(dim=1), torch.ones(2, device=device), atol=1e-4))
            expected_logit_shapes = [(2, 2), (2, 10), (2, 50), (2, 100), (2, 500)]
            actual_logit_shapes = [tuple(item.shape) for item in logits] if isinstance(logits, tuple) else []
            check("ResNet pseudo-label logits", actual_logit_shapes == expected_logit_shapes,
                  f"expected {expected_logit_shapes}, got {actual_logit_shapes}")
        else:
            skip_checks(5, "ResNet._forward_impl returned None")
    except Exception as e:
        skip_checks(6, f"ResNet._forward_impl raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/5: SupConLoss.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/5] SupConLoss.forward - contrastive mask and gradients")
    try:
        criterion = SupConLoss(temperature=0.5)
        features = F.normalize(torch.randn(4, 3, 8, device=device), dim=2).requires_grad_()
        labels = torch.tensor([0, 0, 1, 1], device=device)
        loss = criterion(features, labels=labels)
        check("SupConLoss output not None", loss is not None)
        if loss is not None:
            check("SupConLoss scalar shape", tuple(loss.shape) == (),
                  f"expected scalar, got {tuple(loss.shape)}")
            check("SupConLoss finite", torch.isfinite(loss).item())
            loss.backward()
            check("SupConLoss gradient exists", features.grad is not None and torch.isfinite(features.grad).all().item())
            raised = False
            try:
                criterion(features.detach(), labels=labels, mask=torch.eye(4, device=device))
            except ValueError:
                raised = True
            check("SupConLoss rejects labels with mask", raised)
        else:
            skip_checks(4, "SupConLoss.forward returned None")
    except Exception as e:
        skip_checks(5, f"SupConLoss.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/5: generate_high
    # ==============================================================
    print("-" * 60)
    print("[Test 4/5] generate_high - high-frequency view selection")
    if not torch.cuda.is_available():
        skip_checks(4, "source implementation requires CUDA through mask_radial")
    else:
        try:
            constant = torch.ones(2, 3, 32, 32, device=device) * 0.5
            checker = torch.zeros(2, 3, 32, 32, device=device)
            checker[:, :, ::2, ::2] = 1.0
            checker[:, :, 1::2, 1::2] = 1.0
            high_constant = generate_high(constant, r=8)
            high_checker = generate_high(checker, r=8)
            check("generate_high output not None", high_constant is not None and high_checker is not None)
            if high_constant is not None and high_checker is not None:
                check("generate_high output shape", tuple(high_constant.shape) == (2, 3, 32, 32),
                      f"expected (2, 3, 32, 32), got {tuple(high_constant.shape)}")
                check("generate_high output finite", torch.isfinite(high_constant).all().item() and torch.isfinite(high_checker).all().item())
                check("generate_high suppresses constant image", high_constant.abs().mean().item() < high_checker.abs().mean().item())
            else:
                skip_checks(3, "generate_high returned None")
        except Exception as e:
            skip_checks(4, f"generate_high raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 5/5: AttackPGD.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 5/5] AttackPGD.forward - adversarial contrastive views")
    if not torch.cuda.is_available():
        skip_checks(5, "source implementation requires CUDA through high-frequency generation")
    else:
        try:
            model = resnet18(bn_names=["normal", "pgd", "pgd_ce"]).to(device)
            model.train()
            config = {
                "epsilon": 8.0 / 255.0,
                "num_steps": 1,
                "step_size": 2.0 / 255.0,
                "random_start": True,
                "loss_func": "xent",
            }
            attacker = AttackPGD(model, config).to(device)
            criterion = SupConLoss(temperature=0.5)
            images_t1 = torch.rand(2, 3, 32, 32, device=device)
            images_t2 = torch.rand(2, 3, 32, 32, device=device)
            images_org = torch.rand(2, 3, 32, 32, device=device)
            targets = [
                torch.randint(0, 2, (2,), device=device),
                torch.randint(0, 10, (2,), device=device),
                torch.randint(0, 50, (2,), device=device),
                torch.randint(0, 100, (2,), device=device),
                torch.randint(0, 500, (2,), device=device),
            ]
            outputs = attacker(images_t1, images_t2, images_org, targets, criterion)
            check("AttackPGD output not None", outputs is not None)
            if outputs is not None:
                x1, x2, x_cl, x_ce, x_HFC = outputs
                same_shape = all(tuple(item.shape) == (2, 3, 32, 32) for item in outputs)
                check("AttackPGD output shapes", same_shape)
                check("AttackPGD outputs finite", all(torch.isfinite(item).all().item() for item in outputs))
                cl_within = (x_cl - images_org).abs().max().item() <= config["epsilon"] + 1e-6
                ce_within = (x_ce - images_org).abs().max().item() <= config["epsilon"] + 1e-6
                check("AttackPGD epsilon bounded", cl_within and ce_within)
                clamped = x_cl.min().item() >= 0.0 and x_cl.max().item() <= 1.0 and x_ce.min().item() >= 0.0 and x_ce.max().item() <= 1.0
                check("AttackPGD image range clamped", clamped)
            else:
                skip_checks(4, "AttackPGD.forward returned None")
        except Exception as e:
            skip_checks(5, f"AttackPGD.forward raised {type(e).__name__}: {e}")
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
