"""
ground_truth.py for Meta-Transfer Learning core PyTorch model components.

Source-consolidated from:
- pytorch/models/conv2d_mtl.py
- pytorch/models/resnet_mtl.py
- pytorch/models/mtl.py

Only neural network architecture and meta-adaptation components are included.
Data iteration, trainers, CLI wrappers, logging, saved-weight I/O, and
parameter-update orchestration are intentionally excluded.
"""

import math
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _ConvNdMtl(Module):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNdMtl, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(in_channels, out_channels // groups, 1, 1))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.mtl_weight = Parameter(torch.ones(out_channels, in_channels // groups, 1, 1))
        self.weight.requires_grad=False
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
            self.bias.requires_grad=False
            self.mtl_bias = Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
            self.register_parameter('mtl_bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.mtl_weight.data.uniform_(1, 1)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
            self.mtl_bias.data.uniform_(0, 0)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dMtl(_ConvNdMtl):
    """The class for meta-transfer convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dMtl, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias)

    def forward(self, inp):
        new_mtl_weight = self.mtl_weight.expand(self.weight.shape)
        new_weight = self.weight.mul(new_mtl_weight)
        if self.bias is not None:
            new_bias = self.bias + self.mtl_bias
        else:
            new_bias = None
        return F.conv2d(inp, new_weight, new_bias, self.stride,
                        self.padding, self.dilation, self.groups)


def conv3x3(in_planes, out_planes, stride=1):
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
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

def conv3x3mtl(in_planes, out_planes, stride=1):
    return Conv2dMtl(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlockMtl(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockMtl, self).__init__()
        self.conv1 = conv3x3mtl(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3mtl(planes, planes)
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


class BottleneckMtl(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckMtl, self).__init__()
        self.conv1 = Conv2dMtl(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2dMtl(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Conv2dMtl(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

class ResNetMtl(nn.Module):

    def __init__(self, layers=[4, 4, 4], mtl=True):
        super(ResNetMtl, self).__init__()
        if mtl:
            self.Conv2d = Conv2dMtl
            block = BasicBlockMtl
        else:
            self.Conv2d = nn.Conv2d
            block = BasicBlock
        cfg = [160, 320, 640]
        self.inplanes = iChannels = int(cfg[0]/2)
        self.conv1 = self.Conv2d(3, iChannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(iChannels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, cfg[0], layers[0], stride=2)
        self.layer2 = self._make_layer(block, cfg[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg[2], layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(10, stride=1)

        for m in self.modules():
            if isinstance(m, self.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                self.Conv2d(self.inplanes, planes * block.expansion,
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
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, args, z_dim):
        super().__init__()
        self.args = args
        self.z_dim = z_dim
        self.vars = nn.ParameterList()
        self.fc1_w = nn.Parameter(torch.ones([self.args.way, self.z_dim]))
        torch.nn.init.kaiming_normal_(self.fc1_w)
        self.vars.append(self.fc1_w)
        self.fc1_b = nn.Parameter(torch.zeros(self.args.way))
        self.vars.append(self.fc1_b)

    def forward(self, input_x, the_vars=None):
        if the_vars is None:
            the_vars = self.vars
        fc1_w = the_vars[0]
        fc1_b = the_vars[1]
        net = F.linear(input_x, fc1_w, fc1_b)
        return net

    def parameters(self):
        return self.vars

class MtlLearner(nn.Module):
    """The class for outer loop."""
    def __init__(self, args, mode='meta', num_cls=64):
        super().__init__()
        self.args = args
        self.mode = mode
        self.update_lr = args.base_lr
        self.update_step = args.update_step
        z_dim = 640
        self.base_learner = BaseLearner(args, z_dim)

        if self.mode == 'meta':
            self.encoder = ResNetMtl()
        else:
            self.encoder = ResNetMtl(mtl=False)
            self.pre_fc = nn.Sequential(nn.Linear(640, 1000), nn.ReLU(), nn.Linear(1000, num_cls))

    def forward(self, inp):
        """The function to forward the model.
        Args:
          inp: input images.
        Returns:
          the outputs of MTL model.
        """
        if self.mode=='pre':
            return self.pretrain_forward(inp)
        elif self.mode=='meta':
            data_shot, label_shot, data_query = inp
            return self.meta_forward(data_shot, label_shot, data_query)
        elif self.mode=='preval':
            data_shot, label_shot, data_query = inp
            return self.preval_forward(data_shot, label_shot, data_query)
        else:
            raise ValueError('Please set the correct mode.')

    def pretrain_forward(self, inp):
        """The function to forward pretrain phase.
        Args:
          inp: input images.
        Returns:
          the outputs of pretrain model.
        """
        return self.pre_fc(self.encoder(inp))

    def meta_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-train phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, self.update_step):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q

    def preval_forward(self, data_shot, label_shot, data_query):
        """The function to forward meta-validation during pretrain phase.
        Args:
          data_shot: train images for the task
          label_shot: train labels for the task.
          data_query: test images for the task.
        Returns:
          logits_q: the predictions for the test samples.
        """
        embedding_query = self.encoder(data_query)
        embedding_shot = self.encoder(data_shot)
        logits = self.base_learner(embedding_shot)
        loss = F.cross_entropy(logits, label_shot)
        grad = torch.autograd.grad(loss, self.base_learner.parameters())
        fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, self.base_learner.parameters())))
        logits_q = self.base_learner(embedding_query, fast_weights)

        for _ in range(1, 100):
            logits = self.base_learner(embedding_shot, fast_weights)
            loss = F.cross_entropy(logits, label_shot)
            grad = torch.autograd.grad(loss, fast_weights)
            fast_weights = list(map(lambda p: p[1] - 0.01 * p[0], zip(grad, fast_weights)))
            logits_q = self.base_learner(embedding_query, fast_weights)
        return logits_q


if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(42)

    passed = 0
    failed = 0

    def tensor_isfinite(tensor):
        if hasattr(torch, "isfinite"):
            return torch.isfinite(tensor).all().item()
        data = tensor.detach() if hasattr(tensor, "detach") else tensor.data
        return bool(np.isfinite(data.cpu().numpy()).all())

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

    print("Running Meta-Transfer Learning core model benchmark checks...")
    device = torch.device("cpu")

    try:
        layer = Conv2dMtl(3, 5, kernel_size=3, padding=1, bias=True).to(device)
        x = torch.randn(2, 3, 8, 8, device=device)
        y = layer(x)
        check("conv2dmtl output not None", y is not None)
        if y is not None:
            check("conv2dmtl output shape", tuple(y.shape) == (2, 5, 8, 8), f"got {tuple(y.shape)}")
            check("conv2dmtl output finite", tensor_isfinite(y))
            check("conv2dmtl base weight frozen", layer.weight.requires_grad is False)
            check("conv2dmtl ss params trainable", layer.mtl_weight.requires_grad and layer.mtl_bias.requires_grad)
            y.sum().backward()
            check("conv2dmtl ss gradient", layer.mtl_weight.grad is not None and layer.mtl_bias.grad is not None)
            check("conv2dmtl frozen gradient absent", layer.weight.grad is None and layer.bias.grad is None)
        else:
            skip_checks(6, "Conv2dMtl returned None")
    except Exception as exc:
        skip_checks(7, f"Conv2dMtl checks raised {type(exc).__name__}: {exc}")

    try:
        encoder_mtl = ResNetMtl(layers=[1, 1, 1], mtl=True).to(device)
        encoder_mtl.train()
        x = torch.randn(2, 3, 80, 80, device=device)
        emb = encoder_mtl(x)
        check("resnet mtl embedding not None", emb is not None)
        if emb is not None:
            check("resnet mtl embedding shape", tuple(emb.shape) == (2, 640), f"got {tuple(emb.shape)}")
            check("resnet mtl embedding finite", tensor_isfinite(emb))
            first_mtl = next(m for m in encoder_mtl.modules() if isinstance(m, Conv2dMtl))
            check("resnet uses mtl conv", isinstance(first_mtl, Conv2dMtl))
            check("resnet mtl conv frozen base", first_mtl.weight.requires_grad is False)
            emb.sum().backward()
            check("resnet mtl scale gradient", first_mtl.mtl_weight.grad is not None)
        else:
            skip_checks(5, "ResNetMtl returned None")
    except Exception as exc:
        skip_checks(6, f"ResNetMtl MTL checks raised {type(exc).__name__}: {exc}")

    try:
        encoder_plain = ResNetMtl(layers=[1, 1, 1], mtl=False).to(device)
        encoder_plain.eval()
        x = torch.randn(2, 3, 80, 80, device=device)
        with torch.no_grad():
            emb = encoder_plain(x)
        conv_types = [type(m) for m in encoder_plain.modules() if isinstance(m, (nn.Conv2d, Conv2dMtl))]
        check("plain resnet embedding shape", tuple(emb.shape) == (2, 640), f"got {tuple(emb.shape)}")
        check("plain resnet finite", tensor_isfinite(emb))
        check("plain resnet no mtl conv", Conv2dMtl not in conv_types)
    except Exception as exc:
        skip_checks(3, f"plain ResNet checks raised {type(exc).__name__}: {exc}")

    try:
        args = SimpleNamespace(way=5, base_lr=0.4, update_step=1)
        base = BaseLearner(args, z_dim=640).to(device)
        feats = torch.randn(7, 640, device=device)
        logits = base(feats)
        shifted_vars = [base.fc1_w + 0.1, base.fc1_b + 0.2]
        shifted_logits = base(feats, shifted_vars)
        check("base learner logits shape", tuple(logits.shape) == (7, 5), f"got {tuple(logits.shape)}")
        check("base learner logits finite", tensor_isfinite(logits))
        check("base learner param list", len(list(base.parameters())) == 2)
        check("base learner fast weights affect logits", not torch.allclose(logits, shifted_logits))
    except Exception as exc:
        skip_checks(4, f"BaseLearner checks raised {type(exc).__name__}: {exc}")

    try:
        args = SimpleNamespace(way=5, base_lr=0.01, update_step=1)
        pre_model = MtlLearner(args, mode='pre', num_cls=11).to(device)
        pre_model.eval()
        x = torch.randn(2, 3, 80, 80, device=device)
        with torch.no_grad():
            logits = pre_model(x)
        check("pretrain forward shape", tuple(logits.shape) == (2, 11), f"got {tuple(logits.shape)}")
        check("pretrain forward finite", tensor_isfinite(logits))
        check("pretrain encoder plain conv", not any(isinstance(m, Conv2dMtl) for m in pre_model.encoder.modules()))
    except Exception as exc:
        skip_checks(3, f"pretrain MtlLearner checks raised {type(exc).__name__}: {exc}")

    try:
        args = SimpleNamespace(way=5, base_lr=0.01, update_step=2)
        meta_model = MtlLearner(args, mode='meta').to(device)
        meta_model.train()
        data_shot = torch.randn(5, 3, 80, 80, device=device)
        label_shot = torch.arange(5, dtype=torch.long, device=device)
        data_query = torch.randn(6, 3, 80, 80, device=device)
        logits_q = meta_model((data_shot, label_shot, data_query))
        check("meta forward shape", tuple(logits_q.shape) == (6, 5), f"got {tuple(logits_q.shape)}")
        check("meta forward finite", tensor_isfinite(logits_q))
        check("meta encoder uses mtl conv", any(isinstance(m, Conv2dMtl) for m in meta_model.encoder.modules()))
        objective = logits_q.sum()
        objective.backward()
        first_mtl = next(m for m in meta_model.encoder.modules() if isinstance(m, Conv2dMtl))
        check("meta ss gradient path", first_mtl.mtl_weight.grad is not None)
        check("meta base learner gradients", meta_model.base_learner.fc1_w.grad is not None or meta_model.base_learner.fc1_b.grad is not None)
    except Exception as exc:
        skip_checks(5, f"meta MtlLearner checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
