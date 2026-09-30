import copy
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from torch.nn import Module
from torch.nn.parameter import Parameter


class CosineClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out


class BiC(nn.Module):
    def __init__(self, lr, scheduling, lr_decay_factor, weight_decay, batch_size, epochs):
        super(BiC, self).__init__()
        self.beta = torch.nn.Parameter(torch.ones(1))
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.lr = lr
        self.scheduling = scheduling
        self.lr_decay_factor = lr_decay_factor
        self.weight_decay = weight_decay
        self.class_specific = False
        self.batch_size = batch_size
        self.epochs = epochs
        self.bic_flag = False

    def reset(self, lr=None, scheduling=None, lr_decay_factor=None, weight_decay=None, n_classes=-1):
        with torch.no_grad():
            if lr is None:
                lr = self.lr
            if scheduling is None:
                scheduling = self.scheduling
            if lr_decay_factor is None:
                lr_decay_factor = self.lr_decay_factor
            if weight_decay is None:
                weight_decay = self.weight_decay
            if self.class_specific:
                assert n_classes != -1
                self.beta = torch.nn.Parameter(torch.ones(n_classes).cuda())
                self.gamma = torch.nn.Parameter(torch.zeros(n_classes).cuda())
            else:
                self.beta = torch.nn.Parameter(torch.ones(1).cuda())
                self.gamma = torch.nn.Parameter(torch.zeros(1).cuda())
            self.optimizer = torch.optim.SGD([self.beta, self.gamma], lr=lr, momentum=0.9, weight_decay=weight_decay)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, scheduling, gamma=lr_decay_factor)

    def extract_preds_and_targets(self, model, loader):
        preds, targets = [], []
        with torch.no_grad():
            for (x, y) in loader:
                preds.append(model(x.cuda())['logit'])
                targets.append(y.cuda())
        return torch.cat((preds)), torch.cat((targets))

    def update(self, logger, task_size, model, loader, loss_criterion=None):
        if task_size == 0:
            logger.info("no new task for BiC!")
            return
        if loss_criterion is None:
            loss_criterion = F.cross_entropy

        self.bic_flag = True
        logger.info("Begin BiC ...")
        model.eval()

        for epoch in range(self.epochs):
            preds_, targets_ = self.extract_preds_and_targets(model, loader)
            order = np.arange(preds_.shape[0])
            np.random.shuffle(order)

            preds, targets = preds_.clone(), targets_.clone()
            preds, targets = preds[order], targets[order]
            _loss = 0.0
            _correct = 0
            _count = 0
            for start in range(0, preds.shape[0], self.batch_size):
                if start + self.batch_size < preds.shape[0]:
                    out = preds[start:start + self.batch_size, :].clone()
                    lbls = targets[start:start + self.batch_size]
                else:
                    out = preds[start:, :].clone()
                    lbls = targets[start:]
                if self.class_specific is False:
                    out1 = out[:, :-task_size].clone()
                    out2 = out[:, -task_size:].clone()
                    outputs = torch.cat((out1, out2 * self.beta + self.gamma), 1)
                else:
                    outputs = out * self.beta + self.gamma
                loss = loss_criterion(outputs, lbls)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                _, pred = outputs.max(1)
                _correct += (pred == lbls).sum()
                _count += lbls.size(0)
                _loss += loss.item() * outputs.shape[0]
            logger.info("epoch {} loss {:4f} acc {:4f}".format(epoch, _loss / preds.shape[0], _correct / _count))

            self.scheduler.step()
        logger.info("beta {:.4f} gamma {:.4f}".format(self.beta.cpu().item(), self.gamma.cpu().item()))

    @torch.no_grad()
    def post_process(self, preds, task_size):
        if self.class_specific is False:
            if task_size != 0:
                preds[:, -task_size:] = preds[:, -task_size:] * self.beta + self.gamma
        else:
            preds = preds * self.beta + self.gamma
        return preds


class WA(object):
    def __init__(self):
        self.gamma = None

    @torch.no_grad()
    def update(self, classifier, task_size):
        old_weight_norm = torch.norm(classifier.weight[:-task_size], p=2, dim=1)
        new_weight_norm = torch.norm(classifier.weight[-task_size:], p=2, dim=1)
        self.gamma = old_weight_norm.mean() / new_weight_norm.mean()
        print(self.gamma.cpu().item())

    @torch.no_grad()
    def post_process(self, logits, task_size):
        logits[:, -task_size:] = logits[:, -task_size:] * self.gamma
        return logits


class DownsampleA(nn.Module):
    def __init__(self, nIn, nOut, stride):
        super(DownsampleA, self).__init__()
        assert stride == 2
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        x = self.avg(x)
        return torch.cat((x, x.mul(0)), 1)


class ResNetBasicblock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResNetBasicblock, self).__init__()
        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        basicblock = self.conv_a(x)
        basicblock = self.bn_a(basicblock)
        basicblock = F.relu(basicblock, inplace=True)

        basicblock = self.conv_b(basicblock)
        basicblock = self.bn_b(basicblock)

        if self.downsample is not None:
            residual = self.downsample(x)

        return F.relu(residual + basicblock, inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, block, depth, num_classes, channels=3):
        super(CifarResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
        layer_blocks = (depth - 2) // 6

        self.num_classes = num_classes

        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)

        self.inplanes = 16
        self.stage_1 = self._make_layer(block, 16, layer_blocks, 1)
        self.stage_2 = self._make_layer(block, 32, layer_blocks, 2)
        self.stage_3 = self._make_layer(block, 64, layer_blocks, 2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64 * block.expansion
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = DownsampleA(self.inplanes, planes * block.expansion, stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, feature=False, T=1, labels=False, scale=None, keep=None):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


def resnet20(num_classes=10):
    return CifarResNet(ResNetBasicblock, 20, num_classes)


def resnet10mnist(num_classes=10):
    return CifarResNet(ResNetBasicblock, 10, num_classes, channels=1)


def resnet20mnist(num_classes=10):
    return CifarResNet(ResNetBasicblock, 20, num_classes, channels=1)


def resnet32mnist(num_classes=10):
    return CifarResNet(ResNetBasicblock, 32, num_classes, channels=1)


def resnet32(num_classes=10):
    return CifarResNet(ResNetBasicblock, 32, num_classes)


def resnet44(num_classes=10):
    return CifarResNet(ResNetBasicblock, 44, num_classes)


def resnet56(num_classes=10):
    return CifarResNet(ResNetBasicblock, 56, num_classes)


def resnet110(num_classes=10):
    return CifarResNet(ResNetBasicblock, 110, num_classes)


def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet32":
        return resnet32()
    else:
        raise NotImplementedError("Unknwon convnet type {}.".format(convnet_type))


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100"
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = False
        self.use_bias = use_bias
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.reuse_oldfc = cfg['reuse_oldfc']
        self.out_dim = None
        self.device = device
        self.dataset = dataset

        if self.der:
            self.convnets = nn.ModuleList()
            self.convnets.append(get_convnet(convnet_type, nf=nf, dataset=dataset, start_class=self.start_class))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = get_convnet(convnet_type, nf=nf, dataset=dataset, start_class=self.start_class)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None

        self.n_classes = 0
        self.ntask = 0

        self.postprocessor = None
        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(**cfg['postprocessor']['lr_config'])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        self.to(self.device)

    def forward(self, x):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features = self.convnet(x)
        logits = self.classifier(features)
        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None

        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1
        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)
        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = get_convnet(self.convnet_type)
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        fc = self.classifier
        if fc is not None:
            old_weight = fc.weight.data.clone()

        self.classifier = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes)

        if fc is not None and self.reuse_oldfc:
            self.classifier.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = old_weight
        del fc

        if self.aux_nplus1:
            self.aux_classifier = self._gen_classifier(self.out_dim, n_classes + 1)
        else:
            self.aux_classifier = self._gen_classifier(self.out_dim, self.n_classes + n_classes)

    def _add_classes_single_fc(self, n_classes):
        fc = self.classifier
        if fc is not None:
            old_weight = fc.weight.data.clone()
            if self.use_bias:
                old_bias = fc.bias.data.clone()

        self.classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if fc is not None and self.reuse_oldfc:
            self.classifier.weight.data[:self.n_classes] = old_weight
            if self.use_bias:
                self.classifier.bias.data[:self.n_classes] = old_bias
        del fc

    def _gen_classifier(self, in_features, n_classes):
        if self.weight_normalization:
            classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
            if self.init == "kaiming":
                nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.)

        return classifier


if __name__ == "__main__":
    torch.manual_seed(7)

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

    def base_cfg(der=True, weight_normalization=False, aux_nplus1=True, reuse_oldfc=True):
        return {
            "start_class": 0,
            "weight_normalization": weight_normalization,
            "der": der,
            "aux_n+1": aux_nplus1,
            "reuse_oldfc": reuse_oldfc,
            "postprocessor": {
                "enable": False,
                "type": "none",
                "lr_config": {}
            }
        }

    device = torch.device("cpu")
    image = torch.randn(2, 3, 32, 32)
    der_net = None
    first_weight = None

    print("=" * 70)
    print("DER-ClassIL reproduction benchmark")
    print("=" * 70)

    print("-" * 60)
    print("[Group 1] CosineClassifier")
    try:
        cosine = CosineClassifier(5, 4)
        cosine_x = torch.randn(3, 5)
        cosine_y = cosine(cosine_x)
        check("CosineClassifier output not None", cosine_y is not None)
        check("CosineClassifier weight layout", tuple(cosine.weight.shape) == (4, 5), f"got {tuple(cosine.weight.shape)}")
        check("CosineClassifier sigma exposed", cosine.sigma is not None and tuple(cosine.sigma.shape) == (1,))
        if cosine_y is not None:
            check("CosineClassifier output shape", tuple(cosine_y.shape) == (3, 4), f"got {tuple(cosine_y.shape)}")
            check("CosineClassifier output finite", torch.isfinite(cosine_y).all().item())
            check("Cosine logits bounded by sigma", torch.all(cosine_y.abs() <= cosine.sigma.detach().abs().max() + 1e-5).item())
        else:
            skip_checks(3, "CosineClassifier returned None")
    except Exception as exc:
        skip_checks(6, f"CosineClassifier raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 2] DownsampleA")
    down_y = None
    try:
        downsample = DownsampleA(16, 32, 2)
        down_x = torch.randn(2, 16, 32, 32)
        down_y = downsample(down_x)
        check("DownsampleA output not None", down_y is not None)
        if down_y is not None:
            check("DownsampleA output shape", tuple(down_y.shape) == (2, 32, 16, 16), f"got {tuple(down_y.shape)}")
            check("DownsampleA zero pads channels", torch.allclose(down_y[:, 16:], torch.zeros_like(down_y[:, 16:])))
        else:
            skip_checks(2, "DownsampleA returned None")
    except Exception as exc:
        skip_checks(3, f"DownsampleA raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 3] ResNetBasicblock")
    try:
        if down_y is None:
            skip_checks(4, "ResNetBasicblock depends on a valid DownsampleA output")
        else:
            block = ResNetBasicblock(16, 32, stride=2, downsample=downsample).eval()
            block_y = block(down_x)
            check("ResNetBasicblock output not None", block_y is not None)
            if block_y is not None:
                check("ResNetBasicblock output shape", tuple(block_y.shape) == (2, 32, 16, 16), f"got {tuple(block_y.shape)}")
                check("ResNetBasicblock output finite", torch.isfinite(block_y).all().item())
                check("ResNetBasicblock final ReLU", (block_y >= 0).all().item())
            else:
                skip_checks(3, "ResNetBasicblock returned None")
    except Exception as exc:
        skip_checks(4, f"ResNetBasicblock raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 4] CifarResNet")
    try:
        backbone = resnet20().eval()
        feature = backbone(image)
        check("CifarResNet output not None", feature is not None)
        check("CifarResNet out_dim", backbone.out_dim == 64, f"got {backbone.out_dim}")
        check("CifarResNet stage layout", len(backbone.stage_1) == 3 and len(backbone.stage_2) == 3 and len(backbone.stage_3) == 3)
        if feature is not None:
            check("CifarResNet feature shape", tuple(feature.shape) == (2, 64), f"got {tuple(feature.shape)}")
            check("CifarResNet features finite", torch.isfinite(feature).all().item())
        else:
            skip_checks(2, "CifarResNet returned None")
    except Exception as exc:
        skip_checks(5, f"CifarResNet raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 5] First DER task expansion")
    try:
        der_net = BasicNet("resnet32", base_cfg(der=True), device=device)
        der_net.add_classes(3)
        der_net.eval()
        first = der_net(image)
        check("First DER task counters", der_net.ntask == 1 and der_net.n_classes == 3, f"got ntask={der_net.ntask}, n_classes={der_net.n_classes}")
        check("First DER convnet count", len(der_net.convnets) == 1, f"got {len(der_net.convnets)}")
        check("First DER classifier exists", der_net.classifier is not None)
        check("First DER aux classifier exists", der_net.aux_classifier is not None)
        if der_net.classifier is not None:
            check("First DER classifier shape", tuple(der_net.classifier.weight.shape) == (3, 64), f"got {tuple(der_net.classifier.weight.shape)}")
            first_weight = der_net.classifier.weight.detach().clone()
        else:
            skip_checks(1, "first DER classifier missing")
        if der_net.aux_classifier is not None:
            check("First DER aux classifier shape", tuple(der_net.aux_classifier.weight.shape) == (4, 64), f"got {tuple(der_net.aux_classifier.weight.shape)}")
        else:
            skip_checks(1, "first DER aux classifier missing")
        check("First DER forward output not None", isinstance(first, dict))
        if isinstance(first, dict):
            check("First DER feature shape", tuple(first["feature"].shape) == (2, 64), f"got {tuple(first['feature'].shape)}")
            check("First DER logit shape", tuple(first["logit"].shape) == (2, 3), f"got {tuple(first['logit'].shape)}")
            check("First DER aux absent", first["aux_logit"] is None)
        else:
            skip_checks(3, "First DER forward did not return a dictionary")
    except Exception as exc:
        skip_checks(11, f"First DER task raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 6] Expanded DER task")
    try:
        if der_net is None or first_weight is None:
            skip_checks(11, "expanded DER task depends on a valid first DER task")
        else:
            der_net.add_classes(2)
            der_net.eval()
            second = der_net(image)
            check("Second DER task counters", der_net.ntask == 2 and der_net.n_classes == 5, f"got ntask={der_net.ntask}, n_classes={der_net.n_classes}")
            check("Second DER convnet count", len(der_net.convnets) == 2, f"got {len(der_net.convnets)}")
            check("Expanded DER features_dim", der_net.features_dim == 128, f"got {der_net.features_dim}")
            check("Expanded DER classifier shape", der_net.classifier is not None and tuple(der_net.classifier.weight.shape) == (5, 128))
            if der_net.classifier is not None and tuple(der_net.classifier.weight.shape) == (5, 128):
                check("Expanded DER preserves old weights", torch.allclose(der_net.classifier.weight[:3, :64], first_weight))
            else:
                skip_checks(1, "expanded classifier missing or wrong shape")
            check("Second DER aux classifier shape", der_net.aux_classifier is not None and tuple(der_net.aux_classifier.weight.shape) == (3, 64))
            check("Expanded DER forward output not None", isinstance(second, dict))
            if isinstance(second, dict):
                check("Expanded DER feature shape", tuple(second["feature"].shape) == (2, 128), f"got {tuple(second['feature'].shape)}")
                check("Expanded DER logit shape", tuple(second["logit"].shape) == (2, 5), f"got {tuple(second['logit'].shape)}")
                check("Expanded DER aux shape", tuple(second["aux_logit"].shape) == (2, 3), f"got {tuple(second['aux_logit'].shape)}")
                check("Expanded DER logits finite", torch.isfinite(second["logit"]).all().item())
            else:
                skip_checks(4, "Expanded DER forward did not return a dictionary")
    except Exception as exc:
        skip_checks(11, f"Expanded DER task raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 7] Single-branch BasicNet")
    try:
        single_net = BasicNet("resnet32", base_cfg(der=False, weight_normalization=True), device=device)
        single_net.add_classes(4)
        single_net.eval()
        single = single_net(image)
        check("Single branch uses CosineClassifier", isinstance(single_net.classifier, CosineClassifier))
        check("Single branch features_dim", single_net.features_dim == 64, f"got {single_net.features_dim}")
        check("Single branch output not None", isinstance(single, dict))
        if isinstance(single, dict):
            check("Single branch feature shape", tuple(single["feature"].shape) == (2, 64), f"got {tuple(single['feature'].shape)}")
            check("Single branch logit shape", tuple(single["logit"].shape) == (2, 4), f"got {tuple(single['logit'].shape)}")
            check("Single branch aux absent", single["aux_logit"] is None)
        else:
            skip_checks(3, "Single branch forward did not return a dictionary")
    except Exception as exc:
        skip_checks(6, f"Single-branch BasicNet raised {type(exc).__name__}: {exc}")

    print("-" * 60)
    print("[Group 8] BasicNet copy and freeze")
    try:
        if der_net is None:
            skip_checks(3, "copy/freeze depends on a valid DER model")
        else:
            frozen = der_net.copy().freeze()
            check("BasicNet.copy distinct module", frozen is not der_net)
            check("BasicNet.freeze disables gradients", all(not p.requires_grad for p in frozen.parameters()))
            check("BasicNet.freeze eval mode", not frozen.training)
    except Exception as exc:
        skip_checks(3, f"copy/freeze raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The DER-ClassIL benchmark implementation is complete.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
        raise SystemExit(1)
