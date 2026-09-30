import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
from torchvision import transforms


# ============================================================
# ground_truth.py - DUCK Core Model Components (Self-contained)
# Source: Adversarial/DUCK-master
#
# Contains the backbone components needed for DUCK experiments and the
# distance-based unlearning mechanism from Unlearning_methods.py.
# No dataset construction, checkpoint loading, CLI wrappers, MIA pipelines,
# result publishing, or remote weight downloads are included.
# ============================================================


class _BenchmarkOpt:
    dataset = "cifar10"
    model = "AllCNN"
    device = "cpu"
    num_classes = 3
    lr_unlearn = 0.001
    momentum_unlearn = 0.9
    wd_unlearn = 0.0
    epochs_unlearn = 1
    target_accuracy = -1.0
    scheduler = []
    temperature = 2
    lambda_1 = 1
    lambda_2 = 0
    batch_fgt_ret_ratio = 1
    mode = "HR"


opt = _BenchmarkOpt()


# --- [Original file: src/utils.py] ---
def accuracy(net, loader):
    """Return accuracy on a dataset given by the data loader."""
    correct = 0
    total = 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(opt.device), targets.to(opt.device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return correct / total


# --- [Original file: src/models/allcnn.py] ---
def size_conv(size, kernel, stride=1, padding=0):
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def size_max_pool(size, kernel, stride=None, padding=0):
    if stride == None:
        stride = kernel
    out = int(((size - kernel + 2 * padding) / stride) + 1)
    return out


def calc_feat_linear_cifar(size):
    feat = size_conv(size, 3, 1, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 3, 1, 1)
    out = size_max_pool(feat, 2, 2)
    return out


def calc_feat_linear_mnist(size):
    feat = size_conv(size, 5, 1)
    feat = size_max_pool(feat, 2, 2)
    feat = size_conv(feat, 5, 1)
    out = size_max_pool(feat, 2, 2)
    return out


def init_params(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        nn.init.zeros_(m.bias)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)


class AllCNN(nn.Module):
    def __init__(self, dropout_prob=0.1, n_channels=3, num_classes=10, dropout=False, filters_percentage=1., batch_norm=True,img_size=32):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        self.features = nn.Sequential(
            Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm),
            Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),
            nn.Dropout(p=dropout_prob) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm),  # 14
            nn.Dropout(p=dropout_prob) if dropout else Identity(),
            Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm),
            Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm),
            nn.AvgPool2d(8),
            Flatten()
        )
        if img_size == 32:
            shape = 192
        else:
            shape = 768
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_prob),
            nn.Linear(shape, 384),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(384,num_classes)
        )

    def forward(self, x):
        features = self.features(x)
        output = self.classifier(features)
        return output


# --- [Original file: src/models/model.py] ---
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers

    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, num_classes=100):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        #we use a different inputsize than the original paper
        #so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet18(weights=None,num_classes=10):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)


# --- [Original file: src/Unlearning_methods.py] ---
mean = {
            'cifar10': (0.4914, 0.4822, 0.4465),
            'cifar100': (0.5071, 0.4867, 0.4408),
            'tinyImagenet': (0.485, 0.456, 0.406),
            'VGG':(0.547, 0.460, 0.404),
            }

std = {
            'cifar10': (0.2023, 0.1994, 0.2010),
            'cifar100': (0.2675, 0.2565, 0.2761),
            'tinyImagenet': (0.229, 0.224, 0.225),
            'VGG':[0.323, 0.298, 0.263]
            }


class BaseMethod:
    def __init__(self, net, retain, forget,test=None):
        self.net = net
        self.retain = retain
        self.forget = forget
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=opt.lr_unlearn, momentum=opt.momentum_unlearn, weight_decay=opt.wd_unlearn)
        self.epochs = opt.epochs_unlearn
        self.target_accuracy = opt.target_accuracy
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.scheduler, gamma=0.5)
        if test is None:
            pass
        else:
            print('test loader passed')
            self.test = test
    def loss_f(self, net, inputs, targets):
        return None

    def run(self):
        self.net.train()
        for ep in tqdm(range(self.epochs)):
            for inputs, targets in self.loader:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                self.optimizer.zero_grad()
                loss = self.loss_f(inputs, targets)
                loss.backward()
                self.optimizer.step()

            with torch.no_grad():
                if ep%5==0:
                    self.net.eval()
                    curr_acc = accuracy(self.net, self.forget)
                    acc = accuracy(self.net,self.test)
                    self.net.train()
                    print(f"ACCURACY FORGET SET: {curr_acc:.3f}, target is {self.target_accuracy:.3f}, test is: {acc:.3f}")

            self.scheduler.step()
        self.net.eval()
        return self.net

    def evalNet(self):
        #compute model accuracy on self.loader

        self.net.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for inputs, targets in self.retain:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

            correct2 = 0
            total2 = 0
            for inputs, targets in self.forget:
                inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total2 += targets.size(0)
                correct2+= (predicted == targets).sum().item()

            if not(self.test is None):
                correct3 = 0
                total3 = 0
                for inputs, targets in self.test:
                    inputs, targets = inputs.to(opt.device), targets.to(opt.device)
                    outputs = self.net(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total3 += targets.size(0)
                    correct3+= (predicted == targets).sum().item()
        self.net.train()
        if self.test is None:
            return correct/total,correct2/total2
        else:
            return correct/total,correct2/total2,correct3/total3


class DUCK(BaseMethod):
    def __init__(self, net, retain, forget,test,class_to_remove=None):
        super().__init__(net, retain, forget, test)
        self.loader = None
        self.class_to_remove = class_to_remove
        transform_dset_list = [   transforms.RandomCrop(64, padding=8) if opt.dataset == 'tinyImagenet' else transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean[opt.dataset],std[opt.dataset])]

        if opt.model ==  'ViT':
            transform_dset_list.insert(2,transforms.Resize(224,antialias=True))

        self.transform_dset = transforms.Compose(transform_dset_list)
    def pairwise_cos_dist(self, x, y):
        """
        [TODO] Compute the pairwise cosine distance matrix used to match forget embeddings to centroids.

        Input:
            x: (batch_x, feature_dim) - embeddings for forget samples.
            y: (batch_y, feature_dim) - class centroid embeddings or comparison features.

        Output: (batch_x, batch_y) - pairwise distance matrix where lower values mean more similar directions.

"""
        pass

    def run(self):
        """
        [TODO] Execute DUCK's centroid-guided unlearning procedure.

        Input:
            self.net: image classifier whose backbone embedding and classifier head are updated.
            self.retain: iterable of retained samples, each batch shaped (batch, 3, height, width) with labels.
            self.forget: iterable of samples to forget, each batch shaped (batch, 3, height, width) with labels.
            self.test: iterable used only for monitoring accuracy after unlearning.
            self.class_to_remove: class index or list controlling which retain centroids are available.

        Output: nn.Module - the same network after DUCK updates, left in evaluation mode.

"""
        pass


# ============================================================
# __main__: Automated test suite for 2 ablated functions
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
    print("DUCK: distance-based unlearning benchmark")
    print("Automated Test Suite - 2 ablated targets")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    opt.device = "cpu"
    opt.model = "AllCNN"
    opt.dataset = "cifar10"
    opt.num_classes = 3
    opt.epochs_unlearn = 1
    opt.lr_unlearn = 0.001
    opt.wd_unlearn = 0.0
    opt.lambda_1 = 1.0
    opt.lambda_2 = 0.0
    opt.temperature = 2.0
    opt.batch_fgt_ret_ratio = 1
    opt.target_accuracy = -1.0
    opt.scheduler = []
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/2: DUCK.pairwise_cos_dist
    # ==============================================================
    print("-" * 60)
    print("[Test 1/2] DUCK.pairwise_cos_dist - nearest-centroid cosine distance")
    try:
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device)
        y = torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], device=device)
        model = AllCNN(num_classes=3, dropout_prob=0.0)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(3, 3, 32, 32), torch.tensor([0, 1, 2])), batch_size=3)
        duck = DUCK(model, loader, loader, loader)
        dist = duck.pairwise_cos_dist(x, y)
        check("pairwise_cos_dist output not None", dist is not None)
        if dist is not None:
            check("pairwise_cos_dist output shape", tuple(dist.shape) == (2, 3),
                  f"expected (2, 3), got {tuple(dist.shape)}")
            check("pairwise_cos_dist output finite", torch.isfinite(dist).all().item())
            check("pairwise_cos_dist identical vectors zero", torch.allclose(torch.diag(dist[:, :2]), torch.zeros(2), atol=1e-6))
            check("pairwise_cos_dist opposite vector high", dist[0, 2].item() > 1.9)
            check("pairwise_cos_dist gradient path", dist.requires_grad is False)
        else:
            skip_checks(5, "DUCK.pairwise_cos_dist returned None")
    except Exception as e:
        skip_checks(6, f"DUCK.pairwise_cos_dist raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/2: DUCK.run
    # ==============================================================
    print("-" * 60)
    print("[Test 2/2] DUCK.run - centroid-guided unlearning step")
    try:
        torch.manual_seed(7)
        model = AllCNN(num_classes=3, dropout_prob=0.0).to(device)
        retain_x = torch.randn(6, 3, 32, 32, device=device)
        retain_y = torch.tensor([0, 1, 2, 0, 1, 2], device=device)
        forget_x = torch.randn(3, 3, 32, 32, device=device)
        forget_y = torch.tensor([0, 1, 2], device=device)
        test_x = torch.randn(3, 3, 32, 32, device=device)
        test_y = torch.tensor([0, 1, 2], device=device)
        retain_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(retain_x.cpu(), retain_y.cpu()), batch_size=6, shuffle=False)
        forget_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(forget_x.cpu(), forget_y.cpu()), batch_size=3, shuffle=False)
        test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_x.cpu(), test_y.cpu()), batch_size=3, shuffle=False)
        before = [p.detach().clone() for p in model.parameters()]
        duck = DUCK(model, retain_loader, forget_loader, test_loader)
        updated = duck.run()
        check("DUCK.run output not None", updated is not None)
        if updated is not None:
            check("DUCK.run returns model", isinstance(updated, nn.Module))
            updated.eval()
            with torch.no_grad():
                logits = updated(test_x.cpu())
            check("DUCK.run logits shape", tuple(logits.shape) == (3, 3),
                  f"expected (3, 3), got {tuple(logits.shape)}")
            check("DUCK.run logits finite", torch.isfinite(logits).all().item())
            changed = any(not torch.allclose(p0, p1.detach().cpu()) for p0, p1 in zip(before, updated.parameters()))
            check("DUCK.run updates parameters", changed)
            check("DUCK.run final eval mode", updated.training is False)
        else:
            skip_checks(5, "DUCK.run returned None")
    except Exception as e:
        skip_checks(6, f"DUCK.run raised {type(e).__name__}: {e}")
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
