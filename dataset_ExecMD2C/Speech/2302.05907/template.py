# ============================================================
# ground_truth.py - LipLearner Core Model Components
# Source:
#   Speech/LipLearner-main/pretraining/model/video_cnn.py
#   Speech/LipLearner-main/pretraining/model/model.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast


# --- [Original file: pretraining/model/video_cnn.py] ---
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, se=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.se = se
        
        if(self.se):
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.conv3 = conv1x1(planes, planes//16)
            self.conv4 = conv1x1(planes//16, planes)

    def forward(self, x):
        """[TODO] Reproduce the ResNet BasicBlock with optional SE gating.

        Input:
            x: Tensor with shape (batch, inplanes, height, width).

        Output:
            Tensor with shape (batch, planes, out_height, out_width), matching
            the residual path shape after optional downsampling.

"""
        pass


class ResNet(nn.Module):

    def __init__(self, block, layers, se=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.se = se
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.bn = nn.BatchNorm1d(512)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
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
        layers.append(block(self.inplanes, planes, stride, downsample, se=self.se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, se=self.se))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.bn(x)
        return x        


class VideoCNN(nn.Module):
    def __init__(self, se=False):
        super(VideoCNN, self).__init__()
        
        # frontend3D
        self.frontend3D = nn.Sequential(
                nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
                )
        # resnet
        self.resnet18 = ResNet(BasicBlock, [2, 2, 2, 2], se=se)
        self.dropout = nn.Dropout(p=0.5)

        # backend_gru
        # initialize
        self._initialize_weights()
    
    def visual_frontend_forward(self, x):
        """[TODO] Reproduce the visual frontend feature extractor.

        Input:
            x: Tensor with shape (batch, time, 1, height, width), containing
               grayscale lip-region video frames.

        Output:
            Tensor with shape (batch * time, 512), containing one ResNet feature
            vector per frame after the 3D frontend.

"""
        pass
    
    def forward(self, x):
        """[TODO] Reproduce the VideoCNN sequence encoder.

        Input:
            x: Tensor with shape (batch, time, 1, height, width).

        Output:
            Tensor with shape (batch, time, 512), containing the per-frame
            visual speech features.

"""
        pass

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# --- [Original file: pretraining/model/model.py] ---
class VideoModel(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(VideoModel, self).__init__()   
        self.args = args
        
        self.video_cnn = VideoCNN(se=True)        
        in_dim = 512 + 1
        self.gru = nn.GRU(in_dim, 1024, 3, batch_first=True, bidirectional=True, dropout=0.2)        

        self.prejection_head = nn.Linear(1024*2, args.n_dimention)     
        self.dropout = nn.Dropout(p=dropout)        

    def forward(self, v, border=None):
        """[TODO] Reproduce the LipLearner top-level video encoder.

        Input:
            v: Tensor with shape (batch, time, 1, height, width), video frames.
            border: Tensor with shape (batch, time), per-frame border/duration
                indicator appended to visual features.

        Output:
            Tensor with shape (batch, args.n_dimention), the projected visual
            speech embedding.

"""
        pass


# ============================================================
# __main__: Automated test suite for 4 ablated functions
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

    class Args:
        n_dimention = 32

    print("=" * 70)
    print("LipLearner: Visual Speech Representation Encoder")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: BasicBlock.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] BasicBlock.forward - residual convolution with optional SE gating")
    try:
        module = BasicBlock(64, 64, se=True).to(device)
        module.eval()
        x = torch.randn(2, 64, 16, 16, device=device, requires_grad=True)
        y = module(x)
        check("BasicBlock output not None", y is not None)
        if y is not None:
            check("BasicBlock output shape", tuple(y.shape) == (2, 64, 16, 16), f"expected (2, 64, 16, 16), got {tuple(y.shape)}")
            check("BasicBlock output finite", torch.isfinite(y).all().item())
            check("BasicBlock SE layers exist", hasattr(module, "gap") and hasattr(module, "conv3") and hasattr(module, "conv4"))
            (y ** 2).sum().backward()
            check("BasicBlock routes gradient through SE gate", module.conv4.weight.grad is not None and module.conv4.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "BasicBlock.forward returned None")
    except Exception as e:
        skip_checks(5, f"BasicBlock.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: VideoCNN.visual_frontend_forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] VideoCNN.visual_frontend_forward - 3D frontend to per-frame ResNet features")
    try:
        module = VideoCNN(se=True).to(device)
        module.eval()
        x = torch.randn(2, 4, 1, 32, 32, device=device, requires_grad=True)
        y = module.visual_frontend_forward(x)
        check("visual_frontend output not None", y is not None)
        if y is not None:
            check("visual_frontend output shape", tuple(y.shape) == (8, 512), f"expected (8, 512), got {tuple(y.shape)}")
            check("visual_frontend output finite", torch.isfinite(y).all().item())
            check("visual_frontend flattens batch and time together", y.shape[0] == x.shape[0] * x.shape[1])
            (y ** 2).sum().backward()
            check("visual_frontend routes gradient to 3D frontend", module.frontend3D[0].weight.grad is not None and module.frontend3D[0].weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "VideoCNN.visual_frontend_forward returned None")
    except Exception as e:
        skip_checks(5, f"VideoCNN.visual_frontend_forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: VideoCNN.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] VideoCNN.forward - video clip to feature sequence")
    try:
        module = VideoCNN(se=True).to(device)
        module.eval()
        x = torch.randn(2, 4, 1, 32, 32, device=device, requires_grad=True)
        y = module(x)
        check("VideoCNN output not None", y is not None)
        if y is not None:
            check("VideoCNN output shape", tuple(y.shape) == (2, 4, 512), f"expected (2, 4, 512), got {tuple(y.shape)}")
            check("VideoCNN output finite", torch.isfinite(y).all().item())
            check("VideoCNN preserves temporal length", y.shape[1] == x.shape[1])
            (y ** 2).sum().backward()
            check("VideoCNN routes gradient to ResNet", module.resnet18.layer1[0].conv1.weight.grad is not None and module.resnet18.layer1[0].conv1.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "VideoCNN.forward returned None")
    except Exception as e:
        skip_checks(5, f"VideoCNN.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: VideoModel.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] VideoModel.forward - visual encoder, border feature, GRU projection")
    try:
        module = VideoModel(Args(), dropout=0.0).to(device)
        module.eval()
        v = torch.randn(2, 4, 1, 32, 32, device=device, requires_grad=True)
        border = torch.ones(2, 4, device=device)
        y = module(v, border)
        check("VideoModel output not None", y is not None)
        if y is not None:
            check("VideoModel output shape", tuple(y.shape) == (2, 32), f"expected (2, 32), got {tuple(y.shape)}")
            check("VideoModel output finite", torch.isfinite(y).all().item())
            check("VideoModel GRU consumes visual plus border feature", module.gru.input_size == 513)
            (y ** 2).sum().backward()
            check("VideoModel routes gradient to projection head", module.prejection_head.weight.grad is not None and module.prejection_head.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "VideoModel.forward returned None")
    except Exception as e:
        skip_checks(5, f"VideoModel.forward raised {type(e).__name__}: {e}")
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
