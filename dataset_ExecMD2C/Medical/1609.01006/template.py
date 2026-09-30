"""
ground_truth.py for UNet-Zoo core model components.

Source-consolidated from:
- models.py
- CLSTM.py

Only the UNet, Small UNet, upsampling/skip blocks, ConvLSTM, and bidirectional
ConvLSTM segmentation head are included. Data loading, non-model orchestration,
state-file handling, metrics, plotting, and CLI code are intentionally excluded.
"""

# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable


# --- [Original file: models.py] ---
class UNet(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(UNet, self).__init__()
        num_feat = [64, 128, 256, 512, 1024]

        self.down1 = nn.Sequential(Conv3x3(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[1], num_feat[2]))

        self.down4 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   Conv3x3(num_feat[2], num_feat[3]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    Conv3x3(num_feat[3], num_feat[4]))

        self.up1 = UpConcat(num_feat[4], num_feat[3])
        self.upconv1 = Conv3x3(num_feat[4], num_feat[3])

        self.up2 = UpConcat(num_feat[3], num_feat[2])
        self.upconv2 = Conv3x3(num_feat[3], num_feat[2])

        self.up3 = UpConcat(num_feat[2], num_feat[1])
        self.upconv3 = Conv3x3(num_feat[2], num_feat[1])

        self.up4 = UpConcat(num_feat[1], num_feat[0])
        self.upconv4 = Conv3x3(num_feat[1], num_feat[0])

        self.final = nn.Sequential(nn.Conv2d(num_feat[0],
                                             num_classes,
                                             kernel_size=1),
                                   nn.Softmax2d())

    def forward(self, inputs, return_features=False):
        """
        [TODO] Run the full four-level UNet encoder-decoder path with skip concatenations.

        Input:
            inputs: (batch, num_channels, height, width) - 2D image batch.
            return_features: bool - when True, return the final decoder feature map instead of class probabilities.

        Output:
            If return_features is False: (batch, num_classes, height, width) - per-pixel class probabilities.
            If return_features is True: (batch, 64, height, width) - final decoder features before the classifier head.

"""
        pass


class UNetSmall(nn.Module):
    def __init__(self, num_channels=1, num_classes=2):
        super(UNetSmall, self).__init__()
        num_feat = [32, 64, 128, 256]

        self.down1 = nn.Sequential(Conv3x3Small(num_channels, num_feat[0]))

        self.down2 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.BatchNorm2d(num_feat[0]),
                                   Conv3x3Small(num_feat[0], num_feat[1]))

        self.down3 = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                   nn.BatchNorm2d(num_feat[1]),
                                   Conv3x3Small(num_feat[1], num_feat[2]))

        self.bottom = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                    nn.BatchNorm2d(num_feat[2]),
                                    Conv3x3Small(num_feat[2], num_feat[3]),
                                    nn.BatchNorm2d(num_feat[3]))

        self.up1 = UpSample(num_feat[3], num_feat[2])
        self.upconv1 = nn.Sequential(Conv3x3Small(num_feat[3] + num_feat[2], num_feat[2]),
                                     nn.BatchNorm2d(num_feat[2]))

        self.up2 = UpSample(num_feat[2], num_feat[1])
        self.upconv2 = nn.Sequential(Conv3x3Small(num_feat[2] + num_feat[1], num_feat[1]),
                                     nn.BatchNorm2d(num_feat[1]))

        self.up3 = UpSample(num_feat[1], num_feat[0])
        self.upconv3 = nn.Sequential(Conv3x3Small(num_feat[1] + num_feat[0], num_feat[0]),
                                     nn.BatchNorm2d(num_feat[0]))

        self.final = nn.Sequential(nn.Conv2d(num_feat[0],
                                             1,
                                             kernel_size=1),
                                   nn.Sigmoid())

    def forward(self, inputs, return_features=False):
        """
        [TODO] Run the compact three-level UNet encoder-decoder path.

        Input:
            inputs: (batch, num_channels, height, width) - 2D image batch.
            return_features: bool - when True, return the final decoder features for downstream BDCLSTM use.

        Output:
            If return_features is False: (batch, 1, height, width) - sigmoid segmentation map.
            If return_features is True: (batch, 32, height, width) - final decoder feature map.

"""
        pass


class Conv3x3(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Drop(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Drop, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.Dropout(p=0.2),
                                   nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.BatchNorm2d(out_feat),
                                   nn.ReLU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class Conv3x3Small(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(Conv3x3Small, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(in_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ELU(),
                                   nn.Dropout(p=0.2))

        self.conv2 = nn.Sequential(nn.Conv2d(out_feat, out_feat,
                                             kernel_size=3,
                                             stride=1,
                                             padding=1),
                                   nn.ELU())

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class UpConcat(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpConcat, self).__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.deconv = nn.ConvTranspose2d(in_feat, out_feat,
        #                                  kernel_size=3,
        #                                  stride=1,
        #                                  dilation=1)

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        """
        [TODO] Upsample decoder features with transposed convolution and concatenate the encoder skip.

        Input:
            inputs: (batch, in_feat, height, width) - low-resolution decoder tensor.
            down_outputs: (batch, out_feat, 2*height, 2*width) - encoder skip tensor.

        Output: (batch, 2*out_feat, 2*height, 2*width) - channel concatenation of skip and upsampled decoder tensor.

"""
        pass


class UpSample(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(UpSample, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv = nn.ConvTranspose2d(in_feat,
                                         out_feat,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, inputs, down_outputs):
        """
        [TODO] Upsample compact-UNet decoder features and concatenate the encoder skip.

        Input:
            inputs: (batch, in_feat, height, width) - low-resolution decoder tensor.
            down_outputs: (batch, skip_channels, 2*height, 2*width) - encoder skip tensor.

        Output: (batch, in_feat + skip_channels, 2*height, 2*width) - concatenated decoder and skip tensor.

"""
        pass


# --- [Original file: CLSTM.py] ---
# Batch x NumChannels x Height x Width
# UNET --> BatchSize x 1 (3?) x 240 x 240
# BDCLSTM --> BatchSize x 64 x 240 x240

''' Class CLSTMCell.
    This represents a single node in a CLSTM series.
    It produces just one time (spatial) step output.
'''


class CLSTMCell(nn.Module):

    # Constructor
    def __init__(self, input_channels, hidden_channels,
                 kernel_size, bias=True):
        super(CLSTMCell, self).__init__()

        assert hidden_channels % 2 == 0

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.kernel_size = kernel_size
        self.num_features = 4

        self.padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(self.input_channels + self.hidden_channels,
                              self.num_features * self.hidden_channels,
                              self.kernel_size,
                              1,
                              self.padding)

    # Forward propogation formulation
    def forward(self, x, h, c):
        """
        [TODO] Run one convolutional LSTM cell update for a spatial feature map.

        Input:
            x: (batch, input_channels, height, width) - current input feature map.
            h: (batch, hidden_channels, height, width) - previous hidden state.
            c: (batch, hidden_channels, height, width) - previous cell state.

        Output:
            h: (batch, hidden_channels, height, width) - updated hidden state.
            c: (batch, hidden_channels, height, width) - updated cell state.

"""
        pass

    @staticmethod
    def init_hidden(batch_size, hidden_c, shape):
        try:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda(),
               Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])).cuda())
        except:
            return(Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])),
                    Variable(torch.zeros(batch_size,
                                    hidden_c,
                                    shape[0],
                                    shape[1])))


''' Class CLSTM.
    This represents a series of CLSTM nodes (one direction)
'''


class CLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True):
        super(CLSTM, self).__init__()

        # store stuff
        self.input_channels = [input_channels] + hidden_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = len(hidden_channels)

        self.bias = bias
        self.all_layers = []

        # create a node for each layer in the CLSTM
        for layer in range(self.num_layers):
            name = 'cell{}'.format(layer)
            cell = CLSTMCell(self.input_channels[layer],
                             self.hidden_channels[layer],
                             self.kernel_size,
                             self.bias)
            setattr(self, name, cell)
            self.all_layers.append(cell)

    # Forward propogation
    # x --> BatchSize x NumSteps x NumChannels x Height x Width
    #       BatchSize x 2 x 64 x 240 x 240
    def forward(self, x):
        """
        [TODO] Run a convolutional LSTM over a sequence of spatial feature maps.

        Input:
            x: (batch, steps, input_channels, height, width) - ordered feature-map sequence.

        Output: list of length `steps`, where each item is
            (batch, hidden_channels[-1], height, width) - output from the final recurrent layer.

"""
        pass


class BDCLSTM(nn.Module):
    # Constructor
    def __init__(self, input_channels=64, hidden_channels=[64],
                 kernel_size=5, bias=True, num_classes=2):

        super(BDCLSTM, self).__init__()
        self.forward_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.reverse_net = CLSTM(
            input_channels, hidden_channels, kernel_size, bias)
        self.conv = nn.Conv2d(
            2 * hidden_channels[-1], num_classes, kernel_size=1)
        self.soft = nn.Softmax2d()

    # Forward propogation
    # x --> BatchSize x NumChannels x Height x Width
    #       BatchSize x 64 x 240 x 240
    def forward(self, x1, x2, x3):
        """
        [TODO] Fuse three adjacent UNet feature maps with bidirectional ConvLSTM for segmentation.

        Input:
            x1: (batch, input_channels, height, width) - previous slice feature map.
            x2: (batch, input_channels, height, width) - center slice feature map.
            x3: (batch, input_channels, height, width) - next slice feature map.

        Output: (batch, num_classes, height, width) - per-pixel class probabilities.

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("UNet-Zoo core model component benchmark")
    print("=" * 70)
    print(f"Device: {device}")

    print("-" * 60)
    print("[Group 1] Conv blocks")
    try:
        conv = Conv3x3(1, 8).to(device)
        x = torch.randn(2, 1, 32, 32, device=device)
        y = conv(x)
        check("Conv3x3 output not None", y is not None)
        check("Conv3x3 output shape", tuple(y.shape) == (2, 8, 32, 32), f"got {tuple(y.shape)}")
        check("Conv3x3 output finite", torch.isfinite(y).all().item())
        small = Conv3x3Small(1, 8).to(device)
        ys = small(x)
        check("Conv3x3Small output shape", tuple(ys.shape) == (2, 8, 32, 32), f"got {tuple(ys.shape)}")
        check("Conv3x3Small output finite", torch.isfinite(ys).all().item())
    except Exception as exc:
        skip_checks(5, f"Conv blocks raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 2] Upsampling skip blocks")
    try:
        up_concat = UpConcat(16, 8).to(device)
        low = torch.randn(2, 16, 8, 8, device=device)
        skip = torch.randn(2, 8, 16, 16, device=device)
        y_concat = up_concat(low, skip)
        check("UpConcat output not None", y_concat is not None)
        check("UpConcat output shape", tuple(y_concat.shape) == (2, 16, 16, 16), f"got {tuple(y_concat.shape)}")
        check("UpConcat preserves skip first channels", torch.allclose(y_concat[:, :8], skip, atol=1e-6))

        up_sample = UpSample(16, 8).to(device)
        skip_small = torch.randn(2, 8, 16, 16, device=device)
        y_sample = up_sample(low, skip_small)
        check("UpSample output not None", y_sample is not None)
        check("UpSample output shape", tuple(y_sample.shape) == (2, 24, 16, 16), f"got {tuple(y_sample.shape)}")
        check("UpSample preserves skip last channels", torch.allclose(y_sample[:, 16:], skip_small, atol=1e-6))
    except Exception as exc:
        skip_checks(6, f"Upsampling blocks raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 3] UNet and UNetSmall")
    try:
        unet = UNet(num_channels=1, num_classes=2).to(device)
        unet.eval()
        x_unet = torch.randn(1, 1, 32, 32, device=device)
        y_unet = unet(x_unet)
        f_unet = unet(x_unet, return_features=True)
        check("UNet output not None", y_unet is not None)
        check("UNet output shape", tuple(y_unet.shape) == (1, 2, 32, 32), f"got {tuple(y_unet.shape)}")
        check("UNet class probabilities sum to one", torch.allclose(y_unet.sum(dim=1), torch.ones(1, 32, 32, device=device), atol=1e-5))
        check("UNet feature shape", tuple(f_unet.shape) == (1, 64, 32, 32), f"got {tuple(f_unet.shape)}")

        small_unet = UNetSmall(num_channels=1, num_classes=2).to(device)
        small_unet.eval()
        y_small = small_unet(x_unet)
        f_small = small_unet(x_unet, return_features=True)
        check("UNetSmall output not None", y_small is not None)
        check("UNetSmall output shape", tuple(y_small.shape) == (1, 1, 32, 32), f"got {tuple(y_small.shape)}")
        check("UNetSmall sigmoid range", y_small.min().item() >= 0.0 and y_small.max().item() <= 1.0)
        check("UNetSmall feature shape", tuple(f_small.shape) == (1, 32, 32, 32), f"got {tuple(f_small.shape)}")
    except Exception as exc:
        skip_checks(8, f"UNet models raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 4] CLSTMCell")
    try:
        cell = CLSTMCell(input_channels=4, hidden_channels=6, kernel_size=3).to(device)
        x_cell = torch.randn(2, 4, 8, 8, device=device)
        h0 = torch.zeros(2, 6, 8, 8, device=device)
        c0 = torch.zeros(2, 6, 8, 8, device=device)
        h1, c1 = cell(x_cell, h0, c0)
        check("CLSTMCell hidden not None", h1 is not None)
        check("CLSTMCell cell not None", c1 is not None)
        check("CLSTMCell hidden shape", tuple(h1.shape) == (2, 6, 8, 8), f"got {tuple(h1.shape)}")
        check("CLSTMCell cell shape", tuple(c1.shape) == (2, 6, 8, 8), f"got {tuple(c1.shape)}")
        check("CLSTMCell outputs finite", torch.isfinite(h1).all().item() and torch.isfinite(c1).all().item())
    except Exception as exc:
        skip_checks(5, f"CLSTMCell raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 5] CLSTM sequence")
    try:
        clstm = CLSTM(input_channels=4, hidden_channels=[6], kernel_size=3).to(device)
        seq = torch.randn(2, 3, 4, 8, 8, device=device)
        outputs = clstm(seq)
        check("CLSTM output list", isinstance(outputs, list))
        check("CLSTM output length", len(outputs) == 3)
        check("CLSTM final shape", tuple(outputs[-1].shape) == (2, 6, 8, 8), f"got {tuple(outputs[-1].shape)}")
        check("CLSTM outputs finite", all(torch.isfinite(o).all().item() for o in outputs))
        check("CLSTM outputs change over time", not torch.allclose(outputs[0], outputs[-1]))
    except Exception as exc:
        skip_checks(5, f"CLSTM raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 6] BDCLSTM segmentation head")
    try:
        bdclstm = BDCLSTM(input_channels=4, hidden_channels=[6], kernel_size=3, num_classes=2).to(device)
        x1 = torch.randn(2, 4, 8, 8, device=device)
        x2 = torch.randn(2, 4, 8, 8, device=device)
        x3 = torch.randn(2, 4, 8, 8, device=device)
        y_bdc = bdclstm(x1, x2, x3)
        check("BDCLSTM output not None", y_bdc is not None)
        check("BDCLSTM output shape", tuple(y_bdc.shape) == (2, 2, 8, 8), f"got {tuple(y_bdc.shape)}")
        check("BDCLSTM output finite", torch.isfinite(y_bdc).all().item())
        check("BDCLSTM softmax sums", torch.allclose(y_bdc.sum(dim=1), torch.ones(2, 8, 8, device=device), atol=1e-5))
        check("BDCLSTM has two directions", isinstance(bdclstm.forward_net, CLSTM) and isinstance(bdclstm.reverse_net, CLSTM))
    except Exception as exc:
        skip_checks(5, f"BDCLSTM raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some target functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
