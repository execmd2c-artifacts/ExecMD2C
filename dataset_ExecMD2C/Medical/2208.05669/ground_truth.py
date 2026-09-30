#!/usr/bin/env python3
"""PA-Seg core model components and paper-specific losses."""

# ============================================================
# ground_truth.py - PA-Seg Core Model Components
# Source: Medical/PA-Seg-main
#
# Contains ONLY model architecture components and paper-specific
# loss/regularization modules. No training, inference, dataset, or CLI code.
# ============================================================

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import unfoldNd


# --- [Original file: network/network.py] ---
CHANNEL_1 = 16
CHANNEL_2 = 32
CHANNEL_3 = 64
CHANNEL_4 = 128
CHANNEL_5 = 256
CHANNEL_INT = 8
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('InstanceNorm3d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv3d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
            nn.InstanceNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=CHANNEL_1)
        self.Conv2 = conv_block(ch_in=CHANNEL_1,ch_out=CHANNEL_2)
        self.Conv3 = conv_block(ch_in=CHANNEL_2,ch_out=CHANNEL_3)
        self.Conv4 = conv_block(ch_in=CHANNEL_3,ch_out=CHANNEL_4)
        self.Conv5 = conv_block(ch_in=CHANNEL_4,ch_out=CHANNEL_5)

        self.Up5 = up_conv(ch_in=CHANNEL_5,ch_out=CHANNEL_4)
        self.Up_conv5 = conv_block(ch_in=CHANNEL_5, ch_out=CHANNEL_4)

        self.Up4 = up_conv(ch_in=CHANNEL_4,ch_out=CHANNEL_3)
        self.Up_conv4 = conv_block(ch_in=CHANNEL_4, ch_out=CHANNEL_3)

        self.Up3 = up_conv(ch_in=CHANNEL_3,ch_out=CHANNEL_2)
        self.Up_conv3 = conv_block(ch_in=CHANNEL_3, ch_out=CHANNEL_2)

        self.Up2 = up_conv(ch_in=CHANNEL_2,ch_out=CHANNEL_1)
        self.Up_conv2 = conv_block(ch_in=CHANNEL_2, ch_out=CHANNEL_1)

        self.Conv_1x1 = nn.Conv3d(CHANNEL_1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()

        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=CHANNEL_1)
        self.Conv2 = conv_block(ch_in=CHANNEL_1,ch_out=CHANNEL_2)
        self.Conv3 = conv_block(ch_in=CHANNEL_2,ch_out=CHANNEL_3)
        self.Conv4 = conv_block(ch_in=CHANNEL_3,ch_out=CHANNEL_4)
        self.Conv5 = conv_block(ch_in=CHANNEL_4,ch_out=CHANNEL_5)

        self.Up5 = up_conv(ch_in=CHANNEL_5,ch_out=CHANNEL_4)
        self.Att5 = Attention_block(F_g=CHANNEL_4,F_l=CHANNEL_4,F_int=CHANNEL_3)
        self.Up_conv5 = conv_block(ch_in=CHANNEL_5, ch_out=CHANNEL_4)

        self.Up4 = up_conv(ch_in=CHANNEL_4,ch_out=CHANNEL_3)
        self.Att4 = Attention_block(F_g=CHANNEL_3,F_l=CHANNEL_3,F_int=CHANNEL_2)
        self.Up_conv4 = conv_block(ch_in=CHANNEL_4, ch_out=CHANNEL_3)

        self.Up3 = up_conv(ch_in=CHANNEL_3,ch_out=CHANNEL_2)
        self.Att3 = Attention_block(F_g=CHANNEL_2,F_l=CHANNEL_2,F_int=CHANNEL_1)
        self.Up_conv3 = conv_block(ch_in=CHANNEL_3, ch_out=CHANNEL_2)

        self.Up2 = up_conv(ch_in=CHANNEL_2,ch_out=CHANNEL_1)
        self.Att2 = Attention_block(F_g=CHANNEL_1,F_l=CHANNEL_1,F_int=CHANNEL_INT)
        self.Up_conv2 = conv_block(ch_in=CHANNEL_2, ch_out=CHANNEL_1)

        self.Conv_1x1 = nn.Conv3d(CHANNEL_1,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


# --- [Original file: network/unet2d5.py] ---
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class ConvNormNonlinBlock(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        conv_op=nn.Conv3d,
        conv_kwargs=None,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None):

        """
        Block: Conv->Norm->Activation->Conv->Norm->Activation
        """

        super(ConvNormNonlinBlock, self).__init__()

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.output_channels = output_channels

        self.first_conv = self.conv_op(input_channels, output_channels, **self.conv_kwargs)
        self.first_norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.first_acti = self.nonlin(**self.nonlin_kwargs)

        self.second_conv = self.conv_op(output_channels, output_channels, **self.conv_kwargs)
        self.second_norm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.second_acti = self.nonlin(**self.nonlin_kwargs)

        self.block = nn.Sequential(
            self.first_conv,
            self.first_norm,
            self.first_acti,
            self.second_conv,
            self.second_norm,
            self.second_acti
            )


    def forward(self, x):
        return self.block(x)



class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)


class U_Net2D5(nn.Module):

    def __init__(
        self,
        input_channels=1,
        base_num_features=16,
        num_classes=2,
        num_pool=4,
        conv_op=nn.Conv3d,
        conv_kernel_sizes=None,
        norm_op=nn.InstanceNorm3d,
        norm_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs=None,
        weightInitializer=InitWeights_He(1e-2)):
        """
        2.5D CNN combining 2D and 3D convolutions to dealwith the low through-plane resolution.
        The first two stages have 2D convolutions while the others have 3D convolutions.

        Architecture inspired by:
        Wang,et al: Automatic segmentation of  vestibular  schwannoma  from  t2-weighted  mri
        by  deep  spatial  attention  with hardness-weighted loss. MICCAI 2019.
        """
        super(U_Net2D5, self).__init__()


        if nonlin_kwargs is None:
             nonlin_kwargs = {'negative_slope':1e-2, 'inplace':True}

        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps':1e-5, 'affine':True, 'momentum':0.1}

        self.conv_kwargs = {'stride':1, 'dilation':1, 'bias':True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.num_classes = num_classes

        upsample_mode = 'trilinear'
        pool_op = nn.MaxPool3d
        pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
        if conv_kernel_sizes is None:
            conv_kernel_sizes = [(3, 3, 1)] * 2 + [(3,3,3)]*(num_pool - 1)


        self.input_shape_must_be_divisible_by = np.prod(pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []


        input_features = input_channels
        output_features = base_num_features


        for d in range(num_pool):
            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[d]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[d]
            # add convolutions
            self.conv_blocks_context.append(ConvNormNonlinBlock(input_features, output_features,
                                                                self.conv_op, self.conv_kwargs, self.norm_op,
                                                                self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))

            self.td.append(pool_op(pool_op_kernel_sizes[d]))
            input_features = output_features
            output_features = 2* output_features # Number of kernel increases by a factor 2 after each pooling


        final_num_features = self.conv_blocks_context[-1].output_channels
        self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[num_pool]
        self.conv_kwargs['padding'] = self.conv_pad_sizes[num_pool]
        self.conv_blocks_context.append(ConvNormNonlinBlock(input_features, final_num_features,
                                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                                            self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))


        # now lets build the localization pathway
        for u in range(num_pool):
            nfeatures_from_skip = self.conv_blocks_context[-(2 + u)].output_channels # self.conv_blocks_context[-1] is bottleneck, so start with -2
            n_features_after_tu_and_concat = nfeatures_from_skip * 2

            # the first conv reduces the number of features to match those of skip
            # the following convs work on that number of features
            # if not convolutional upsampling then the final conv reduces the num of features again
            if u != num_pool - 1:
                final_num_features = self.conv_blocks_context[-(3 + u)].output_channels
            else:
                final_num_features = nfeatures_from_skip

            self.tu.append(Upsample(scale_factor=pool_op_kernel_sizes[-(u+1)], mode=upsample_mode))


            self.conv_kwargs['kernel_size'] = self.conv_kernel_sizes[-(u+1)]
            self.conv_kwargs['padding'] = self.conv_pad_sizes[-(u+1)]
            self.conv_blocks_localization.append(ConvNormNonlinBlock(n_features_after_tu_and_concat, final_num_features,
                                                            self.conv_op, self.conv_kwargs, self.norm_op,
                                                            self.norm_op_kwargs, self.nonlin, self.nonlin_kwargs))


        self.final_conv = conv_op(self.conv_blocks_localization[-1].output_channels, num_classes, 1, 1, 0, 1, 1, False)



        # register all modules properly
        self.conv_blocks_localization = nn.ModuleList(self.conv_blocks_localization)
        self.conv_blocks_context = nn.ModuleList(self.conv_blocks_context)
        self.td = nn.ModuleList(self.td)
        self.tu = nn.ModuleList(self.tu)

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)

    def forward(self, x):
        skips = []
        seg_outputs = []
        for d in range(len(self.conv_blocks_context) - 1):
            x = self.conv_blocks_context[d](x)
            skips.append(x)
            x = self.td[d](x)

        x = self.conv_blocks_context[-1](x)

        for u in range(len(self.tu)):
            x = self.tu[u](x)
            x = torch.cat((x, skips[-(u + 1)]), dim=1)
            x = self.conv_blocks_localization[u](x)

        output = self.final_conv(x)
        return output


# --- [Original file: network/net_dict.py] ---
NetDict = {
    "U_Net": U_Net,
    "AttU_Net": AttU_Net
}

def get_network(network_type, input_channels, output_channels):
    if network_type == "U_Net2D5":
        network = U_Net2D5(input_channels=input_channels, num_classes=output_channels)
    elif network_type == "U_Net" or network_type == "AttU_Net":
        network = NetDict[network_type](img_ch=input_channels, output_ch=output_channels)
    else:
        raise RuntimeError
    return network


# --- [Original file: utilities/gated_crf_loss3d.py] ---
class ModelLossSemsegGatedCRF3D22D(torch.nn.Module):

    def forward(
            self, y_hat_softmax, kernels_desc, kernels_radius, sample, height_input, width_input, depth_input,
            mask_src=None, mask_dst=None, compatibility=None, custom_modality_downsamplers=None, out_kernels_vis=False
    ):
        assert len(kernels_radius) == 3
        assert y_hat_softmax.dim() == 5, 'Prediction must be a NCHWD batch'
        N, C, height_pred, width_pred, depth_pred = y_hat_softmax.shape
        C_input = sample.shape[1]
        device = y_hat_softmax.device

        assert height_input % height_pred == 0 and depth_input % depth_pred == 0 and width_input % width_pred == 0 \
            and depth_input * width_pred == width_input * depth_pred, \
            f'[{depth_input}x{width_input}] !~= [{depth_pred}x{width_pred}]'

        #reshape [N, C, H, W, D] tensor to [N*D, C, H, W]
        y_hat_softmax = y_hat_softmax.permute(0, 4, 1, 2, 3).reshape(N * depth_pred, C, height_pred, width_pred)
        sample = sample.permute(0, 4, 1, 2, 3).reshape(N * depth_input, C_input, height_input, width_input)

        kernels = self._create_kernels(
            kernels_desc, kernels_radius, sample, N * depth_pred, height_pred, width_pred, device, custom_modality_downsamplers
        )

        denom = N * depth_pred * height_pred * width_pred

        y_hat_unfolded = self._unfold(y_hat_softmax, kernels_radius)
        kernels_size = [x * 2 + 1 for x in kernels_radius]
        product_kernel_x_y_hat = (kernels * y_hat_unfolded) \
            .reshape(N * depth_pred, C, kernels_size[0] * kernels_size[1], -1).sum(dim=2, keepdim=False)
        product_kernel_x_y_hat = product_kernel_x_y_hat.reshape(N * depth_pred, C, height_pred, width_pred)

        if compatibility is None:
            # Using shortcut for Pott's class compatibility model
            loss = -(product_kernel_x_y_hat * y_hat_softmax).sum()
            # comment out to save computation, total loss may go below 0
            loss = kernels.sum() + loss
        else:
            raise ValueError

        out = {
            'loss': loss / denom,
        }

        return out

    @staticmethod
    def _downsample(img, modality, height_dst, width_dst, custom_modality_downsamplers):
        if custom_modality_downsamplers is not None and modality in custom_modality_downsamplers:
            f_down = custom_modality_downsamplers[modality]
        else:
            f_down = F.adaptive_avg_pool2d
        return f_down(img, (height_dst, width_dst))

    @staticmethod
    def _create_kernels(
            kernels_desc, kernels_radius, sample, N, height_pred, width_pred, device, custom_modality_downsamplers
    ):
        kernels = None
        for i, desc in enumerate(kernels_desc):
            weight = desc['weight']
            features = []
            for modality, sigma in desc.items():
                if modality == 'weight':
                    continue
                if modality == 'xy':
                    feature = ModelLossSemsegGatedCRF3D22D._get_mesh(
                        N, height_pred, width_pred, device)
                else:
                    # assert modality in sample, 'Modality {} is listed in {}-th kernel descriptor, but not present in the sample'.format(modality, i)
                    feature = sample
                    feature = ModelLossSemsegGatedCRF3D22D._downsample(
                        feature, modality, height_pred, width_pred, custom_modality_downsamplers
                    )
                feature /= sigma
                features.append(feature)
            features = torch.cat(features, dim=1)
            kernel = weight * \
                ModelLossSemsegGatedCRF3D22D._create_kernels_from_features(
                    features, kernels_radius)
            kernels = kernel if kernels is None else kernel + kernels
        return kernels

    @staticmethod
    def _create_kernels_from_features(features, radius):
        N, C, H, W = features.shape
        kernels = ModelLossSemsegGatedCRF3D22D._unfold(features, radius)
        kernels = kernels - kernels[:, :, radius[0],
                                    radius[1], :].reshape(N, C, 1, 1, -1)
        kernels = (-0.5 * kernels ** 2).sum(dim=1, keepdim=True).exp()
        kernels[:, :, radius[0], radius[1], :] = 0
        return kernels

    @staticmethod
    def _get_mesh(N, H, W, device):
        return torch.cat((
            torch.arange(0, H, 1, dtype=torch.float32, device=device).reshape(
                1, 1, H, 1).repeat(N, 1, 1, W),
            torch.arange(0, W, 1, dtype=torch.float32, device=device).reshape(
                1, 1, 1, W).repeat(N, 1, H, 1),
        ), 1)

    @staticmethod
    def _unfold(img, radius):
        N, C, H, W = img.shape
        diameter = [x * 2 + 1 for x in radius]
        return unfoldNd.unfoldNd(img, kernel_size=(diameter[0], diameter[1]), dilation=1, padding=(radius[0], radius[1]), \
            stride=1).reshape(N, C, diameter[0], diameter[1], -1)


# --- [Original file: utilities/losses.py] ---
class VarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, output, input):
        # output [N, C, H, W, D], input [N, 1, H, W, D]
        soft_output = torch.softmax(output, dim=1)
        loss = 0.0
        for b in range(0, output.shape[0]):
            sub_input = input[b, 0, :, :, :]
            # mask out minimum value voxels
            mask = ((sub_input - torch.min(sub_input)) > 1e-5).float()
            # mask = 1.0
            for c in range(0, output.shape[1]):
                sub_soft_output = soft_output[b, c, :, :, :]
                if c == 0:
                    nums = torch.sum(sub_soft_output * mask)
                    mean = torch.sum(sub_input * sub_soft_output * mask) / nums
                    variance = torch.sum((sub_input - mean) ** 2 *sub_soft_output * mask) / nums
                else:
                    nums = torch.sum(sub_soft_output)
                    mean = torch.sum(sub_input * sub_soft_output) / nums
                    variance = torch.sum((sub_input - mean) ** 2 *sub_soft_output) / nums
                loss += variance
        loss = loss / output.shape[0]
        return loss

class KDLoss(nn.Module):
    '''
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    '''
    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = nn.functional.kl_div(nn.functional.log_softmax(out_s/self.T, dim=1),
                        nn.functional.softmax(out_t/self.T, dim=1),
                        reduction='batchmean') * self.T * self.T
        return loss


# ============================================================
# __main__: Automated test suite for 7 ablated functions
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

    device = torch.device("cpu")

    print("=" * 70)
    print("PA-Seg: Core Architecture and Regularization Benchmark")
    print("Automated Test Suite - 7 ablated functions")
    print("=" * 70)
    print(f"Device: {device}")
    print()

    print("-" * 60)
    print("[Test 1/7] U_Net2D5.forward - 2.5D encoder-decoder with skip restoration")
    try:
        model = U_Net2D5(input_channels=1, base_num_features=4, num_classes=2, num_pool=2).to(device)
        model.eval()
        x = torch.randn(1, 1, 16, 16, 16, device=device, requires_grad=True)
        y = model(x)
        check("U_Net2D5 output not None", y is not None)
        if y is not None:
            check("U_Net2D5 output shape", y.shape == (1, 2, 16, 16, 16), f"got {tuple(y.shape)}")
            check("U_Net2D5 output finite", torch.isfinite(y).all().item())
            y.sum().backward()
            check("U_Net2D5 input gradient", x.grad is not None and torch.isfinite(x.grad).all().item())
        else:
            skip_checks(3, "U_Net2D5.forward returned None")
    except Exception as exc:
        skip_checks(4, f"U_Net2D5.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/7] Attention_block.forward - gated skip modulation")
    try:
        block = Attention_block(F_g=4, F_l=4, F_int=2).to(device)
        block.eval()
        g = torch.randn(2, 4, 4, 4, 4, device=device, requires_grad=True)
        x = torch.ones(2, 4, 4, 4, 4, device=device, requires_grad=True)
        y = block(g, x)
        check("Attention_block output not None", y is not None)
        if y is not None:
            check("Attention_block output shape", y.shape == (2, 4, 4, 4, 4), f"got {tuple(y.shape)}")
            check("Attention_block output finite", torch.isfinite(y).all().item())
            check("Attention_block gates bounded", y.min().item() >= 0.0 and y.max().item() <= 1.0)
            y.sum().backward()
            check("Attention_block gradients", g.grad is not None and x.grad is not None)
        else:
            skip_checks(4, "Attention_block.forward returned None")
    except Exception as exc:
        skip_checks(5, f"Attention_block.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/7] AttU_Net.forward - attention gates participate in decoding")
    try:
        model = AttU_Net(img_ch=1, output_ch=2).to(device)
        model.eval()
        calls = {"Att2": 0, "Att3": 0, "Att4": 0, "Att5": 0}
        handles = []
        for name in calls:
            handles.append(getattr(model, name).register_forward_hook(
                lambda module, inputs, output, key=name: calls.__setitem__(key, calls[key] + 1)))
        x = torch.randn(1, 1, 32, 32, 32, device=device)
        with torch.no_grad():
            y = model(x)
        for handle in handles:
            handle.remove()
        check("AttU_Net output not None", y is not None)
        if y is not None:
            check("AttU_Net output shape", y.shape == (1, 2, 32, 32, 32), f"got {tuple(y.shape)}")
            check("AttU_Net output finite", torch.isfinite(y).all().item())
            check("AttU_Net all attention gates called", all(v == 1 for v in calls.values()), str(calls))
        else:
            skip_checks(3, "AttU_Net.forward returned None")
    except Exception as exc:
        skip_checks(4, f"AttU_Net.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/7] ModelLossSemsegGatedCRF3D22D._create_kernels_from_features - local gated kernels")
    try:
        features = torch.arange(1 * 2 * 4 * 4, dtype=torch.float32, device=device).reshape(1, 2, 4, 4)
        kernels = ModelLossSemsegGatedCRF3D22D._create_kernels_from_features(features, [1, 1, 0])
        check("CRF kernels not None", kernels is not None)
        if kernels is not None:
            check("CRF kernels shape", kernels.shape == (1, 1, 3, 3, 16), f"got {tuple(kernels.shape)}")
            check("CRF kernels finite", torch.isfinite(kernels).all().item())
            check("CRF kernels nonnegative", (kernels >= 0).all().item())
            check("CRF center kernel zero", torch.allclose(kernels[:, :, 1, 1, :], torch.zeros_like(kernels[:, :, 1, 1, :])))
        else:
            skip_checks(4, "kernel creation returned None")
    except Exception as exc:
        skip_checks(5, f"kernel creation raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 5/7] ModelLossSemsegGatedCRF3D22D.forward - 3D-to-2D gated CRF loss")
    try:
        loss_fn = ModelLossSemsegGatedCRF3D22D().to(device)
        logits = torch.randn(1, 2, 4, 4, 2, device=device, requires_grad=True)
        probs = torch.softmax(logits, dim=1)
        sample = torch.randn(1, 1, 4, 4, 2, device=device)
        out = loss_fn(probs, [{"weight": 1.0, "xy": 2.0, "rgb": 0.5}], [1, 1, 0], sample, 4, 4, 2)
        check("CRF forward output not None", out is not None)
        if out is not None:
            check("CRF forward returns dict", isinstance(out, dict) and "loss" in out)
            loss = out["loss"] if isinstance(out, dict) and "loss" in out else None
            check("CRF loss scalar", loss is not None and loss.dim() == 0)
            check("CRF loss finite", loss is not None and torch.isfinite(loss).item())
            if loss is not None:
                loss.backward()
                check("CRF loss gradient", logits.grad is not None and torch.isfinite(logits.grad).all().item())
            else:
                skip_checks(1, "CRF loss missing")
        else:
            skip_checks(4, "CRF forward returned None")
    except Exception as exc:
        skip_checks(5, f"CRF forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 6/7] VarianceLoss.forward - probability-weighted intensity variance")
    try:
        loss_fn = VarianceLoss().to(device)
        logits = torch.randn(1, 2, 4, 4, 4, device=device, requires_grad=True)
        image = torch.linspace(0.0, 1.0, 64, device=device).reshape(1, 1, 4, 4, 4)
        loss = loss_fn(logits, image)
        check("VarianceLoss output not None", loss is not None)
        if loss is not None:
            check("VarianceLoss scalar", loss.dim() == 0)
            check("VarianceLoss finite", torch.isfinite(loss).item())
            check("VarianceLoss nonnegative", loss.item() >= 0.0)
            loss.backward()
            check("VarianceLoss gradient", logits.grad is not None and torch.isfinite(logits.grad).all().item())
            near_constant = torch.ones(1, 1, 4, 4, 4, device=device)
            near_constant[:, :, 0, 0, 0] = 0.0
            background_logits = torch.zeros(1, 1, 4, 4, 4, device=device)
            zero_loss = loss_fn(background_logits, near_constant)
            check("VarianceLoss zero on valid constant background support", torch.allclose(zero_loss, torch.zeros_like(zero_loss), atol=1e-5))
        else:
            skip_checks(5, "VarianceLoss.forward returned None")
    except Exception as exc:
        skip_checks(6, f"VarianceLoss.forward raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 7/7] KDLoss.forward - temperature-scaled cross knowledge distillation")
    try:
        loss_fn = KDLoss(T=4.0).to(device)
        student = torch.randn(8, 2, device=device, requires_grad=True)
        teacher = torch.randn(8, 2, device=device)
        loss = loss_fn(student, teacher)
        check("KDLoss output not None", loss is not None)
        if loss is not None:
            check("KDLoss scalar", loss.dim() == 0)
            check("KDLoss finite", torch.isfinite(loss).item())
            check("KDLoss nonnegative", loss.item() >= -1e-6)
            same_loss = loss_fn(student, student.detach())
            check("KDLoss zero for identical logits", torch.allclose(same_loss, torch.zeros_like(same_loss), atol=1e-5))
            loss.backward()
            check("KDLoss gradient", student.grad is not None and torch.isfinite(student.grad).all().item())
        else:
            skip_checks(5, "KDLoss.forward returned None")
    except Exception as exc:
        skip_checks(6, f"KDLoss.forward raised {type(exc).__name__}: {exc}")
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
