"""
ground_truth.py for DSR anomaly detection core model components.

Source-consolidated from:
- discrete_model.py
- dsr_model.py

Only neural network architecture components are included. Optimization loops,
data loaders, score computation, CLI wrappers, saved-weight I/O, and experiment
orchestration are intentionally excluded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    # Source for the VectorQuantizerEMA module: https://github.com/zalandoresearch/pytorch-vq-vae
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def get_quantized(self, inputs):
        """
        TODO: Quantize a BCHW feature map with the learned embedding table.

        Input:
            inputs: tensor of shape (batch, embedding_dim, height, width).
        Output:
            quantized tensor with the same BCHW shape.

        Compare every spatial feature vector with all codebook entries, select
        the nearest entry per location, rebuild the quantized feature map, and
        preserve the straight-through estimator behavior so gradients follow the
        input path while values come from the selected embeddings.
        """
        pass

    def forward(self, inputs):
        """
        TODO: Run EMA vector quantization for a BCHW feature tensor.

        Input:
            inputs: tensor of shape (batch, embedding_dim, height, width).
        Output:
            tuple (loss, quantized, perplexity, encodings), where loss and
            perplexity are scalars, quantized has the same BCHW shape as the
            input, and encodings is a one-hot assignment matrix with one row per
            spatial feature vector.

        Preserve the source semantics: nearest-code assignment in embedding
        space, optional exponential-moving-average codebook update in training
        mode with cluster smoothing, commitment loss against the unquantized
        features, straight-through quantized output, and perplexity computed from
        average code usage.
        """
        pass


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class EncoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)
        return self._residual_stack(x)


class EncoderTop(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(EncoderTop, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._residual_stack(x)
        return x


class DecoderBot(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(DecoderBot, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class DiscreteLatentModel(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                 commitment_cost, decay=0, test=False):
        # def __init__(self, num_embeddings=512, embedding_dim=128, commitment_cost=0.25, decay=0):
        super(DiscreteLatentModel, self).__init__()
        self.test = test
        self._encoder_t = EncoderTop(num_hiddens, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._encoder_b = EncoderBot(3, num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)

        self._pre_vq_conv_bot = nn.Conv2d(in_channels=num_hiddens + embedding_dim,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1)

        self._pre_vq_conv_top = nn.Conv2d(in_channels=num_hiddens,
                                          out_channels=embedding_dim,
                                          kernel_size=1,
                                          stride=1)

        self._vq_vae_top = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._vq_vae_bot = VectorQuantizerEMA(num_embeddings, embedding_dim,
                                              commitment_cost, decay)

        self._decoder_b = DecoderBot(embedding_dim*2,
                                     num_hiddens,
                                     num_residual_layers,
                                     num_residual_hiddens)


        self.upsample_t = nn.ConvTranspose2d(
            embedding_dim, embedding_dim, 4, stride=2, padding=1
        )


    def forward(self, x):
        """
        TODO: Run the DSR discrete latent model with dual top/bottom quantization.

        Input:
            x: image tensor of shape (batch, 3, height, width).
        Output:
            tuple (bottom_loss, top_loss, reconstruction, top_quantized,
            bottom_quantized). For a 64x64 input with embedding_dim=8, the
            top quantized tensor is (batch, 8, 8, 8), the bottom quantized tensor
            is (batch, 8, 16, 16), and reconstruction is
            (batch, 3, 64, 64).

        Encode the image into high- and low-level feature subspaces, quantize the
        low-level subspace first, upsample it to align with the high-level
        feature grid, build and quantize the high-level subspace, then decode the
        joined quantized representation back into an image. Preserve the order of
        returned losses and quantized tensors used by the rest of DSR.
        """
        pass


class FeatureEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 4),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        TODO: Encode a feature map for the subspace restriction network.

        Input:
            x: tensor of shape (batch, in_channels, height, width).
        Output:
            tuple of three feature tensors at full, half, and quarter spatial
            resolution with channels base_width, 2 * base_width, and
            4 * base_width.

        Apply the restriction encoder blocks and pooling stages in source order.
        The returned multi-scale features must preserve their spatial alignment
        for the paired restriction decoder.
        """
        pass


class FeatureDecoder(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super().__init__()

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 nn.InstanceNorm2d(base_width * 2),
                                 nn.ReLU(inplace=True))

        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 nn.InstanceNorm2d(base_width),
                                 nn.ReLU(inplace=True))
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            nn.InstanceNorm2d(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3):
        """
        TODO: Decode subspace restriction features back to the original feature grid.

        Inputs:
            b1, b2, b3: feature tensors from FeatureEncoder, ordered from full
            resolution to deepest quarter-resolution representation.
        Output:
            tensor of shape (batch, out_channels, height, width).

        Starting from the deepest representation, perform the two upsampling
        stages and decoder blocks used by the source restriction network. This
        decoder does not concatenate encoder skip tensors even though those
        tensors are passed through the interface.
        """
        pass


class SubspaceRestrictionNetwork(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.base_width = base_width
        self.encoder = FeatureEncoder(in_channels, self.base_width)
        self.decoder = FeatureDecoder(self.base_width, out_channels=out_channels)

    def forward(self, x):
        """
        TODO: Run the subspace restriction U-Net-like network.

        Input:
            x: feature tensor of shape (batch, in_channels, height, width).
        Output:
            restricted feature tensor of shape
            (batch, out_channels, height, width).

        Encode the input into the three-scale feature tuple, then decode the
        tuple through the paired restriction decoder while preserving the source
        feature ordering.
        """
        pass


class SubspaceRestrictionModule(nn.Module):
    def __init__(self, embedding_size=64):
        super(SubspaceRestrictionModule, self).__init__()

        base_width = embedding_size
        self.unet = SubspaceRestrictionNetwork(in_channels=base_width, out_channels=base_width, base_width=embedding_size)

    def forward(self, x, quantization):
        """
        TODO: Restore a feature tensor to the learned normal subspace and re-quantize it.

        Inputs:
            x: anomalous feature tensor of shape
               (batch, embedding_size, height, width).
            quantization: callable vector quantizer accepting the restored
               feature tensor.
        Output:
            tuple (restored_features, requantized_features, quantization_loss),
            where the first two tensors have the same shape as x.

        Pass the feature tensor through the restriction network, quantize the
        restored feature map with the provided quantizer, and return the restored
        continuous features, the quantized features, and the quantizer loss in
        the exact order expected by DSR.
        """
        pass


class ImageReconstructionNetwork(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ImageReconstructionNetwork, self).__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels*2, kernel_size=3, padding=1),
            norm_layer(in_channels*2),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels*2, in_channels * 2, kernel_size=3, padding=1),
            norm_layer(in_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2, in_channels * 4, kernel_size=3, padding=1),
            norm_layer(in_channels * 4),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))

        self.pre_vq_conv = nn.Conv2d(in_channels=in_channels*4,
                                 out_channels=64,
                                 kernel_size=1,
                                 stride=1)



        #self.vq = VectorQuantizerEMA(512, 64, 0.25, 0.99)

        self.upblock1 = nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self.upblock2 = nn.ConvTranspose2d(in_channels=64,
                                                out_channels=64,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_1 = nn.Conv2d(in_channels=64,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        """
        TODO: Reconstruct an RGB image from joined quantized DSR feature maps.

        Input:
            inputs: joined feature tensor of shape
            (batch, in_channels, height, width), typically the aligned top and
            bottom quantized subspaces.
        Output:
            RGB reconstruction tensor of shape
            (batch, 3, 4 * height, 4 * width).

        Follow the source object-specific reconstruction path: downsample and
        project the joined features, upsample back to the feature grid, refine
        with the residual stack, then use the two transposed-convolution stages
        to produce the final RGB image.
        """
        pass


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, base_width):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True))
        self.mp1 = nn.Sequential(nn.MaxPool2d(2))
        self.block2 = nn.Sequential(
            nn.Conv2d(base_width, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True))
        self.mp2 = nn.Sequential(nn.MaxPool2d(2))
        self.block3 = nn.Sequential(
            nn.Conv2d(base_width * 2, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True))
        self.mp3 = nn.Sequential(nn.MaxPool2d(2))
        self.block4 = nn.Sequential(
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True))

    def forward(self, x):
        """
        TODO: Encode an image-like tensor into four U-Net skip features.

        Input:
            x: tensor of shape (batch, in_channels, height, width).
        Output:
            tuple (b1, b2, b3, b4), where spatial resolution is full, half,
            quarter, and one-eighth resolution, and channels follow
            base_width, 2 * base_width, 4 * base_width, 4 * base_width.

        Apply the source encoder blocks and pooling stages in order and return
        every skip tensor required by the decoder. Preserving both order and
        spatial scale is necessary for decoder concatenation.
        """
        pass


class UnetDecoder(nn.Module):
    def __init__(self, base_width, out_channels=1):
        super().__init__()
        norm_layer = nn.InstanceNorm2d
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
                                 norm_layer(base_width * 4),
                                 nn.ReLU(inplace=True))
        # cat with base*4
        self.db1 = nn.Sequential(
            nn.Conv2d(base_width * (4 + 4), base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 4, base_width * 4, kernel_size=3, padding=1),
            norm_layer(base_width * 4),
            nn.ReLU(inplace=True)
        )

        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 4, base_width * 2, kernel_size=3, padding=1),
                                 norm_layer(base_width * 2),
                                 nn.ReLU(inplace=True))
        # cat with base*2
        self.db2 = nn.Sequential(
            nn.Conv2d(base_width * (2 + 2), base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width * 2, base_width * 2, kernel_size=3, padding=1),
            norm_layer(base_width * 2),
            nn.ReLU(inplace=True)
        )

        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                 nn.Conv2d(base_width * 2, base_width, kernel_size=3, padding=1),
                                 norm_layer(base_width),
                                 nn.ReLU(inplace=True))
        # cat with base*1
        self.db3 = nn.Sequential(
            nn.Conv2d(base_width * (1 + 1), base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_width, base_width, kernel_size=3, padding=1),
            norm_layer(base_width),
            nn.ReLU(inplace=True)
        )

        self.fin_out = nn.Sequential(nn.Conv2d(base_width, out_channels, kernel_size=3, padding=1))

    def forward(self, b1, b2, b3, b4):
        """
        TODO: Decode U-Net skip features into dense logits/features.

        Inputs:
            b1, b2, b3, b4: encoder features ordered from full resolution to the
            deepest one-eighth-resolution feature. For a 32x32 input and
            base_width=8, their shapes are (batch, 8, 32, 32),
            (batch, 16, 16, 16), (batch, 32, 8, 8), and (batch, 32, 4, 4).
        Output:
            tensor of shape (batch, out_channels, height, width).

        Decode from the deepest feature, concatenate each upsampled feature with
        the matching encoder skip along the channel dimension, apply the paired
        decoder block, and finish with the output projection.
        """
        pass


class UnetModel(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def forward(self, x):
        """
        TODO: Run the shared U-Net model used by DSR segmentation modules.

        Input:
            x: tensor of shape (batch, in_channels, height, width).
        Output:
            tensor of shape (batch, out_channels, height, width).

        Preserve the source encoder-decoder composition: collect all four
        encoder skip features and pass them to the decoder in the same order.
        """
        pass


class AnomalyDetectionModule(nn.Module):
    def __init__(self, embedding_size=64):
        super(AnomalyDetectionModule, self).__init__()
        self.unet = UnetModel(in_channels=6, out_channels=2, base_width=64)
    def forward(self, image_real, image_anomaly):
        """
        TODO: Predict anomaly logits from reconstructed-normal and general images.

        Inputs:
            image_real: tensor of shape (batch, 3, height, width).
            image_anomaly: tensor of shape (batch, 3, height, width).
        Output:
            anomaly logits of shape (batch, 2, height, width).

        Combine the two RGB images along the channel dimension to create the
        six-channel segmentation input, then run the internal U-Net.
        """
        pass


class UpsamplingModule(nn.Module):
    def __init__(self, embedding_size=64):
        super(UpsamplingModule, self).__init__()
        self.unet = UnetModel(in_channels=8, out_channels=2, base_width=64)
        #self.unet = UNetNormalSkip(in_channels=4 * embedding_size + 16, out_channels=embedding_size)
    def forward(self, image_real, image_anomaly, segmentation_map):
        """
        TODO: Refine/upscale anomaly logits using images and a coarse segmentation map.

        Inputs:
            image_real: tensor of shape (batch, 3, height, width).
            image_anomaly: tensor of shape (batch, 3, height, width).
            segmentation_map: tensor of shape (batch, 2, height, width).
        Output:
            refined logits of shape (batch, 2, height, width).

        Combine the two RGB images and the two-channel segmentation map into the
        eight-channel input expected by the refinement U-Net, then return its
        logits.
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

    print("Running DSR anomaly detection core model benchmark checks...")
    device = torch.device("cpu")

    try:
        vq = VectorQuantizerEMA(num_embeddings=16, embedding_dim=8, commitment_cost=0.25, decay=0.99).to(device)
        vq.eval()
        x = torch.randn(2, 8, 8, 8, device=device)
        with torch.no_grad():
            loss, quantized, perplexity, encodings = vq(x)
            quantized_direct = vq.get_quantized(x)
        check("vq forward loss not None", loss is not None)
        check("vq quantized shape", tuple(quantized.shape) == (2, 8, 8, 8), f"got {tuple(quantized.shape)}")
        check("vq quantized finite", torch.isfinite(quantized).all().item())
        check("vq perplexity scalar finite", perplexity.ndim == 0 and torch.isfinite(perplexity).item())
        check("vq encodings shape", tuple(encodings.shape) == (2 * 8 * 8, 16), f"got {tuple(encodings.shape)}")
        check("vq encodings one hot", torch.allclose(encodings.sum(dim=1), torch.ones(encodings.shape[0], device=device)))
        check("vq get_quantized shape", tuple(quantized_direct.shape) == (2, 8, 8, 8), f"got {tuple(quantized_direct.shape)}")
    except Exception as exc:
        skip_checks(7, f"VectorQuantizerEMA checks raised {type(exc).__name__}: {exc}")

    try:
        model = DiscreteLatentModel(
            num_hiddens=16,
            num_residual_layers=1,
            num_residual_hiddens=8,
            num_embeddings=32,
            embedding_dim=8,
            commitment_cost=0.25,
            decay=0.99,
        ).to(device)
        model.eval()
        image = torch.randn(2, 3, 64, 64, device=device)
        with torch.no_grad():
            loss_b, loss_t, recon, quantized_t, quantized_b = model(image)
        check("discrete outputs not None", all(v is not None for v in [loss_b, loss_t, recon, quantized_t, quantized_b]))
        check("discrete recon shape", tuple(recon.shape) == (2, 3, 64, 64), f"got {tuple(recon.shape)}")
        check("discrete top quantized shape", tuple(quantized_t.shape) == (2, 8, 8, 8), f"got {tuple(quantized_t.shape)}")
        check("discrete bottom quantized shape", tuple(quantized_b.shape) == (2, 8, 16, 16), f"got {tuple(quantized_b.shape)}")
        check("discrete outputs finite", all(torch.isfinite(v).all().item() for v in [loss_b, loss_t, recon, quantized_t, quantized_b]))
        check("discrete upsample alignment", tuple(model.upsample_t(quantized_t).shape[-2:]) == tuple(quantized_b.shape[-2:]))
    except Exception as exc:
        skip_checks(6, f"DiscreteLatentModel checks raised {type(exc).__name__}: {exc}")

    try:
        quantizer = VectorQuantizerEMA(num_embeddings=16, embedding_dim=8, commitment_cost=0.25, decay=0.99).to(device)
        quantizer.eval()
        restriction = SubspaceRestrictionModule(embedding_size=8).to(device)
        restriction.eval()
        features = torch.randn(2, 8, 16, 16, device=device)
        with torch.no_grad():
            restored, requantized, loss = restriction(features, quantizer)
        check("subspace outputs not None", all(v is not None for v in [restored, requantized, loss]))
        check("subspace restored shape", tuple(restored.shape) == (2, 8, 16, 16), f"got {tuple(restored.shape)}")
        check("subspace requantized shape", tuple(requantized.shape) == (2, 8, 16, 16), f"got {tuple(requantized.shape)}")
        check("subspace finite", all(torch.isfinite(v).all().item() for v in [restored, requantized, loss]))
        check("subspace changed by network", not torch.allclose(restored, features))
    except Exception as exc:
        skip_checks(5, f"SubspaceRestrictionModule checks raised {type(exc).__name__}: {exc}")

    try:
        encoder = FeatureEncoder(in_channels=8, base_width=8).to(device)
        decoder = FeatureDecoder(base_width=8, out_channels=8).to(device)
        encoder.eval()
        decoder.eval()
        features = torch.randn(2, 8, 16, 16, device=device)
        with torch.no_grad():
            b1, b2, b3 = encoder(features)
            decoded = decoder(b1, b2, b3)
        check("feature encoder shapes", [tuple(t.shape) for t in [b1, b2, b3]] == [(2, 8, 16, 16), (2, 16, 8, 8), (2, 32, 4, 4)], f"got {[tuple(t.shape) for t in [b1, b2, b3]]}")
        check("feature encoder finite", all(torch.isfinite(t).all().item() for t in [b1, b2, b3]))
        check("feature decoder output shape", tuple(decoded.shape) == (2, 8, 16, 16), f"got {tuple(decoded.shape)}")
        check("feature decoder finite", torch.isfinite(decoded).all().item())
    except Exception as exc:
        skip_checks(4, f"FeatureEncoder/FeatureDecoder checks raised {type(exc).__name__}: {exc}")

    try:
        recon_net = ImageReconstructionNetwork(
            in_channels=16,
            num_hiddens=16,
            num_residual_layers=1,
            num_residual_hiddens=8,
        ).to(device)
        recon_net.eval()
        quant_join = torch.randn(2, 16, 16, 16, device=device)
        with torch.no_grad():
            image_recon = recon_net(quant_join)
        check("image reconstruction not None", image_recon is not None)
        if image_recon is not None:
            check("image reconstruction shape", tuple(image_recon.shape) == (2, 3, 64, 64), f"got {tuple(image_recon.shape)}")
            check("image reconstruction finite", torch.isfinite(image_recon).all().item())
            check("image reconstruction upsamples spatially", image_recon.shape[-1] == quant_join.shape[-1] * 4)
        else:
            skip_checks(3, "ImageReconstructionNetwork returned None")
    except Exception as exc:
        skip_checks(4, f"ImageReconstructionNetwork checks raised {type(exc).__name__}: {exc}")

    try:
        unet = UnetModel(in_channels=6, out_channels=2, base_width=8).to(device)
        unet.eval()
        x = torch.randn(2, 6, 32, 32, device=device)
        with torch.no_grad():
            logits = unet(x)
            enc_feats = unet.encoder(x)
        check("unet output shape", tuple(logits.shape) == (2, 2, 32, 32), f"got {tuple(logits.shape)}")
        check("unet output finite", torch.isfinite(logits).all().item())
        check("unet encoder returns four skips", isinstance(enc_feats, tuple) and len(enc_feats) == 4)
        if isinstance(enc_feats, tuple) and len(enc_feats) == 4:
            expected = [(2, 8, 32, 32), (2, 16, 16, 16), (2, 32, 8, 8), (2, 32, 4, 4)]
            check("unet skip feature shapes", [tuple(t.shape) for t in enc_feats] == expected, f"got {[tuple(t.shape) for t in enc_feats]}")
        else:
            skip_checks(1, "UnetEncoder did not return four skip tensors")

        anomaly_module = AnomalyDetectionModule(embedding_size=64).to(device)
        upsample_module = UpsamplingModule(embedding_size=64).to(device)
        anomaly_module.eval()
        upsample_module.eval()
        image_real = torch.randn(1, 3, 32, 32, device=device)
        image_anomaly = torch.randn(1, 3, 32, 32, device=device)
        segmentation_map = torch.randn(1, 2, 32, 32, device=device)
        with torch.no_grad():
            anomaly_logits = anomaly_module(image_real, image_anomaly)
            upsample_logits = upsample_module(image_real, image_anomaly, segmentation_map)
        check("anomaly module logits shape", tuple(anomaly_logits.shape) == (1, 2, 32, 32), f"got {tuple(anomaly_logits.shape)}")
        check("upsampling module logits shape", tuple(upsample_logits.shape) == (1, 2, 32, 32), f"got {tuple(upsample_logits.shape)}")
        check("segmentation logits finite", torch.isfinite(anomaly_logits).all().item() and torch.isfinite(upsample_logits).all().item())
        probs = torch.softmax(upsample_logits, dim=1)
        check("upsampling softmax channel sum", torch.allclose(probs.sum(dim=1), torch.ones_like(probs[:, 0]), atol=1e-5))
    except Exception as exc:
        skip_checks(9, f"Unet/anomaly/upsampling checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
