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
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        quantized = inputs + (quantized - inputs).detach()

        return quantized.permute(0, 3, 1, 2).contiguous()

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input ** 2, dim=1, keepdim=True)
                     + torch.sum(self._embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                    (self._ema_cluster_size + self._epsilon)
                    / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings


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
        #Encoder Hi
        enc_b = self._encoder_b(x)

        #Encoder Lo -- F_Lo
        enc_t = self._encoder_t(enc_b)
        zt = self._pre_vq_conv_top(enc_t)

        # Quantize F_Lo with K_Lo
        loss_t, quantized_t, perplexity_t, encodings_t = self._vq_vae_top(zt)
        # Upsample Q_Lo
        up_quantized_t = self.upsample_t(quantized_t)

        # Concatenate and transform the output of Encoder_Hi and upsampled Q_lo -- F_Hi
        feat = torch.cat((enc_b, up_quantized_t), dim=1)
        zb = self._pre_vq_conv_bot(feat)

        # Quantize F_Hi with K_Hi
        loss_b, quantized_b, perplexity_b, encodings_b = self._vq_vae_bot(zb)

        # Concatenate Q_Hi and Q_Lo and input it into the General appearance decoder
        quant_join = torch.cat((up_quantized_t, quantized_b), dim=1)
        recon_fin = self._decoder_b(quant_join)

        #return loss_b, loss_t, recon_fin, encodings_t, encodings_b, quantized_t, quantized_b
        return loss_b, loss_t, recon_fin, quantized_t, quantized_b


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
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        return b1, b2, b3


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
        up2 = self.up2(b3)
        db2 = self.db2(up2)

        up3 = self.up3(db2)
        db3 = self.db3(up3)

        out = self.fin_out(db3)
        return out


class SubspaceRestrictionNetwork(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.base_width = base_width
        self.encoder = FeatureEncoder(in_channels, self.base_width)
        self.decoder = FeatureDecoder(self.base_width, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3 = self.encoder(x)
        output = self.decoder(b1, b2, b3)
        return output


class SubspaceRestrictionModule(nn.Module):
    def __init__(self, embedding_size=64):
        super(SubspaceRestrictionModule, self).__init__()

        base_width = embedding_size
        self.unet = SubspaceRestrictionNetwork(in_channels=base_width, out_channels=base_width, base_width=embedding_size)

    def forward(self, x, quantization):
        x = self.unet(x)
        loss_b, quantized_b, perplexity_b, encodings_b = quantization(x)
        return x, quantized_b, loss_b


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
        x = self.block1(inputs)
        x = self.mp1(x)
        x = self.block2(x)
        x = self.mp2(x)
        x = self.pre_vq_conv(x)

        x = self.upblock1(x)
        x = F.relu(x)
        x = self.upblock2(x)
        x = F.relu(x)
        x = self._conv_1(x)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


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
        b1 = self.block1(x)
        mp1 = self.mp1(b1)
        b2 = self.block2(mp1)
        mp2 = self.mp2(b2)
        b3 = self.block3(mp2)
        mp3 = self.mp3(b3)
        b4 = self.block4(mp3)
        return b1, b2, b3, b4


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

        up1 = self.up1(b4)
        cat1 = torch.cat((up1, b3), dim=1)
        db1 = self.db1(cat1)

        up2 = self.up2(db1)
        cat2 = torch.cat((up2, b2), dim=1)
        db2 = self.db2(cat2)

        up3 = self.up3(db2)
        cat3 = torch.cat((up3, b1), dim=1)
        db3 = self.db3(cat3)

        out = self.fin_out(db3)
        return out


class UnetModel(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, base_width=64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels, base_width)
        self.decoder = UnetDecoder(base_width, out_channels=out_channels)

    def forward(self, x):
        b1, b2, b3, b4 = self.encoder(x)
        output = self.decoder(b1, b2, b3, b4)
        return output


class AnomalyDetectionModule(nn.Module):
    def __init__(self, embedding_size=64):
        super(AnomalyDetectionModule, self).__init__()
        self.unet = UnetModel(in_channels=6, out_channels=2, base_width=64)
    def forward(self, image_real, image_anomaly):
        img_x = torch.cat((image_real, image_anomaly),dim=1)
        x = self.unet(img_x)
        return x


class UpsamplingModule(nn.Module):
    def __init__(self, embedding_size=64):
        super(UpsamplingModule, self).__init__()
        self.unet = UnetModel(in_channels=8, out_channels=2, base_width=64)
        #self.unet = UNetNormalSkip(in_channels=4 * embedding_size + 16, out_channels=embedding_size)
    def forward(self, image_real, image_anomaly, segmentation_map):
        img_x = torch.cat((image_real, image_anomaly, segmentation_map),dim=1)
        x = self.unet(img_x)
        return x


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
