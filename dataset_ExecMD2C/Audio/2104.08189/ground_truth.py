import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: utils.py] ---
def get_mask_from_lengths(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device, dtype=torch.long)
    mask = (ids < lengths.unsqueeze(1)).bool()
    return mask


def get_mask(lengths, max_len=None):
    if not max_len:
        max_len = torch.max(lengths).item()
    lens = torch.arange(max_len)
    mask = lens[:max_len].unsqueeze(0) < lengths.unsqueeze(1)
    return mask


# --- [Original file: embedding.py] ---
def merge(tensors, dim=0, value=0, dtype=None):
      """Merges list of tensors into one."""
      tensors = [tensor if isinstance(tensor, torch.Tensor) else torch.tensor(tensor) for tensor in tensors]
      dim = dim if dim != -1 else len(tensors[0].shape) - 1
      dtype = tensors[0].dtype if dtype is None else dtype
      max_len = max(tensor.shape[dim] for tensor in tensors)
      new_tensors = []
      for tensor in tensors:
          pad = (2 * len(tensor.shape)) * [0]
          pad[-2 * dim - 1] = max_len - tensor.shape[dim]
          new_tensors.append(F.pad(tensor, pad=pad, value=value))
      return torch.stack(new_tensors).to(dtype=dtype)

def repeat_merge(x, reps, pad):
        """Repeats `x` values according to `reps` tensor and merges."""
        return merge(
            tensors=[torch.repeat_interleave(text1, durs1) for text1, durs1 in zip(x, reps)], value=pad, dtype=x.dtype,
        )
class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer.."""

    EPS = 1e-6

    def __init__(
        self, idim, embed_dim=64, padding_idx=0, sigma_c=2.0, merge_blanks=False,
    ):
        super().__init__()

        self.embed = nn.Embedding(idim, embedding_dim=embed_dim, padding_idx=padding_idx)
        self.pad = 0
        self.sigma_c = sigma_c
        self.merge_blanks = merge_blanks

    def forward(self, text, durs):
        """See base class."""
        # Fake padding
        text = F.pad(text, [0, 2, 0, 0], value=self.pad)
        durs = F.pad(durs, [0, 2, 0, 0], value=0)

        repeats = repeat_merge(text, durs, self.pad)
        print(repeats.shape)
        total_time = repeats.shape[-1]

        # Centroids: [B,T,N]
        c = (durs / 2.0) + F.pad(torch.cumsum(durs, dim=-1)[:, :-1], [1, 0, 0, 0], value=0)
        c = c.unsqueeze(1).repeat(1, total_time, 1)

        # Sigmas: [B,T,N]
        sigmas = durs
        sigmas = sigmas.float() / self.sigma_c
        sigmas = sigmas.unsqueeze(1).repeat(1, total_time, 1) + self.EPS
        assert c.shape == sigmas.shape

        # Times at indexes
        t = torch.arange(total_time, device=c.device).view(1, -1, 1).repeat(durs.shape[0], 1, durs.shape[-1]).float()
        t = t + 0.5

        ns = slice(None)
        if self.merge_blanks:
            ns = slice(1, None, 2)

        # Weights: [B,T,N]
        d = torch.distributions.normal.Normal(c, sigmas)
        w = d.log_prob(t).exp()[:, :, ns]  # [B,T,N]
        pad_mask = (text == self.pad)[:, ns].unsqueeze(1).repeat(1, total_time, 1)
        w.masked_fill_(pad_mask, 0.0)  # noqa
        w = w / (w.sum(-1, keepdim=True) + self.EPS)
        pad_mask = (repeats == self.pad).unsqueeze(-1).repeat(1, 1, text[:, ns].size(1))  # noqa
        w.masked_fill_(pad_mask, 0.0)  # noqa
        pad_mask[:, :, :-1] = False
        w.masked_fill_(pad_mask, 1.0)  # noqa

        # Embeds
        u = torch.bmm(w, self.embed(text)[:, ns, :])  # [B,T,E]

        return u


# --- [Original file: module.py] ---
class MaskedInstanceNorm1d(nn.Module):
    """Instance norm + masking."""

    MAX_CNT = 1e5

    def __init__(self, d_channel: int, unbiased: bool = True, affine: bool = False):
        super().__init__()

        self.d_channel = d_channel
        self.unbiased = unbiased

        self.affine = affine
        if self.affine:
            gamma = torch.ones(d_channel, dtype=torch.float)
            beta = torch.zeros_like(gamma)
            self.register_parameter('gamma', nn.Parameter(gamma))
            self.register_parameter('beta', nn.Parameter(beta))

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:  # noqa
        """`x`: [B,C,T], `x_mask`: [B,T] => [B,C,T]."""
        x_mask = x_mask.unsqueeze(1).type_as(x)  # [B,1,T]
        cnt = x_mask.sum(dim=-1, keepdim=True)  # [B,1,1]

        # Mean: [B,C,1]
        cnt_for_mu = cnt.clamp(1.0, self.MAX_CNT)
        mu = (x * x_mask).sum(dim=-1, keepdim=True) / cnt_for_mu

        # Variance: [B,C,1]
        sigma = (x - mu) ** 2
        cnt_fot_sigma = (cnt - int(self.unbiased)).clamp(1.0, self.MAX_CNT)
        sigma = (sigma * x_mask).sum(dim=-1, keepdim=True) / cnt_fot_sigma
        sigma = (sigma + 1e-8).sqrt()

        y = (x - mu) / sigma

        if self.affine:
            gamma = self.gamma.unsqueeze(0).unsqueeze(-1)
            beta = self.beta.unsqueeze(0).unsqueeze(-1)
            y = y * gamma + beta

        return y


class StyleResidual(nn.Module):
    """Styling."""

    def __init__(self, d_channel: int, d_style: int, kernel_size: int = 1):
        super().__init__()

        self.rs = nn.Conv1d(
            in_channels=d_style, out_channels=d_channel, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
        )

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """`x`: [B,C,T], `s`: [B,S,T] => [B,C,T]."""
        return x + self.rs(s)


class Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.
    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.
    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
        self,
        odim: int,
        n_layers: int = 5,
        n_chans: int = 512,
        n_filts: int = 5,
        dropout_rate: float = 0.5,
        use_batch_norm: bool = True,
    ):
        """Initialize postnet module.
        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super(Postnet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for layer in range(n_layers - 1):
            ichans = odim if layer == 0 else n_chans
            ochans = odim if layer == n_layers - 1 else n_chans
            if use_batch_norm:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.BatchNorm1d(ochans),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
            else:
                self.postnet += [
                    torch.nn.Sequential(
                        torch.nn.Conv1d(
                            ichans,
                            ochans,
                            n_filts,
                            stride=1,
                            padding=(n_filts - 1) // 2,
                            bias=False,
                        ),
                        torch.nn.Tanh(),
                        torch.nn.Dropout(dropout_rate),
                    )
                ]
        ichans = n_chans if n_layers != 1 else odim
        if use_batch_norm:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.BatchNorm1d(odim),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        else:
            self.postnet += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        ichans,
                        odim,
                        n_filts,
                        stride=1,
                        padding=(n_filts - 1) // 2,
                        bias=False,
                    ),
                    torch.nn.Dropout(dropout_rate),
                )
            ]

    def forward(self, xs):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (B, idim, Tmax).
        Returns:
            Tensor: Batch of padded output tensor. (B, odim, Tmax).
        """
        for postnet in self.postnet:
            xs = postnet(xs)
        return xs


# --- [Original file: quartznet.py] ---
class SepConv1d(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,):
        super(SepConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels,
                                         in_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=(kernel_size - 1) // 2,
                                         dilation=dilation,
                                         groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

class ConvBN1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):

      super(ConvBN1d, self).__init__()
      self.conv = nn.Sequential(
          nn.Conv1d(in_channels, out_channels, kernel_size, stride,
                    padding=(kernel_size - 1) // 2),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(),
          nn.Dropout(0.1)
      )

    def forward(self, x, mask=None):
        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        return self.conv(x), mask

class ActSepConv1d(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dilation=1,
                 dropout=0.1,):
        super(ActSepConv1d, self).__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.ReLU(),
            SepConv1d(in_channels, out_channels, kernel_size, stride, dilation)
        )

    def forward(self, x, mask=None):

        if mask is not None:
            x = x * mask.unsqueeze(1).to(device=x.device)
        x = self.model(x)
        return x, mask



class QuartzNetBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, R=5, dropout=0.1):
        super(QuartzNetBlock, self).__init__()

        model = [SepConv1d(in_channels, out_channels, kernel_size, stride)]

        for i in range(R - 1):
            model += [ActSepConv1d(out_channels, out_channels, kernel_size, stride)]

        self.model = nn.Sequential(*model)

        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, 1),
            nn.BatchNorm1d(out_channels)
        )

    def forward(self, x, mask=None):
        x = x * mask.unsqueeze(1).to(device=x.device) if mask is not None else x
        x = self.residual(x) + self.model(x, mask)

        return F.relu(x), mask


class QuartzNet5x5(nn.Module):

    def __init__(self, idim, odim, qdim=256, kernels=[5, 7, 9, 11, 13]):
        super(QuartzNet5x5, self).__init__()



        self.conv1 = nn.Sequential(
            ConvBN1d(idim, qdim, 3),
            ConvBN1d(qdim, qdim, 3),
            ConvBN1d(qdim, qdim, 3)
        )

        quartznet = []

        for k in kernels:
            quartznet.append(QuartzNetBlock(qdim, qdim, k))
        self.quartznet = nn.Sequential(*quartznet)

        self.conv2 = nn.Sequential(
            ConvBN1d(qdim, qdim * 2, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(qdim * 2, odim, 1)
        )

    def forward(self, x, mask=None):
        x = self.conv1(x, mask)
        x = self.quartznet(x, mask)
        x = self.conv2(x, mask)
        x = self.conv3(x, mask)
        return x

class QuartzNet9x5(nn.Module):

    def __init__(self, idim=256, odim=80, qdim=256, kernels1=[5, 7, 9, 13, 15, 17], kernels2=[21, 23,25]):
        super(QuartzNet9x5, self).__init__()



        self.conv1 = nn.Sequential(
            ConvBN1d(idim, qdim, 3),
            ConvBN1d(qdim, qdim, 3),
            ConvBN1d(qdim, qdim, 3)
        )

        quartznet = []

        for k in kernels1:
            quartznet.append(QuartzNetBlock(qdim, qdim, k))

        n = qdim * 2
        for k in kernels2:
            quartznet.append(QuartzNetBlock(qdim, n, k))
            qdim = n
        self.quartznet = nn.Sequential(*quartznet)

        self.conv2 = nn.Sequential(
            ConvBN1d(n, n * 2, 1)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(n * 2, odim, 1)
        )

    def forward(self, x, mask=None):
        x = self.conv1(x, mask)
        x = self.quartznet(x, mask)
        x = self.conv2(x, mask)
        x = self.conv3(x, mask)
        return x


# --- [Original file: model.py] ---
class GraphemeDuration(nn.Module):

    def __init__(self, idim, embed_dim=64, padding_idx=0):
        super(GraphemeDuration, self).__init__()
        self.embed = nn.Embedding(idim, embedding_dim=embed_dim, padding_idx=padding_idx)
        self.predictor = QuartzNet5x5(embed_dim, 32)
        self.projection = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, text, text_len, is_mask=True):
        x, x_len = self.embed(text).transpose(1, 2), text_len
        if is_mask:
            mask = get_mask_from_lengths(x_len)
        else:
            mask = None
        out = self.predictor(x, mask)
        out = self.projection(out).squeeze(1)

        return out

    @staticmethod
    def _metrics(true_durs, true_text_len, pred_durs):
        loss = F.mse_loss(pred_durs, (true_durs + 1).float().log(), reduction='none')
        mask = get_mask_from_lengths(true_text_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()

        durs_pred = pred_durs.exp() - 1
        durs_pred[durs_pred < 0.0] = 0.0
        durs_pred = durs_pred.round().long()

        acc = ((true_durs == durs_pred) * mask).sum().float() / mask.sum() * 100
        acc_dist_1 = (((true_durs - durs_pred).abs() <= 1) * mask).sum().float() / mask.sum() * 100
        acc_dist_3 = (((true_durs - durs_pred).abs() <= 3) * mask).sum().float() / mask.sum() * 100

        return loss, acc, acc_dist_1, acc_dist_3


class PitchPredictor(nn.Module):

    def __init__(self, idim,  embed_dim=64):
        super(PitchPredictor, self).__init__()
        self.embed = GaussianEmbedding(idim, embed_dim)
        self.predictor = QuartzNet5x5(embed_dim, 32)
        self.sil_proj = nn.Conv1d(32, 1, kernel_size=1)
        self.body_proj = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, text, durs, is_mask=True):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        if is_mask:
            mask = get_mask_from_lengths(x_len)
        else:
            mask = None
        out = self.predictor(x, mask)
        uv = self.sil_proj(out).squeeze(1)
        value = self.body_proj(out).squeeze(1)

        return uv, value

    def _metrics(self, true_f0, true_f0_mask, pred_f0_sil, pred_f0_body):
        sil_mask = true_f0 < 1e-5
        sil_gt = sil_mask.long()
        sil_loss = F.binary_cross_entropy_with_logits(input=pred_f0_sil, target=sil_gt.float(), reduction='none', )
        sil_loss *= true_f0_mask.type_as(sil_loss)
        sil_loss = sil_loss.sum() / true_f0_mask.sum()
        sil_acc = ((torch.sigmoid(pred_f0_sil) > 0.5).long() == sil_gt).float()  # noqa
        sil_acc *= true_f0_mask.type_as(sil_acc)
        sil_acc = sil_acc.sum() / true_f0_mask.sum()

        body_mse = F.mse_loss(pred_f0_body, (true_f0 - self.f0_mean) / self.f0_std, reduction='none')
        body_mask = ~sil_mask
        body_mse *= body_mask.type_as(body_mse)  # noqa
        body_mse = body_mse.sum() / body_mask.sum()  # noqa
        body_mae = ((pred_f0_body * self.f0_std + self.f0_mean) - true_f0).abs()
        body_mae *= body_mask.type_as(body_mae)  # noqa
        body_mae = body_mae.sum() / body_mask.sum()  # noqa

        loss = sil_loss + body_mse

        return loss, sil_acc, body_mae


class TalkNet2(nn.Module):

    def __init__(self, idim, odim=80, embed_dim=256, postnet_layers = 0):
        super(TalkNet2, self).__init__()
        self.embed = GaussianEmbedding(idim, embed_dim)
        self.norm_f0 = MaskedInstanceNorm1d(1)
        self.res_f0 = StyleResidual(embed_dim, 1, kernel_size=3)

        self.generator = QuartzNet9x5(embed_dim, odim)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                odim=odim,
                n_layers=postnet_layers,
                n_chans=256,
                n_filts=5,
                use_batch_norm=True,
                dropout_rate=0.5,
            )
        )


    def forward(self, text, durs, f0, is_mask=True):
        x, x_len = self.embed(text, durs).transpose(1, 2), durs.sum(-1)
        f0, f0_mask = f0.clone(), f0 > 0.0
        f0 = self.norm_f0(f0.unsqueeze(1), f0_mask)
        f0[~f0_mask.unsqueeze(1)] = 0.0
        x = self.res_f0(x, f0)
        if is_mask:
            mask = get_mask_from_lengths(x_len)
        else:
            mask = None

        before_outs = self.generator(x, mask)
        if self.postnet is None:
            return before_outs, None
        else:
            after_outs = before_outs + self.postnet(
                before_outs
            )
            return before_outs, after_outs



    @staticmethod
    def _metrics(true_mel, true_mel_len, pred_mel):
        loss = F.mse_loss(pred_mel, true_mel, reduction='none').mean(dim=-2)
        mask = get_mask_from_lengths(true_mel_len)
        loss *= mask.float()
        loss = loss.sum() / mask.sum()
        return loss


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
    print("TalkNet2 benchmark: duration, pitch, Gaussian expansion, and mel generation")
    print("=" * 70)

    class TinyPredictor(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.proj = nn.Conv1d(in_channels, out_channels, 1)

        def forward(self, x, mask=None):
            if mask is not None:
                x = x * mask.unsqueeze(1).to(device=x.device)
            return self.proj(x)

    class MaskSequential(nn.Sequential):
        def forward(self, x, mask=None):
            for module in self:
                result = module(x, mask)
                if isinstance(result, tuple):
                    x, mask = result
                else:
                    x = result
            return x

    print("-" * 70)
    print("[Test 1/6] merge, repeat_merge, and GaussianEmbedding")
    try:
        merged = merge([torch.tensor([1, 2]), torch.tensor([3])], value=0)
        repeated = repeat_merge(torch.tensor([[1, 2, 3], [4, 5, 0]]), torch.tensor([[1, 2, 0], [2, 1, 0]]), 0)
        ge = GaussianEmbedding(idim=8, embed_dim=4)
        text = torch.tensor([[1, 2, 3], [3, 2, 1]], dtype=torch.long)
        durs = torch.tensor([[2, 1, 0], [1, 2, 1]], dtype=torch.long)
        out = ge(text, durs)
        check("merge output not None", merged is not None)
        check("merge pads to maximum length", merged.shape == (2, 2) and merged[1, 1].item() == 0, str(merged))
        check("repeat_merge expands and pads", repeated.shape == (2, 3) and repeated[0].tolist() == [1, 2, 2], str(repeated))
        check("GaussianEmbedding output not None", out is not None)
        if out is not None:
            check("GaussianEmbedding output shape follows duration sum", out.shape == (2, 4, 4), str(tuple(out.shape)))
            check("GaussianEmbedding output finite", torch.isfinite(out).all().item())
            out.sum().backward()
            check("GaussianEmbedding keeps embedding gradients", ge.embed.weight.grad is not None and torch.isfinite(ge.embed.weight.grad).all().item())
        else:
            skip_checks(3, "GaussianEmbedding returned None")
    except Exception as exc:
        skip_checks(7, f"embedding utilities raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 2/6] MaskedInstanceNorm1d and StyleResidual")
    try:
        norm = MaskedInstanceNorm1d(1)
        x = torch.tensor([[[1.0, 3.0, 99.0]], [[2.0, 4.0, 6.0]]])
        mask = torch.tensor([[True, True, False], [True, True, True]])
        y = norm(x, mask)
        sr = StyleResidual(d_channel=2, d_style=1, kernel_size=1)
        styled = sr(torch.ones(2, 2, 3), torch.randn(2, 1, 3))
        check("MaskedInstanceNorm output not None", y is not None)
        if y is not None:
            check("MaskedInstanceNorm preserves shape", y.shape == x.shape, str(tuple(y.shape)))
            masked_mean = (y * mask.unsqueeze(1)).sum(-1) / mask.sum(-1, keepdim=True)
            check("MaskedInstanceNorm zero-centers valid frames", torch.allclose(masked_mean, torch.zeros_like(masked_mean), atol=1e-5), str(masked_mean))
            check("MaskedInstanceNorm output finite", torch.isfinite(y).all().item())
        else:
            skip_checks(3, "MaskedInstanceNorm returned None")
        check("StyleResidual output shape", styled.shape == (2, 2, 3), str(tuple(styled.shape)))
    except Exception as exc:
        skip_checks(5, f"normalization/style residual raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 3/6] SepConv1d and QuartzNetBlock")
    try:
        sep = SepConv1d(3, 5, kernel_size=3)
        sep.eval()
        inp = torch.randn(2, 3, 6)
        mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, True]])
        sep_out = sep(inp, mask)
        block = QuartzNetBlock(3, 5, kernel_size=3, R=1)
        block.model = MaskSequential(*list(block.model.children()))
        block.eval()
        block_out, block_mask = block(inp, mask)
        check("SepConv1d output not None", sep_out is not None)
        if sep_out is not None:
            check("SepConv1d output shape", sep_out.shape == (2, 5, 6), str(tuple(sep_out.shape)))
            check("SepConv1d output finite", torch.isfinite(sep_out).all().item())
            check("SepConv1d uses depthwise groups", sep.depthwise.groups == 3, str(sep.depthwise.groups))
        else:
            skip_checks(3, "SepConv1d returned None")
        check("QuartzNetBlock returns tensor and mask", block_out is not None and block_mask is mask)
        if block_out is not None:
            check("QuartzNetBlock output shape", block_out.shape == (2, 5, 6), str(tuple(block_out.shape)))
            check("QuartzNetBlock ReLU non-negative output", (block_out >= 0).all().item())
        else:
            skip_checks(2, "QuartzNetBlock returned None")
    except Exception as exc:
        skip_checks(7, f"QuartzNet primitive raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 4/6] GraphemeDuration forward and metrics")
    try:
        gd = GraphemeDuration(idim=12, embed_dim=4)
        gd.predictor = TinyPredictor(4, 32)
        text = torch.tensor([[1, 2, 3, 0], [2, 3, 4, 5]])
        text_len = torch.tensor([3, 4])
        pred = gd(text, text_len)
        true_durs = torch.tensor([[1, 2, 0, 0], [1, 1, 2, 3]])
        loss, acc, acc1, acc3 = GraphemeDuration._metrics(true_durs, text_len, torch.log(true_durs.float() + 1.0))
        check("GraphemeDuration output not None", pred is not None)
        if pred is not None:
            check("GraphemeDuration output shape", pred.shape == (2, 4), str(tuple(pred.shape)))
            check("GraphemeDuration output finite", torch.isfinite(pred).all().item())
        else:
            skip_checks(2, "GraphemeDuration returned None")
        check("Duration metrics scalar finite loss", loss.dim() == 0 and torch.isfinite(loss).item())
        check("Duration exact prediction reaches full accuracy", torch.allclose(acc, torch.tensor(100.0)), str(acc))
        check("Duration distance metrics are bounded", 0 <= acc1.item() <= 100 and 0 <= acc3.item() <= 100)
    except Exception as exc:
        skip_checks(6, f"GraphemeDuration raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 5/6] PitchPredictor forward and metrics")
    try:
        pp = PitchPredictor(idim=12, embed_dim=4)
        pp.predictor = TinyPredictor(4, 32)
        pp.f0_mean = 5.0
        pp.f0_std = 2.0
        text = torch.tensor([[1, 2, 3], [3, 2, 1]])
        durs = torch.tensor([[1, 2, 1], [2, 1, 1]])
        uv, value = pp(text, durs)
        true_f0 = torch.tensor([[0.0, 5.0, 7.0, 0.0], [4.0, 0.0, 8.0, 6.0]])
        f0_mask = torch.ones_like(true_f0).bool()
        loss, sil_acc, body_mae = pp._metrics(true_f0, f0_mask, torch.zeros_like(true_f0), (true_f0 - pp.f0_mean) / pp.f0_std)
        check("PitchPredictor outputs not None", uv is not None and value is not None)
        if uv is not None and value is not None:
            check("PitchPredictor output shapes", uv.shape == (2, 4) and value.shape == (2, 4), f"{tuple(uv.shape)}, {tuple(value.shape)}")
            check("PitchPredictor outputs finite", torch.isfinite(uv).all().item() and torch.isfinite(value).all().item())
        else:
            skip_checks(2, "PitchPredictor returned None")
        check("Pitch metrics scalar finite loss", loss.dim() == 0 and torch.isfinite(loss).item())
        check("Pitch metrics finite accuracy and MAE", torch.isfinite(sil_acc).item() and torch.isfinite(body_mae).item())
    except Exception as exc:
        skip_checks(5, f"PitchPredictor raised {type(exc).__name__}: {exc}")

    print("-" * 70)
    print("[Test 6/6] TalkNet2 forward and mel metric")
    try:
        tn = TalkNet2(idim=12, odim=3, embed_dim=4, postnet_layers=0)
        tn.generator = TinyPredictor(4, 3)
        text = torch.tensor([[1, 2, 3], [3, 2, 1]])
        durs = torch.tensor([[1, 2, 1], [2, 1, 1]])
        f0 = torch.tensor([[0.0, 5.0, 7.0, 0.0], [4.0, 0.0, 8.0, 6.0]])
        before, after = tn(text, durs, f0)
        mel_loss = TalkNet2._metrics(torch.zeros(2, 3, 4), torch.tensor([4, 3]), torch.zeros(2, 3, 4))
        check("TalkNet2 before output not None", before is not None)
        if before is not None:
            check("TalkNet2 before output shape", before.shape == (2, 3, 4), str(tuple(before.shape)))
            check("TalkNet2 before output finite", torch.isfinite(before).all().item())
        else:
            skip_checks(2, "TalkNet2 returned no before output")
        check("TalkNet2 returns None after output without postnet", after is None)
        check("TalkNet2 mel metric scalar zero for perfect prediction", torch.allclose(mel_loss, torch.tensor(0.0)), str(mel_loss))
    except Exception as exc:
        skip_checks(5, f"TalkNet2 raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
