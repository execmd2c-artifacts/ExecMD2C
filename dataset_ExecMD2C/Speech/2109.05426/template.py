"""Ground-truth core model components for the Zero-Shot-TTS benchmark.

This file consolidates only the model architecture components from the original
repository. Corpus handling, score reporting, saved artifact I/O, and generation
orchestration are not included.
"""

import math
from typing import List
from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import torch.nn.functional as F


# --- [Original file: module.py] ---

class LayerNorm(torch.nn.Module):
    def __init__(self, nout: int):
        super(LayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(nout, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x.transpose(1, -1))
        x = x.transpose(1, -1)
        return x


class DurationPredictor(torch.nn.Module):
    """Duration predictor module.
    This is a module of duration predictor described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain from the hidden embeddings of encoder.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    Note:
        The calculation domain of outputs is different between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`, those are calculated in linear domain.
    """

    def __init__(
        self, idim, n_layers=2, n_chans=256, kernel_size=3, dropout_rate=0.1, offset=1.0
    ):
        """Initilize duration predictor module.
        Args:
            idim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.
        """
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = idim if idx == 0 else n_chans
            self.conv += [
                torch.nn.Sequential(
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans),
                    torch.nn.Dropout(dropout_rate),
                )
            ]
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(
        self,
        xs: torch.Tensor,
        x_masks: Optional[torch.Tensor] = None,
        is_inference: bool = False,
    ):
        """[TODO] Predict token durations in training or inference domain.

        Input:
            xs: (batch, time, idim) - encoder hidden states for text tokens.
            x_masks: optional (batch, time) - True values mark padded tokens.
            is_inference: bool - choose log-domain training output or integer
                linear-domain inference durations.

        Output:
            (batch, time) - duration predictions. Training mode returns floating
            log-domain values; inference mode returns nonnegative integer
            durations.

"""
        pass

    def forward(self, xs: torch.Tensor, x_masks: Optional[torch.Tensor] = None):
        """Calculate forward propagation.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            Tensor: Batch of predicted durations in log domain (B, Tmax).
        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks: Optional[torch.Tensor] = None):
        """Inference duration.
        Args:
            xs (Tensor): Batch of input sequences (B, Tmax, idim).
            x_masks (ByteTensor, optional): Batch of masks indicating padded part (B, Tmax).
        Returns:
            LongTensor: Batch of predicted durations in linear domain (B, Tmax).
        """
        return self._forward(xs, x_masks, True)


class Conv1dBNReLU(nn.Module):
    """Linear layer with Batch Normalization.
    x -> conv1d -> BN -> o
    Args:
        in_features (int): number of channels in the input tensor.
        out_features (int ): number of channels in the output tensor.
        bias (bool, optional): enable/disable bias in the linear layer. Defaults to True.
        init_gain (str, optional): method to set the gain for weight initialization. Defaults to 'linear'.
    """

    def __init__(self, in_features, out_features, kernel_size=5, bias=False, init_gain="relu"):
        super().__init__()
        self.linear_layer = torch.nn.Conv1d(in_features, out_features, kernel_size=kernel_size, bias=bias)
        self.batch_normalization = nn.BatchNorm1d(out_features, momentum=0.1, eps=1e-5)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        """
        Shapes:
            x: [B, C, T]
        """
        out = self.linear_layer(x)
        out = self.batch_normalization(out)
        return F.relu(out)


def pad_2d_tensor(xs: List[torch.Tensor], pad_value: float = 0.0):
    max_len = max([xs[i].size(0) for i in range(len(xs))])

    out_list = []

    for i, batch in enumerate(xs):
        one_batch_padded = F.pad(
            batch, (0, 0, 0, max_len - batch.size(0)), "constant", pad_value
        )
        out_list.append(one_batch_padded)

    out_padded = torch.stack(out_list)
    return out_padded


class LengthRegulator(torch.nn.Module):
    """Length regulator module for feed-forward Transformer.
    This is a module of length regulator described in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.
    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, pad_value: float = 0.0):
        """Initilize length regulator module.
        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(
        self,
        xs: torch.Tensor,
        ds: torch.Tensor,
        ilens: torch.Tensor,
        alpha: float = 1.0,
    ) -> torch.Tensor:
        """[TODO] Expand each token embedding sequence according to durations.

        Input:
            xs: (batch, max_tokens, channels) - token or phoneme-level hidden
                representations.
            ds: (batch, max_tokens) - integer duration assigned to each token.
            ilens: (batch,) - valid token count for each sample before padding.
            alpha: scalar - positive speed-control factor for duration scaling.

        Output:
            (batch, max_expanded_time, channels) - batch of frame-level hidden
            representations padded to the longest expanded sequence.

"""
        pass

    def _repeat_one_sequence(self, x: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """[TODO] Repeat one valid token sequence according to duration counts.

        Input:
            x: (tokens, channels) - one unpadded token-level sequence.
            d: (tokens,) - integer duration for each token.

        Output:
            (sum_positive_durations, channels) - frame-level sequence produced
            by repeating each token representation.

"""
        pass


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for non-recurrent neural networks.
    Implementation based on "Attention Is All You Need"
    Args:
       channels (int): embedding size
       dropout (float): dropout parameter
    """

    def __init__(self, channels, dropout_p=0.0, max_len=5000):
        super().__init__()
        if channels % 2 != 0:
            raise ValueError(
                "Cannot use sin/cos positional encoding with " "odd channels (got channels={:d})".format(channels)
            )
        pe = torch.zeros(max_len, channels)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, channels, 2).float() / channels)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer("pe", pe)
        if dropout_p > 0:
            self.dropout = nn.Dropout(p=dropout_p)
        self.channels = channels

    def forward(self, x, mask=None, first_idx=None, last_idx=None):
        """
        Shapes:
            x: [B, C, T]
            mask: [B, 1, T]
            first_idx: int
            last_idx: int
        """

        x = x * math.sqrt(self.channels)
        if first_idx is None:
            if self.pe.size(2) < x.size(2):
                raise RuntimeError(
                    f"Sequence is {x.size(2)} but PositionalEncoding is"
                    f" limited to {self.pe.size(2)}. See max_len argument."
                )
            if mask is not None:
                pos_enc = self.pe[:, :, : x.size(2)] * mask
            else:
                pos_enc = self.pe[:, :, : x.size(2)]
            x = x + pos_enc
        else:
            x = x + self.pe[:, :, first_idx:last_idx]
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        return x


class Postnet(torch.nn.Module):
    """Postnet module for Spectrogram prediction network.
    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet predicts refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.
    .. _`Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(
        self,
        idim: int,
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


def make_pad_mask(lengths: List[int], xs: torch.Tensor = None, length_dim: int = -1):
    """Make mask tensor containing indices of padded part."""
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


# --- [Original file: transformer.py] ---

def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.cpu().unsqueeze(1).expand(-1, max_len)

    return mask


class FFTransformer(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn=1024, kernel_size_up=9, kernel_size_down=1,
                 dropout_p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_out_channels, num_heads, dropout=dropout_p)

        self.conv1 = nn.Conv1d(in_out_channels, hidden_channels_ffn, kernel_size=kernel_size_up,
                               padding=(kernel_size_up - 1) // 2)
        self.conv2 = nn.Conv1d(hidden_channels_ffn, in_out_channels, kernel_size=kernel_size_down,
                               padding=(kernel_size_down - 1) // 2)

        self.norm1 = nn.LayerNorm(in_out_channels)
        self.norm2 = nn.LayerNorm(in_out_channels)

        self.dropout = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """[TODO] Apply one feed-forward Transformer layer to channel-first speech features.

        Input:
            src: (batch, channels, time) - hidden speech/text sequence features.
            src_mask: optional attention mask for time-to-time attention.
            src_key_padding_mask: optional (batch, time) mask for padded keys.

        Output:
            transformed: (batch, channels, time) - updated sequence features.
            alignment: attention alignment tensor over the time dimension.

"""
        pass


class FFTransformerBlock(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn, num_layers, kernel_size_up=9, kernel_size_down=1,
                 dropout_p=0.1):
        super().__init__()
        self.fft_layers = nn.ModuleList(
            [
                FFTransformer(
                    in_out_channels=in_out_channels,
                    num_heads=num_heads,
                    hidden_channels_ffn=hidden_channels_ffn,
                    kernel_size_up=kernel_size_up,
                    kernel_size_down=kernel_size_down,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, length=None):  # pylint: disable=unused-argument
        """[TODO] Stack feed-forward Transformer layers with length-derived padding masks.

        Input:
            x: (batch, channels, time) - sequence representation to encode or
                decode.
            length: (batch,) - valid time length for each sample.

        Output:
            x: (batch, channels, time) - representation after all FFT layers.
            mask: (batch, time) - boolean padding mask derived from lengths.

"""
        pass


# --- [Original file: model.py] ---

class MelGenerator(nn.Module):
    def __init__(self, idim, hp):
        super().__init__()
        # use idx 0 as padding idx
        padding_idx = 0


        # Embedding
        self.text_embed = nn.Embedding(
                num_embeddings=idim, embedding_dim=512, padding_idx=padding_idx
            )

        self.text_encoder_prenet = nn.Sequential(
            Conv1dBNReLU(512, 512),
            nn.Dropout(hp.model.prenet_dropout),
            Conv1dBNReLU(512, 512),
            nn.Dropout(hp.model.prenet_dropout),
            Conv1dBNReLU(512, 512),
            nn.Dropout(hp.model.prenet_dropout),
            nn.Conv1d(512, hp.model.adim, 1)

        )


        #Encoder
        self.pos_enc = PositionalEncoding(hp.model.adim, dropout_p=0.1)
        self.encoder = FFTransformerBlock(hp.model.adim, hp.model.aheads, hp.model.eunits, hp.model.elayers,
                                          kernel_size_up=hp.model.positionwise_conv_kernel_size1,
                                          kernel_size_down=hp.model.positionwise_conv_kernel_size2, dropout_p=0.1)

        # Spectrogram Encoder
        self.input_layer = nn.Conv1d(hp.audio.num_mels, hp.model.adim, 1)
        self.spec_embed = nn.Conv1d(hp.model.adim, hp.model.adim, 1)
        self.spec_pos_enc = PositionalEncoding(hp.model.adim, dropout_p=0.1)
        self.spec_encoder = FFTransformerBlock(hp.model.adim, hp.model.sheads, hp.model.sunits, hp.model.slayers,
                                               kernel_size_up=hp.model.positionwise_conv_kernel_size1,
                                               kernel_size_down=hp.model.positionwise_conv_kernel_size2, dropout_p=0.1)


        # Duration Predictor
        self.duration_predictor = DurationPredictor(
            idim=hp.model.adim,
            n_layers=hp.model.duration_predictor_layers,
            n_chans=hp.model.duration_predictor_chans,
            kernel_size=hp.model.duration_predictor_kernel_size,
            dropout_rate=hp.model.duration_predictor_dropout_rate,
        )


        # Length regulator
        self.length_regulator = LengthRegulator()


        # Decoder
        self.pos_dec = PositionalEncoding(hp.model.ddim, dropout_p=0.1)
        self.decoder = FFTransformerBlock(hp.model.ddim, hp.model.aheads, hp.model.dunits, hp.model.dlayers,
                                          kernel_size_up=hp.model.positionwise_conv_kernel_size1,
                                          kernel_size_down=hp.model.positionwise_conv_kernel_size2, dropout_p=0.1)




        # Postnet
        self.postnet = (
            None
            if hp.model.postnet_layers == 0
            else Postnet(
                idim=hp.audio.num_mels,
                odim=hp.audio.num_mels,
                n_layers=hp.model.postnet_layers,
                n_chans=hp.model.postnet_chans,
                n_filts=hp.model.postnet_filts,
                use_batch_norm=hp.model.use_batch_norm,
                dropout_rate=hp.model.postnet_dropout_rate,
            )
        )


        self.spectrogram_out = nn.Linear(hp.model.adim, hp.audio.num_mels)

    def forward(self, text, duration, ilens, mel):
        '''
        inputs:
            text : [B, Lmax, Dim]
            duration: [B, Lmax]
        outputs :
            mel_spec : [B, Tmax, Bin]
        '''

        # Embedding
        emb = self.embed(text)                  # [B, Lmax, 512]
        emb = emb.transpose(1, -1).contiguous() # [B, 512, Lmax]

        emb = self.text_encoder_prenet(emb)     # [B, 256, Lmax]

        # Encoder
        emb = self.pos_enc(emb)                 # [B, 256, Lmax]
        hs, mask = self.encoder(emb, ilens)         # [B, 256, Lmax]

        # forward duration predictor and length regulator
        d_masks = make_pad_mask(ilens).to(text.device)

        d_outs = self.duration_predictor(hs.transpose(1, -1), d_masks)  # (B, Tmax)

        # Spectrogram Encoder
        spec_emb = F.relu(self.input_layer(mel.transpose(-2, -1)))  # [B, 256, Tmax]
        spec_emb = F.relu(self.spec_embed(spec_emb))
        spec_emb = self.spec_pos_enc(spec_emb)
        hs_spec, mask_spec = self.spec_encoder(spec_emb, ilens)  # [B, 256, Lmax]

        # Length Regulator
        hs = self.length_regulator(hs.transpose(1, -1), duration, ilens)    # [B, Tmax, 256]



        # Decoder
        hs = hs.transpose(1, -1).contiguous()       # [B, 256, Tmax]
        hs = hs + hs_spec
        olens = duration.sum(-1)  # [B]
        hs = self.pos_dec(hs)
        out, mask = self.decoder(hs, olens)             # [B, 256, Tmax]

        mel_spec = self.spectrogram_out(out.transpose(1, -1))       # [B, Tmax, 80]

        # Postnet -> (B, Lmax//r * r, odim)
        if self.postnet is None:
            mel_spec_fine = None
        else:
            mel_spec_fine = mel_spec + self.postnet(
                mel_spec.transpose(1, 2)
            ).transpose(1, 2)

        return mel_spec, mel_spec_fine, mask, olens, d_outs


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

    print("Zero-Shot-TTS benchmark checks")

    print("[Test 1/5] DurationPredictor._forward")
    try:
        predictor = DurationPredictor(idim=4, n_layers=1, n_chans=8, kernel_size=3, dropout_rate=0.0)
        predictor.eval()
        xs = torch.randn(2, 5, 4)
        masks = torch.tensor([[False, False, False, False, True], [False, False, False, True, True]])
        output = predictor._forward(xs, masks, False)
        inference_output = predictor._forward(xs, masks, True)
        check("DurationPredictor output not None", output is not None)
        if output is not None:
            check("DurationPredictor output shape", tuple(output.shape) == (2, 5), str(tuple(output.shape)))
            check("DurationPredictor output finite", torch.isfinite(output).all().item())
            check("DurationPredictor mask zeroes padded positions",
                  torch.all(output[masks] == 0).item(), str(output[masks]))
            check("DurationPredictor inference returns nonnegative long durations",
                  inference_output.dtype == torch.long and torch.all(inference_output >= 0).item(),
                  f"{inference_output.dtype}, min={inference_output.min().item()}")
        else:
            skip_checks(4, "DurationPredictor returned None")
    except Exception as exc:
        skip_checks(5, f"DurationPredictor._forward raised {type(exc).__name__}: {exc}")

    print("[Test 2/5] LengthRegulator._repeat_one_sequence")
    try:
        regulator = LengthRegulator()
        sequence = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
        durations = torch.tensor([1, 2, 1])
        repeated = regulator._repeat_one_sequence(sequence, durations)
        zero_repeated = regulator._repeat_one_sequence(sequence, torch.tensor([0, 0, 0]))
        expected = torch.tensor([[1.0, 10.0], [2.0, 20.0], [2.0, 20.0], [3.0, 30.0]])
        check("LengthRegulator repeat output not None", repeated is not None)
        if repeated is not None:
            check("LengthRegulator repeat output shape", tuple(repeated.shape) == (4, 2), str(tuple(repeated.shape)))
            check("LengthRegulator repeat pattern", torch.equal(repeated, expected), str(repeated))
            check("LengthRegulator all-zero durations fallback", tuple(zero_repeated.shape) == (3, 2),
                  str(tuple(zero_repeated.shape)))
        else:
            skip_checks(3, "LengthRegulator repeat returned None")
    except Exception as exc:
        skip_checks(4, f"LengthRegulator._repeat_one_sequence raised {type(exc).__name__}: {exc}")

    print("[Test 3/5] LengthRegulator.forward")
    try:
        regulator = LengthRegulator()
        xs = torch.tensor([
            [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]],
            [[4.0, 4.0], [5.0, 5.0], [6.0, 6.0]],
        ])
        durations = torch.tensor([[1, 2, 1], [2, 1, 0]])
        ilens = torch.tensor([3, 2])
        expanded = regulator(xs, durations, ilens)
        check("LengthRegulator forward output not None", expanded is not None)
        if expanded is not None:
            check("LengthRegulator forward output shape", tuple(expanded.shape) == (2, 4, 2),
                  str(tuple(expanded.shape)))
            check("LengthRegulator forward finite", torch.isfinite(expanded).all().item())
            check("LengthRegulator pads shorter expanded sequence",
                  torch.equal(expanded[1, 3], torch.zeros(2)), str(expanded[1, 3]))
        else:
            skip_checks(3, "LengthRegulator forward returned None")
    except Exception as exc:
        skip_checks(4, f"LengthRegulator.forward raised {type(exc).__name__}: {exc}")

    print("[Test 4/5] FFTransformer.forward")
    try:
        fft = FFTransformer(in_out_channels=8, num_heads=2, hidden_channels_ffn=16,
                            kernel_size_up=3, kernel_size_down=1, dropout_p=0.0)
        fft.eval()
        src = torch.randn(2, 8, 5)
        padding_mask = torch.tensor([[False, False, False, False, False],
                                     [False, False, False, True, True]])
        transformed, alignment = fft(src, src_key_padding_mask=padding_mask)
        check("FFTransformer output not None", transformed is not None and alignment is not None)
        if transformed is not None and alignment is not None:
            check("FFTransformer output shape", tuple(transformed.shape) == (2, 8, 5),
                  str(tuple(transformed.shape)))
            check("FFTransformer output finite", torch.isfinite(transformed).all().item())
            check("FFTransformer alignment shape", tuple(alignment.shape) == (2, 5, 5),
                  str(tuple(alignment.shape)))
        else:
            skip_checks(3, "FFTransformer returned None")
    except Exception as exc:
        skip_checks(4, f"FFTransformer.forward raised {type(exc).__name__}: {exc}")

    print("[Test 5/5] FFTransformerBlock.forward")
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        block = FFTransformerBlock(in_out_channels=8, num_heads=2, hidden_channels_ffn=16,
                                   num_layers=2, kernel_size_up=3, kernel_size_down=1,
                                   dropout_p=0.0).to(device)
        block.eval()
        x = torch.randn(2, 8, 5, device=device)
        lengths = torch.tensor([5, 3], device=device)
        block_output, block_mask = block(x, lengths)
        check("FFTransformerBlock output not None", block_output is not None and block_mask is not None)
        if block_output is not None and block_mask is not None:
            check("FFTransformerBlock output shape", tuple(block_output.shape) == (2, 8, 5),
                  str(tuple(block_output.shape)))
            check("FFTransformerBlock output finite", torch.isfinite(block_output).all().item())
            check("FFTransformerBlock mask marks padded frames",
                  tuple(block_mask.shape) == (2, 5) and bool(block_mask[1, 3].item()) and bool(block_mask[1, 4].item()),
                  str(block_mask))
        else:
            skip_checks(3, "FFTransformerBlock returned None")
    except Exception as exc:
        skip_checks(4, f"FFTransformerBlock.forward raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
