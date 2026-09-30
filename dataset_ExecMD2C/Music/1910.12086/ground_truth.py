"""Ground-truth core model components for the audio2score benchmark.

This file consolidates the DeepSpeech2-style model architecture used by the
project. Training loops, corpus preparation, CTC loss wrapping, score reporting,
saved artifact I/O, and generation orchestration are intentionally excluded.
"""

import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


# --- [Original file: deepspeech/model.py] ---

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

windows = {'hamming': np.hamming, 'hanning': np.hanning, 'blackman': np.blackman, 'bartlett': np.bartlett}


class SequenceWise(nn.Module):
    def __init__(self, module):
        """
        Collapses input of dim T*N*H to (T*N)*H, and applies to a module.
        Allows handling of variable sequence lengths and minibatch sizes.
        :param module: Module to apply input to.
        """
        super(SequenceWise, self).__init__()
        self.module = module

    def forward(self, x):
        t, n = x.size(0), x.size(1)
        x = x.view(t * n, -1)
        x = self.module(x)
        x = x.view(t, n, -1)
        return x

    def __repr__(self):
        tmpstr = self.__class__.__name__ + ' (\n'
        tmpstr += self.module.__repr__()
        tmpstr += ')'
        return tmpstr


class MaskConv(nn.Module):
    def __init__(self, seq_module):
        """
        Adds padding to the output of the module based on the given lengths. This is to ensure that the
        results of the model do not change when batch sizes change during inference.
        Input needs to be in the shape of (BxCxDxT)
        :param seq_module: The sequential module containing the conv stack.
        """
        super(MaskConv, self).__init__()
        self.seq_module = seq_module

    def forward(self, x, lengths):
        """
        :param x: The input of size BxCxDxT
        :param lengths: The actual length of each sequence in the batch
        :return: Masked output from the module
        """
        for module in self.seq_module:
            x = module(x)
            mask = torch.BoolTensor(x.size()).fill_(False)
            if x.is_cuda:
                mask = mask.cuda()
            for i, length in enumerate(lengths):
                length = length.item()
                if (mask[i].size(-1) - length) > 0:
                    mask[i].narrow(-1, length, mask[i].size(-1) - length).fill_(True)
            x = x.masked_fill(mask, 0)
        return x, lengths


class InferenceBatchSoftmax(nn.Module):
    def forward(self, input_):
        if not self.training:
            return F.log_softmax(input_, dim=-1)
        else:
            return input_


class BatchRNN(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type=nn.LSTM, bidirectional=False, batch_norm=True):
        super(BatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_norm = SequenceWise(nn.BatchNorm1d(input_size)) if batch_norm else None
        self.rnn = rnn_type(input_size=input_size, hidden_size=hidden_size,
                            bidirectional=bidirectional, bias=True)
        self.num_directions = 2 if bidirectional else 1

    def flatten_parameters(self):
        self.rnn.flatten_parameters()

    def forward(self, x, output_lengths):
        if self.batch_norm is not None:
            x = self.batch_norm(x)
        x = nn.utils.rnn.pack_padded_sequence(x, output_lengths)
        x, h = self.rnn(x)
        x, _ = nn.utils.rnn.pad_packed_sequence(x)
        if self.bidirectional:
            x = x.view(x.size(0), x.size(1), 2, -1).sum(2).view(x.size(0), x.size(1), -1)  # (TxNxH*2) -> (TxNxH) by sum
        return x


class Lookahead(nn.Module):
    # Wang et al 2016 - Lookahead Convolution Layer for Unidirectional Recurrent Neural Networks
    # input shape - sequence, batch, feature - TxNxH
    # output shape - same as input
    def __init__(self, n_features, context):
        super(Lookahead, self).__init__()
        assert context > 0
        self.context = context
        self.n_features = n_features
        self.pad = (0, self.context - 1)
        self.conv = nn.Conv1d(self.n_features, self.n_features, kernel_size=self.context, stride=1,
                              groups=self.n_features, padding=0, bias=None)

    def forward(self, x):
        x = x.transpose(0, 1).transpose(1, 2)
        x = F.pad(x, pad=self.pad, value=0)
        x = self.conv(x)
        x = x.transpose(1, 2).transpose(0, 1).contiguous()
        return x

    def __repr__(self):
        return self.__class__.__name__ + '(' \
               + 'n_features=' + str(self.n_features) \
               + ', context=' + str(self.context) + ')'


class DeepSpeech(nn.Module):
    def __init__(self, model_conf, audio_conf, labels):
        super(DeepSpeech, self).__init__()

        # model metadata needed for serialization/deserialization
        self.name = 'deepspeech'
        self.version = '2.0.0'
        self.model_conf = model_conf
        self.audio_conf = audio_conf
        self.labels = labels

        conv_channels = self.model_conf.getint('conv_channels', 8)
        self.frame_multiplier = self.model_conf.getint('frame_multiplier', 1)
        rnn_type = self.model_conf.get('rnn_type', 'lstm')
        hidden_size = self.model_conf.getint('hidden_size', 1024)
        hidden_layers = self.model_conf.getint('hidden_layers', 3)
        bidirectional = self.model_conf.getboolean('bidirectional', True)
        context = self.model_conf.getint('lookahead_context', 40)
        dropout = self.model_conf.getfloat('dropout', 0.0)
        batchnorm = self.model_conf.getboolean('batchnorm', True)

        # Read audio configuration
        input_format = self.audio_conf.get("input_format", 'stft')
        sample_rate = self.audio_conf.getint("sample_rate", 22050)
        window_size = self.audio_conf.getfloat("window_size", 0.09288)
        window_stride = self.audio_conf.getfloat("window_stride", 0.02322)
        window = windows.get(self.audio_conf['window'], windows['hamming'])
        bins_per_octave = self.audio_conf.getint("bins_per_octave")
        num_octaves = self.audio_conf.getint("num_octaves")

        num_classes = len(self.labels)

        if input_format == 'stft':
            cnn_kernel_sizes = [(9, 3), (3, 3)]
            cnn_strides = [(2, 1), (2, 1)]
            cnn_paddings = [(4, 1), (1, 1)]
            rnn_input_size = int(math.floor((sample_rate * window_size) / 2))
        else:
            cnn_kernel_sizes = [(3, 3), (3, 3)]
            cnn_strides = [(1, 1), (2, 1)]
            cnn_paddings = [(1, 1), (1, 1)]
            if input_format == 'cqt':
                rnn_input_size = bins_per_octave * num_octaves
            elif input_format == 'log':
                rnn_input_size = bins_per_octave * num_octaves - 2
            else:
                raise NotImplementedError

        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=cnn_kernel_sizes[0], stride=cnn_strides[0], padding=cnn_paddings[0]),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=cnn_kernel_sizes[1], stride=cnn_strides[1], padding=cnn_paddings[1]),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
        )) if batchnorm else MaskConv(nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=cnn_kernel_sizes[0], stride=cnn_strides[0], padding=cnn_paddings[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=cnn_kernel_sizes[1], stride=cnn_strides[1], padding=cnn_paddings[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        ))

        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor(rnn_input_size - cnn_kernel_sizes[0][0] + 2 * cnn_paddings[0][0]) / cnn_strides[0][0] + 1)
        rnn_input_size = int(math.floor(rnn_input_size - cnn_kernel_sizes[1][0] + 2 * cnn_paddings[1][0]) / cnn_strides[1][0] + 1)
        rnn_input_size *= conv_channels
        rnn_input_size = rnn_input_size // self.frame_multiplier
        
        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size, hidden_size=hidden_size, rnn_type=supported_rnns[rnn_type],
                       bidirectional=bidirectional, batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(hidden_layers - 1):
            rnn = BatchRNN(input_size=hidden_size, hidden_size=hidden_size, rnn_type=supported_rnns[rnn_type],
                           bidirectional=bidirectional, batch_norm=batchnorm)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(hidden_size, context=context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes, bias=False),
        ) if batchnorm else nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes, bias=False),
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, x_lengths):
        output_lengths = self.get_seq_lens(x_lengths)

        x = x.transpose(1,2).contiguous() # NxHxT
        x = x.unsqueeze(1) # Add channel dimension NxCxHxT

        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).contiguous() # NxTxH
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * self.frame_multiplier, sizes[2] // self.frame_multiplier)  # Expand time dimension
        x = x.transpose(0, 1).contiguous()  # TxNxH
        output_lengths *= self.frame_multiplier

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if self.lookahead:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()


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

    class TinyConfig:
        def __init__(self, values):
            self.values = values

        def get(self, key, default=None):
            return self.values.get(key, default)

        def getint(self, key, default=None):
            return int(self.values.get(key, default))

        def getfloat(self, key, default=None):
            return float(self.values.get(key, default))

        def getboolean(self, key, default=None):
            value = self.values.get(key, default)
            if isinstance(value, bool):
                return value
            return str(value).lower() in {"1", "yes", "true", "on"}

        def __getitem__(self, key):
            return self.values[key]

    print("audio2score benchmark checks")

    print("[Test 1/5] MaskConv.forward")
    try:
        mask_conv = MaskConv(nn.Sequential(nn.Identity()))
        x = torch.ones(2, 1, 3, 5)
        lengths = torch.tensor([5, 3])
        masked, returned_lengths = mask_conv(x, lengths)
        check("MaskConv output not None", masked is not None)
        if masked is not None:
            check("MaskConv output shape", tuple(masked.shape) == (2, 1, 3, 5), str(tuple(masked.shape)))
            check("MaskConv preserves lengths", torch.equal(returned_lengths, lengths), str(returned_lengths))
            check("MaskConv zeroes padded time steps", torch.all(masked[1, :, :, 3:] == 0).item(), str(masked[1]))
            check("MaskConv preserves valid time steps", torch.all(masked[0] == 1).item() and torch.all(masked[1, :, :, :3] == 1).item())
        else:
            skip_checks(4, "MaskConv returned None")
    except Exception as exc:
        skip_checks(5, f"MaskConv.forward raised {type(exc).__name__}: {exc}")

    print("[Test 2/5] BatchRNN.forward")
    try:
        batch_rnn = BatchRNN(input_size=4, hidden_size=6, rnn_type=nn.GRU, bidirectional=True, batch_norm=False)
        batch_rnn.eval()
        sequence = torch.randn(5, 2, 4)
        lengths = torch.tensor([5, 3])
        rnn_output = batch_rnn(sequence, lengths)
        check("BatchRNN output not None", rnn_output is not None)
        if rnn_output is not None:
            check("BatchRNN output shape", tuple(rnn_output.shape) == (5, 2, 6), str(tuple(rnn_output.shape)))
            check("BatchRNN output finite", torch.isfinite(rnn_output).all().item())
            check("BatchRNN bidirectional sum keeps hidden size", rnn_output.size(2) == 6, str(rnn_output.size(2)))
            check("BatchRNN packed sequence length", rnn_output.size(0) == int(lengths.max().item()), str(rnn_output.size(0)))
        else:
            skip_checks(4, "BatchRNN returned None")
    except Exception as exc:
        skip_checks(5, f"BatchRNN.forward raised {type(exc).__name__}: {exc}")

    print("[Test 3/5] Lookahead.forward")
    try:
        lookahead = Lookahead(n_features=3, context=2)
        lookahead.eval()
        sequence = torch.randn(4, 2, 3)
        looked = lookahead(sequence)
        check("Lookahead output not None", looked is not None)
        if looked is not None:
            check("Lookahead output shape", tuple(looked.shape) == (4, 2, 3), str(tuple(looked.shape)))
            check("Lookahead output finite", torch.isfinite(looked).all().item())
            check("Lookahead preserves time length", looked.size(0) == sequence.size(0), str(looked.size(0)))
        else:
            skip_checks(3, "Lookahead returned None")
    except Exception as exc:
        skip_checks(4, f"Lookahead.forward raised {type(exc).__name__}: {exc}")

    print("[Test 4/5] DeepSpeech.get_seq_lens")
    try:
        model_conf = TinyConfig({
            "conv_channels": 2,
            "frame_multiplier": 1,
            "rnn_type": "gru",
            "hidden_size": 8,
            "hidden_layers": 1,
            "bidirectional": True,
            "lookahead_context": 2,
            "dropout": 0.0,
            "batchnorm": False,
        })
        audio_conf = TinyConfig({
            "input_format": "log",
            "sample_rate": 22050,
            "window_size": 0.09288,
            "window_stride": 0.02322,
            "window": "hamming",
            "bins_per_octave": 4,
            "num_octaves": 5,
        })
        labels = list("_abcdef")
        model = DeepSpeech(model_conf, audio_conf, labels)
        input_lengths = torch.tensor([11, 7])
        output_lengths = model.get_seq_lens(input_lengths)
        check("get_seq_lens output not None", output_lengths is not None)
        if output_lengths is not None:
            check("get_seq_lens output shape", tuple(output_lengths.shape) == (2,), str(tuple(output_lengths.shape)))
            check("get_seq_lens uses integer dtype", not output_lengths.dtype.is_floating_point, str(output_lengths.dtype))
            check("get_seq_lens preserves log-time length for stride-one time convs",
                  torch.equal(output_lengths, torch.tensor([11, 7])), str(output_lengths))
        else:
            skip_checks(3, "get_seq_lens returned None")
    except Exception as exc:
        skip_checks(4, f"DeepSpeech.get_seq_lens raised {type(exc).__name__}: {exc}")

    print("[Test 5/5] DeepSpeech.forward")
    try:
        model_conf = TinyConfig({
            "conv_channels": 2,
            "frame_multiplier": 1,
            "rnn_type": "gru",
            "hidden_size": 8,
            "hidden_layers": 1,
            "bidirectional": True,
            "lookahead_context": 2,
            "dropout": 0.0,
            "batchnorm": False,
        })
        audio_conf = TinyConfig({
            "input_format": "log",
            "sample_rate": 22050,
            "window_size": 0.09288,
            "window_stride": 0.02322,
            "window": "hamming",
            "bins_per_octave": 4,
            "num_octaves": 5,
        })
        labels = list("_abcdef")
        model = DeepSpeech(model_conf, audio_conf, labels)
        model.eval()
        features = torch.randn(2, 11, 18)
        input_lengths = torch.tensor([11, 9])
        logits, logit_lengths = model(features, input_lengths)
        check("DeepSpeech output not None", logits is not None and logit_lengths is not None)
        if logits is not None and logit_lengths is not None:
            check("DeepSpeech logits shape", tuple(logits.shape) == (2, 11, len(labels)), str(tuple(logits.shape)))
            check("DeepSpeech output lengths", torch.equal(logit_lengths, torch.tensor([11, 9])), str(logit_lengths))
            check("DeepSpeech logits finite", torch.isfinite(logits).all().item())
            check("DeepSpeech eval outputs log probabilities",
                  torch.allclose(torch.exp(logits).sum(dim=-1), torch.ones_like(logits[:, :, 0]), atol=1e-5),
                  str(torch.exp(logits).sum(dim=-1)))
        else:
            skip_checks(4, "DeepSpeech.forward returned None")
    except Exception as exc:
        skip_checks(5, f"DeepSpeech.forward raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
