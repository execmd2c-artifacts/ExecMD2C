# ============================================================
# template.py - MultiWOZ Baseline Core Model Components
# Source:
#   Miscellaneous/multiwoz-master/model/model.py
#   Miscellaneous/multiwoz-master/model/policy.py
#
# Contains ONLY model architecture components and direct dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

from __future__ import division, print_function, unicode_literals

import math
import random
from functools import reduce
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# --- [Original file: model/model.py] ---
SOS_token = 0
EOS_token = 1
UNK_token = 2
PAD_token = 3


def init_lstm(cell, gain=1):
    init_gru(cell, gain)

    # positive forget gate bias (Jozefowicz et al., 2015)
    for _, _, ih_b, hh_b in cell.all_weights:
        l = len(ih_b)
        ih_b[l // 4:l // 2].data.fill_(1.0)
        hh_b[l // 4:l // 2].data.fill_(1.0)


def init_gru(gru, gain=1):
    gru.reset_parameters()
    for _, hh, _, _ in gru.all_weights:
        for i in range(0, hh.size(0), gru.hidden_size):
            torch.nn.init.orthogonal_(hh[i:i+gru.hidden_size],gain=gain)


def whatCellType(input_size, hidden_size, cell_type, dropout_rate):
    if cell_type == 'rnn':
        cell = nn.RNN(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'gru':
        cell = nn.GRU(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'lstm':
        cell = nn.LSTM(input_size, hidden_size, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell
    elif cell_type == 'bigru':
        cell = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_gru(cell)
        return cell
    elif cell_type == 'bilstm':
        cell = nn.LSTM(input_size, hidden_size, bidirectional=True, dropout=dropout_rate, batch_first=False)
        init_lstm(cell)
        return cell


class EncoderRNN(nn.Module):
    def __init__(self, input_size,  embedding_size, hidden_size, cell_type, depth, dropout):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.n_layers = depth
        self.dropout = dropout
        self.bidirectional = False
        if 'bi' in cell_type:
            self.bidirectional = True
        padding_idx = 3
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=padding_idx)
        self.rnn = whatCellType(embedding_size, hidden_size,
                    cell_type, dropout_rate=self.dropout)

    def forward(self, input_seqs, input_lens, hidden=None):
        """
        [TODO] Encode an unsorted batch of variable-length dialogue utterance tokens.

        Input:
            input_seqs: LongTensor with shape (batch, time), containing token ids.
            input_lens: sequence lengths with shape (batch,), not necessarily sorted.
            hidden: optional recurrent initial state matching the selected cell type.

        Output:
            outputs: Tensor with shape (time, batch, hidden_size), restored to the
                original batch order.
            hidden: final recurrent state restored to the original batch order; for
                LSTM this is a tuple of two tensors with shape
                (layers_or_directions, batch, hidden_size).

"""
        pass


class SeqAttnDecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout_p=0.1, max_length=30):
        super(SeqAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embedding_size
        self.output_size = output_size
        self.n_layers = 1
        self.dropout_p = dropout_p

        # Define layers
        self.embedding = nn.Embedding(output_size, embedding_size)
        self.dropout = nn.Dropout(dropout_p)

        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size + hidden_size, hidden_size, cell_type, dropout_rate=self.dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

        self.score = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.attn_combine = nn.Linear(embedding_size + hidden_size, embedding_size)

        # attention
        self.method = 'concat'
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, input, hidden, encoder_outputs):
        """
        [TODO] Run one attention-based decoder step for response generation.

        Input:
            input: LongTensor with shape (batch, 1), containing the previous
                decoder token for each dialogue.
            hidden: recurrent decoder state with shape (1, batch, hidden_size),
                or an LSTM tuple of two tensors with that shape.
            encoder_outputs: Tensor with shape (source_time, batch, hidden_size).

        Output:
            output: log-probability tensor with shape (batch, output_size).
            hidden: updated recurrent decoder state with the same structure and
                batch/hidden dimensions as the input hidden state.

"""
        pass


class DecoderRNN(nn.Module):
    def __init__(self, embedding_size, hidden_size, output_size, cell_type, dropout=0.1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type
        padding_idx = 3
        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      padding_idx=padding_idx
                                      )
        if 'bi' in cell_type:  # we dont need bidirectionality in decoding
            cell_type = cell_type.strip('bi')
        self.rnn = whatCellType(embedding_size, hidden_size, cell_type, dropout_rate=dropout)
        self.dropout_rate = dropout
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, not_used):
        embedded = self.embedding(input).transpose(0, 1)  # [B,1] -> [ 1,B, D]
        embedded = F.dropout(embedded, self.dropout_rate)

        output = embedded
        #output = F.relu(embedded)

        output, hidden = self.rnn(output, hidden)

        out = self.out(output.squeeze(0))
        output = F.log_softmax(out, dim=1)

        return output, hidden


# --- [Original file: model/policy.py] ---
class DefaultPolicy(nn.Module):
    def __init__(self, hidden_size_pol, hidden_size, db_size, bs_size):
        super(DefaultPolicy, self).__init__()
        self.hidden_size = hidden_size


        self.W_u = nn.Linear(hidden_size, hidden_size_pol, bias=False)
        self.W_bs = nn.Linear(bs_size, hidden_size_pol, bias=False)
        self.W_db = nn.Linear(db_size, hidden_size_pol, bias=False)

    def forward(self, encodings, db_tensor, bs_tensor, act_tensor=None):
        """
        [TODO] Build the decoder's initial policy state from dialogue context features.

        Input:
            encodings: encoder final hidden state, either a tensor with shape
                (1, batch, hidden_size) or an LSTM tuple whose first element has
                that shape.
            db_tensor: database pointer/features with shape (batch, db_size).
            bs_tensor: belief-state features with shape (batch, bs_size).
            act_tensor: optional action tensor, unused by this default policy.

        Output:
            For non-LSTM encodings, a tensor with shape
                (1, batch, hidden_size_pol).
            For LSTM encodings, a tuple whose first item has shape
                (1, batch, hidden_size_pol) and whose second item preserves the
                original cell state.

"""
        pass


# --- [Original file: model/model.py] ---
class Model(nn.Module):
    def __init__(self, args, input_lang_index2word, output_lang_index2word, input_lang_word2index, output_lang_word2index):
        super(Model, self).__init__()
        self.args = args
        self.max_len = args.max_len

        self.output_lang_index2word = output_lang_index2word
        self.input_lang_index2word = input_lang_index2word

        self.output_lang_word2index = output_lang_word2index
        self.input_lang_word2index = input_lang_word2index

        self.hid_size_enc = args.hid_size_enc
        self.hid_size_dec = args.hid_size_dec
        self.hid_size_pol = args.hid_size_pol

        self.emb_size = args.emb_size
        self.db_size = args.db_size
        self.bs_size = args.bs_size
        self.cell_type = args.cell_type
        if 'bi' in self.cell_type:
            self.num_directions = 2
        else:
            self.num_directions = 1
        self.depth = args.depth
        self.use_attn = args.use_attn
        self.attn_type = args.attention_type

        self.dropout = args.dropout
        self.device = torch.device("cuda" if args.cuda else "cpu")

        self.model_dir = args.model_dir
        self.model_name = args.model_name
        self.teacher_forcing_ratio = args.teacher_ratio
        self.vocab_size = args.vocab_size
        self.epsln = 10E-5


        torch.manual_seed(args.seed)
        self.build_model()
        self.getCount()
        try:
            assert self.args.beam_width > 0
            self.beam_search = True
        except:
            self.beam_search = False

        self.global_step = 0

    def cuda_(self, var):
        return var.cuda() if self.args.cuda else var

    def build_model(self):
        self.encoder = EncoderRNN(len(self.input_lang_index2word), self.emb_size, self.hid_size_enc,
                                  self.cell_type, self.depth, self.dropout).to(self.device)

        self.policy = DefaultPolicy(self.hid_size_pol, self.hid_size_enc, self.db_size, self.bs_size).to(self.device)

        if self.use_attn:
            if self.attn_type == 'bahdanau':
                self.decoder = SeqAttnDecoderRNN(self.emb_size, self.hid_size_dec, len(self.output_lang_index2word), self.cell_type, self.dropout, self.max_len).to(self.device)
        else:
            self.decoder = DecoderRNN(self.emb_size, self.hid_size_dec, len(self.output_lang_index2word), self.cell_type, self.dropout).to(self.device)

        if self.args.mode == 'train':
            self.gen_criterion = nn.NLLLoss(ignore_index=3, size_average=True)  # logsoftmax is done in decoder part
            self.setOptimizers()

    def forward(self, input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor):
        """
        [TODO] Run the MultiWOZ baseline encoder-policy-decoder forward path.

        Input:
            input_tensor: LongTensor with shape (batch, source_time), containing
                delexicalized user/system context token ids.
            input_lengths: source sequence lengths with shape (batch,).
            target_tensor: LongTensor with shape (batch, target_time), containing
                response token ids used for teacher forcing.
            target_lengths: target sequence lengths with shape (batch,).
            db_tensor: database pointer/features with shape (batch, db_size).
            bs_tensor: belief-state features with shape (batch, bs_size).

        Output:
            proba: tensor with shape (batch, target_time, vocab_size), containing
                log probabilities for every generated timestep.
            second return value: placeholder matching the original interface.
            decoded_sent: placeholder matching the original interface.

"""
        pass

    def setOptimizers(self):
        self.optimizer_policy = None
        if self.args.optim == 'sgd':
            self.optimizer = optim.SGD(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adadelta':
            self.optimizer = optim.Adadelta(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)
        elif self.args.optim == 'adam':
            self.optimizer = optim.Adam(lr=self.args.lr_rate, params=filter(lambda x: x.requires_grad, self.parameters()), weight_decay=self.args.l2_norm)

    def getCount(self):
        learnable_parameters = filter(lambda p: p.requires_grad, self.parameters())
        param_cnt = sum([reduce((lambda x, y: x * y), param.shape) for param in learnable_parameters])
        print('Model has', param_cnt, ' parameters.')


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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
    print("MultiWOZ baseline encoder-policy-decoder model")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: EncoderRNN.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] EncoderRNN.forward - unsorted variable-length encoding")
    try:
        encoder = EncoderRNN(input_size=20, embedding_size=6, hidden_size=8, cell_type="lstm", depth=1, dropout=0.0).to(device)
        input_tensor = torch.tensor([
            [4, 5, 6, 7, 8],
            [9, 10, 11, PAD_token, PAD_token],
            [12, 13, 14, 15, PAD_token],
        ], dtype=torch.long, device=device)
        lengths = np.array([5, 3, 4])
        with torch.no_grad():
            outputs, hidden = encoder(input_tensor, lengths)
        check("Encoder output not None", outputs is not None)
        if outputs is not None:
            padded_tail_zero = torch.allclose(outputs[3:, 1, :], torch.zeros_like(outputs[3:, 1, :]), atol=1e-6)
            hidden_shape_ok = isinstance(hidden, tuple) and tuple(hidden[0].shape) == (1, 3, 8) and tuple(hidden[1].shape) == (1, 3, 8)
            check("Encoder output shape", tuple(outputs.shape) == (5, 3, 8), f"expected (5, 3, 8), got {tuple(outputs.shape)}")
            check("Encoder hidden shape", hidden_shape_ok)
            check("Encoder output finite", torch.isfinite(outputs).all().item())
            check("Encoder restores unsorted batch with padded tail zeros", padded_tail_zero)
        else:
            skip_checks(4, "EncoderRNN.forward returned None")
    except Exception as e:
        skip_checks(5, f"EncoderRNN.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: SeqAttnDecoderRNN.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] SeqAttnDecoderRNN.forward - attention decoder step")
    try:
        decoder = SeqAttnDecoderRNN(embedding_size=6, hidden_size=8, output_size=15, cell_type="lstm", dropout_p=0.0, max_length=6).to(device)
        decoder_input = torch.tensor([[SOS_token], [4], [5]], dtype=torch.long, device=device)
        hidden = (torch.randn(1, 3, 8, device=device), torch.randn(1, 3, 8, device=device))
        encoder_outputs = torch.randn(5, 3, 8, device=device)
        with torch.no_grad():
            output, next_hidden = decoder(decoder_input, hidden, encoder_outputs)
        check("Attention decoder output not None", output is not None)
        if output is not None:
            probs_sum_to_one = torch.allclose(output.exp().sum(dim=1), torch.ones(3, device=device), atol=1e-5)
            next_hidden_ok = isinstance(next_hidden, tuple) and tuple(next_hidden[0].shape) == (1, 3, 8) and tuple(next_hidden[1].shape) == (1, 3, 8)
            check("Attention decoder output shape", tuple(output.shape) == (3, 15), f"expected (3, 15), got {tuple(output.shape)}")
            check("Attention decoder log probabilities finite", torch.isfinite(output).all().item())
            check("Attention decoder probabilities normalize by vocabulary", probs_sum_to_one)
            check("Attention decoder returns recurrent hidden state", next_hidden_ok)
        else:
            skip_checks(4, "SeqAttnDecoderRNN.forward returned None")
    except Exception as e:
        skip_checks(5, f"SeqAttnDecoderRNN.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: DefaultPolicy.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] DefaultPolicy.forward - belief and database conditioned policy")
    try:
        policy = DefaultPolicy(hidden_size_pol=8, hidden_size=8, db_size=4, bs_size=5).to(device)
        enc_h = torch.randn(1, 3, 8, device=device)
        enc_c = torch.randn(1, 3, 8, device=device)
        db_tensor = torch.randn(3, 4, device=device)
        bs_tensor = torch.randn(3, 5, device=device)
        with torch.no_grad():
            policy_hidden = policy((enc_h, enc_c), db_tensor, bs_tensor)
            expected = torch.tanh(policy.W_u(enc_h[0]) + policy.W_db(db_tensor) + policy.W_bs(bs_tensor)).unsqueeze(0)
        check("Policy output not None", policy_hidden is not None)
        if policy_hidden is not None:
            shape_ok = isinstance(policy_hidden, tuple) and tuple(policy_hidden[0].shape) == (1, 3, 8)
            carries_cell = isinstance(policy_hidden, tuple) and torch.allclose(policy_hidden[1], enc_c)
            check("Policy output shape", shape_ok)
            check("Policy output finite", torch.isfinite(policy_hidden[0]).all().item() if isinstance(policy_hidden, tuple) else False)
            check("Policy preserves LSTM cell state", carries_cell)
            check("Policy fuses hidden, DB, and belief-state inputs", torch.allclose(policy_hidden[0], expected, atol=1e-6) if isinstance(policy_hidden, tuple) else False)
        else:
            skip_checks(4, "DefaultPolicy.forward returned None")
    except Exception as e:
        skip_checks(5, f"DefaultPolicy.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: Model.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] Model.forward - encoder-policy-decoder generation path")
    try:
        vocab_size = 18
        index2word = {str(i): f"tok_{i}" for i in range(vocab_size)}
        word2index = {f"tok_{i}": i for i in range(vocab_size)}
        args = SimpleNamespace(
            max_len=6,
            hid_size_enc=8,
            hid_size_dec=8,
            hid_size_pol=8,
            emb_size=6,
            db_size=4,
            bs_size=5,
            cell_type="lstm",
            depth=1,
            use_attn=True,
            attention_type="bahdanau",
            dropout=0.0,
            cuda=False,
            model_dir="",
            model_name="",
            teacher_ratio=1.0,
            vocab_size=vocab_size,
            seed=42,
            mode="test",
            beam_width=0,
        )
        model = Model(args, index2word, index2word, word2index, word2index).to(device)
        input_tensor = torch.tensor([
            [4, 5, 6, 7, 8],
            [9, 10, 11, PAD_token, PAD_token],
        ], dtype=torch.long, device=device)
        input_lengths = np.array([5, 3])
        target_tensor = torch.tensor([
            [4, 5, EOS_token, PAD_token],
            [6, 7, 8, EOS_token],
        ], dtype=torch.long, device=device)
        target_lengths = np.array([3, 4])
        db_tensor = torch.randn(2, 4, device=device)
        bs_tensor = torch.randn(2, 5, device=device)
        with torch.no_grad():
            proba, _, decoded_sent = model(input_tensor, input_lengths, target_tensor, target_lengths, db_tensor, bs_tensor)
        check("Model forward output not None", proba is not None)
        if proba is not None:
            normalized = torch.allclose(proba.exp().sum(dim=2), torch.ones(2, 4, device=device), atol=1e-5)
            check("Model forward probability tensor shape", tuple(proba.shape) == (2, 4, vocab_size), f"expected (2, 4, {vocab_size}), got {tuple(proba.shape)}")
            check("Model forward output finite", torch.isfinite(proba).all().item())
            check("Model forward returns log probabilities at every target step", normalized)
            check("Model forward keeps decoded sentence placeholder", decoded_sent is None)
        else:
            skip_checks(4, "Model.forward returned None")
    except Exception as e:
        skip_checks(5, f"Model.forward raised {type(e).__name__}: {e}")
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
