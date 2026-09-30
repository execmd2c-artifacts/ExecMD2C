"""
ground_truth.py for Transformer Hawkes Process core model components.

Source-consolidated from:
- transformer/Constants.py
- transformer/Modules.py
- transformer/SubLayers.py
- transformer/Layers.py
- transformer/Models.py
- Utils.py

Only the neural temporal point-process architecture, attention blocks, intensity
functions, and direct prediction/loss helpers are included. Experiment drivers,
data preparation, parameter-update procedures, saved-state I/O, and CLI code are
intentionally excluded.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


PAD = 0


# --- [Original file: transformer/Modules.py] ---
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        """
        TODO: Compute scaled dot-product attention with optional masking.

        Inputs:
            q: Query tensor with shape (batch, n_head, len_q, d_k).
            k: Key tensor with shape (batch, n_head, len_k, d_k).
            v: Value tensor with shape (batch, n_head, len_v, d_v).
            mask: Optional boolean-like tensor broadcastable to
                  (batch, n_head, len_q, len_k); True entries are blocked.
        Output:
            Tuple (output, attn), where output has shape
            (batch, n_head, len_q, d_v) and attn has shape
            (batch, n_head, len_q, len_k).

"""
        pass


# --- [Original file: transformer/SubLayers.py] ---
class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention module """

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5, attn_dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        """
        TODO: Apply Transformer multi-head self-attention with residual output.

        Inputs:
            q: Query states with shape (batch, len_q, d_model).
            k: Key states with shape (batch, len_k, d_model).
            v: Value states with shape (batch, len_v, d_model).
            mask: Optional tensor with shape (batch, len_q, len_k), blocking
                  future or padded keys.
        Output:
            Tuple (output, attn), where output has shape
            (batch, len_q, d_model) and attn has shape
            (batch, n_head, len_q, len_k).

"""
        pass


class PositionwiseFeedForward(nn.Module):
    """ Two-layer position-wise feed-forward neural network. """

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


# --- [Original file: transformer/Layers.py] ---
class EncoderLayer(nn.Module):
    """ Compose with two layers """

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, normalize_before=normalize_before)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        """
        TODO: Run one THP encoder block with attention, FFN, and padding masks.

        Inputs:
            enc_input: Tensor with shape (batch, seq_len, d_model).
            non_pad_mask: Tensor with shape (batch, seq_len, 1), with zero at
                          padded event positions.
            slf_attn_mask: Tensor with shape (batch, seq_len, seq_len), blocking
                           padded keys and future events.
        Output:
            Tuple (enc_output, enc_slf_attn), where enc_output has shape
            (batch, seq_len, d_model) and enc_slf_attn has shape
            (batch, n_head, seq_len, seq_len).

"""
        pass


# --- [Original file: transformer/Models.py] ---
def get_non_pad_mask(seq):
    """ Get the non-padding positions. """

    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)


def get_attn_key_pad_mask(seq_k, seq_q):
    """ For masking out the padding part of key sequence. """

    # expand to fit the shape of key query attention matrix
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk
    return padding_mask


def get_subsequent_mask(seq):
    """ For masking out the subsequent info, i.e., masked self-attention. """

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls
    return subsequent_mask


class Encoder(nn.Module):
    """ A encoder model with self attention mechanism. """

    def __init__(
            self,
            num_types, d_model, d_inner,
            n_layers, n_head, d_k, d_v, dropout):
        super().__init__()

        self.d_model = d_model

        # position vector, used for temporal encoding
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=torch.device('cuda'))

        # event type embedding
        self.event_emb = nn.Embedding(num_types + 1, d_model, padding_idx=PAD)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout, normalize_before=False)
            for _ in range(n_layers)])

    def temporal_enc(self, time, non_pad_mask):
        """
        TODO: Encode continuous event times into sinusoidal THP features.

        Inputs:
            time: Tensor with shape (batch, seq_len), containing event times.
            non_pad_mask: Tensor with shape (batch, seq_len, 1), with zero at
                          padded event positions.
        Output:
            Tensor with shape (batch, seq_len, d_model).

"""
        pass

    def forward(self, event_type, event_time, non_pad_mask):
        """
        TODO: Encode event types and times with causal Transformer layers.

        Inputs:
            event_type: Long tensor with shape (batch, seq_len), using zero for
                        padding and positive ids for event types.
            event_time: Tensor with shape (batch, seq_len), containing event
                        timestamps.
            non_pad_mask: Tensor with shape (batch, seq_len, 1).
        Output:
            Encoded sequence tensor with shape (batch, seq_len, d_model).

"""
        pass


class Predictor(nn.Module):
    """ Prediction of next event type. """

    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out


class RNN_layers(nn.Module):
    """
    Optional recurrent layers. This is inspired by the fact that adding
    recurrent layers on top of the Transformer helps language modeling.
    """

    def __init__(self, d_model, d_rnn):
        super().__init__()

        self.rnn = nn.LSTM(d_model, d_rnn, num_layers=1, batch_first=True)
        self.projection = nn.Linear(d_rnn, d_model)

    def forward(self, data, non_pad_mask):
        lengths = non_pad_mask.squeeze(2).long().sum(1).cpu()
        pack_enc_output = nn.utils.rnn.pack_padded_sequence(
            data, lengths, batch_first=True, enforce_sorted=False)
        temp = self.rnn(pack_enc_output)[0]
        out = nn.utils.rnn.pad_packed_sequence(temp, batch_first=True)[0]

        out = self.projection(out)
        return out


class Transformer(nn.Module):
    """ A sequence to sequence model with attention mechanism. """

    def __init__(
            self,
            num_types, d_model=256, d_rnn=128, d_inner=1024,
            n_layers=4, n_head=4, d_k=64, d_v=64, dropout=0.1):
        super().__init__()

        self.encoder = Encoder(
            num_types=num_types,
            d_model=d_model,
            d_inner=d_inner,
            n_layers=n_layers,
            n_head=n_head,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
        )

        self.num_types = num_types

        # convert hidden vectors into a scalar
        self.linear = nn.Linear(d_model, num_types)

        # parameter for the weight of time difference
        self.alpha = nn.Parameter(torch.tensor(-0.1))

        # parameter for the softplus function
        self.beta = nn.Parameter(torch.tensor(1.0))

        # OPTIONAL recurrent layer, this sometimes helps
        self.rnn = RNN_layers(d_model, d_rnn)

        # prediction of next time stamp
        self.time_predictor = Predictor(d_model, 1)

        # prediction of next event type
        self.type_predictor = Predictor(d_model, num_types)

    def forward(self, event_type, event_time):
        """
        TODO: Run the full Transformer Hawkes Process forward pass.

        Inputs:
            event_type: Long tensor with shape (batch, seq_len), using zero for
                        padding and positive event-type ids otherwise.
            event_time: Tensor with shape (batch, seq_len), containing event
                        timestamps aligned with event_type.
        Output:
            Tuple (enc_output, (type_prediction, time_prediction)):
                enc_output has shape (batch, valid_len, d_model) after the
                optional recurrent layer restores non-padded sequences.
                type_prediction has shape (batch, valid_len, num_types).
                time_prediction has shape (batch, valid_len, 1).

"""
        pass


# --- [Original file: Utils.py] ---
def softplus(x, beta):
    """
    TODO: Compute the THP intensity activation with hard thresholding.

    Inputs:
        x: Tensor of arbitrary shape containing raw intensity values.
        beta: Positive scalar tensor or parameter controlling softness.
    Output:
        Tensor with the same shape as x.

"""
    pass


def compute_event(event, non_pad_mask):
    """ Log-likelihood of events. """

    # add 1e-9 in case some events have 0 likelihood
    event += math.pow(10, -9)
    event.masked_fill_(~non_pad_mask.bool(), 1.0)

    result = torch.log(event)
    return result


def compute_integral_biased(all_lambda, time, non_pad_mask):
    """ Log-likelihood of non-events, using linear interpolation. """

    diff_time = (time[:, 1:] - time[:, :-1]) * non_pad_mask[:, 1:]
    diff_lambda = (all_lambda[:, 1:] + all_lambda[:, :-1]) * non_pad_mask[:, 1:]

    biased_integral = diff_lambda * diff_time
    result = 0.5 * biased_integral
    return result


def compute_integral_unbiased(model, data, time, non_pad_mask, type_mask):
    """
    TODO: Estimate the non-event log-likelihood integral with Monte Carlo time samples.

    Inputs:
        model: THP-like module exposing linear, alpha, beta, and num_types.
        data: Hidden sequence tensor with shape (batch, seq_len, d_model).
        time: Timestamp tensor with shape (batch, seq_len).
        non_pad_mask: Tensor with shape (batch, seq_len), masking padded events.
        type_mask: One-hot event-type tensor with shape
                   (batch, seq_len, num_types).
    Output:
        Tensor with shape (batch, seq_len - 1), containing non-event integral
        estimates for intervals between consecutive events.

"""
    pass


def log_likelihood(model, data, time, types):
    """
    TODO: Compute event and non-event log-likelihood terms for a THP sequence.

    Inputs:
        model: THP model exposing linear, beta, alpha, and num_types.
        data: Hidden sequence tensor with shape (batch, seq_len, d_model).
        time: Timestamp tensor with shape (batch, seq_len).
        types: Long tensor with shape (batch, seq_len), using zero for padding
               and one-based ids for observed event types.
    Output:
        Tuple (event_ll, non_event_ll), both with shape (batch,).

"""
    pass


def type_loss(prediction, types, loss_func):
    """ Event prediction loss, cross entropy or label smoothing. """

    # convert [1,2,3] based types to [0,1,2]; also convert padding events to -1
    truth = types[:, 1:] - 1
    prediction = prediction[:, :-1, :]

    pred_type = torch.max(prediction, dim=-1)[1]
    correct_num = torch.sum(pred_type == truth)

    # compute cross entropy loss
    if isinstance(loss_func, LabelSmoothingLoss):
        loss = loss_func(prediction, truth)
    else:
        loss = loss_func(prediction.transpose(1, 2), truth)

    loss = torch.sum(loss)
    return loss, correct_num


def time_loss(prediction, event_time):
    """ Time prediction loss. """

    prediction.squeeze_(-1)

    true = event_time[:, 1:] - event_time[:, :-1]
    prediction = prediction[:, :-1]

    # event time gap prediction
    diff = prediction - true
    se = torch.sum(diff * diff)
    return se


class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        super(LabelSmoothingLoss, self).__init__()

        self.eps = label_smoothing
        self.num_classes = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, output, target):
        """
        output (FloatTensor): (batch_size) x n_classes
        target (LongTensor): batch_size
        """

        non_pad_mask = target.ne(self.ignore_index).float()

        target[target.eq(self.ignore_index)] = 0
        one_hot = F.one_hot(target, num_classes=self.num_classes).float()
        one_hot = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps / self.num_classes

        log_prb = F.log_softmax(output, dim=-1)
        loss = -(one_hot * log_prb).sum(dim=-1)
        loss = loss * non_pad_mask
        return loss


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
    print("Transformer Hawkes Process core model benchmark")
    print("=" * 70)

    # ==========================================================
    # Test 1: mask helpers
    # ==========================================================
    print("[Test 1/7] padding and causal masks")
    try:
        seq = torch.tensor([[1, 2, 0, 0], [3, 1, 2, 0]], dtype=torch.long)
        non_pad = get_non_pad_mask(seq)
        key_mask = get_attn_key_pad_mask(seq, seq)
        sub_mask = get_subsequent_mask(seq)
        check("non-pad mask shape", tuple(non_pad.shape) == (2, 4, 1), f"got {tuple(non_pad.shape)}")
        check("non-pad mask zeroes padding", non_pad[0, 2, 0].item() == 0 and non_pad[1, 2, 0].item() == 1)
        check("key padding mask shape", tuple(key_mask.shape) == (2, 4, 4), f"got {tuple(key_mask.shape)}")
        check("key padding mask marks PAD", key_mask[0, 0, 2].item() == 1 and key_mask[0, 0, 1].item() == 0)
        check("subsequent mask shape", tuple(sub_mask.shape) == (2, 4, 4), f"got {tuple(sub_mask.shape)}")
        check("subsequent mask is causal", sub_mask[0, 0, 1].item() == 1 and sub_mask[0, 1, 0].item() == 0)
    except Exception as exc:
        skip_checks(6, f"mask helpers raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2: ScaledDotProductAttention
    # ==========================================================
    print("[Test 2/7] scaled dot-product attention")
    try:
        attn_layer = ScaledDotProductAttention(temperature=2.0, attn_dropout=0.0)
        q = torch.randn(2, 2, 3, 4)
        k = torch.randn(2, 2, 3, 4)
        v = torch.randn(2, 2, 3, 5)
        mask = torch.zeros(2, 1, 3, 3, dtype=torch.bool)
        mask[:, :, :, 2] = True
        output, attn = attn_layer(q, k, v, mask=mask)
        check("attention output not None", output is not None)
        if output is not None:
            check("attention output shape", tuple(output.shape) == (2, 2, 3, 5), f"got {tuple(output.shape)}")
            check("attention weights shape", tuple(attn.shape) == (2, 2, 3, 3), f"got {tuple(attn.shape)}")
            check("attention output finite", torch.isfinite(output).all().item())
            check("masked attention nearly zero", torch.allclose(attn[..., 2], torch.zeros_like(attn[..., 2]), atol=1e-6))
        else:
            skip_checks(4, "attention returned None")
    except Exception as exc:
        skip_checks(5, f"scaled attention raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3: MultiHeadAttention and EncoderLayer
    # ==========================================================
    print("[Test 3/7] multi-head attention and encoder layer")
    try:
        mha = MultiHeadAttention(n_head=2, d_model=8, d_k=4, d_v=4, dropout=0.0, normalize_before=False)
        x = torch.randn(2, 4, 8)
        mask = torch.triu(torch.ones(4, 4, dtype=torch.bool), diagonal=1).unsqueeze(0).expand(2, -1, -1)
        out, attn = mha(x, x, x, mask=mask)
        check("MHA output not None", out is not None)
        if out is not None:
            check("MHA output shape", tuple(out.shape) == (2, 4, 8), f"got {tuple(out.shape)}")
            check("MHA attention shape", tuple(attn.shape) == (2, 2, 4, 4), f"got {tuple(attn.shape)}")
            check("MHA output finite", torch.isfinite(out).all().item())
        else:
            skip_checks(3, "MHA returned None")
        layer = EncoderLayer(d_model=8, d_inner=16, n_head=2, d_k=4, d_v=4, dropout=0.0, normalize_before=False)
        non_pad = torch.tensor([[[1.0], [1.0], [1.0], [0.0]], [[1.0], [1.0], [0.0], [0.0]]])
        enc_out, enc_attn = layer(x, non_pad_mask=non_pad, slf_attn_mask=mask)
        check("encoder layer output shape", tuple(enc_out.shape) == (2, 4, 8), f"got {tuple(enc_out.shape)}")
        check("encoder layer pads zeroed", torch.allclose(enc_out[0, 3], torch.zeros_like(enc_out[0, 3]), atol=1e-6))
    except Exception as exc:
        skip_checks(6, f"MHA/EncoderLayer raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4: Encoder temporal encoding and forward path
    # ==========================================================
    print("[Test 4/7] temporal encoder and causal event encoding")
    try:
        if not torch.cuda.is_available():
            skip_checks(7, "source Encoder stores position_vec on CUDA")
        else:
            device = torch.device("cuda")
            encoder = Encoder(num_types=3, d_model=8, d_inner=16, n_layers=1, n_head=2, d_k=4, d_v=4, dropout=0.0).to(device)
            event_type = torch.tensor([[1, 2, 3, 0], [2, 1, 0, 0]], dtype=torch.long, device=device)
            event_time = torch.tensor([[0.1, 0.4, 0.9, 0.0], [0.2, 0.7, 0.0, 0.0]], dtype=torch.float, device=device)
            non_pad = get_non_pad_mask(event_type)
            tem = encoder.temporal_enc(event_time, non_pad)
            enc = encoder(event_type, event_time, non_pad)
            check("temporal encoding not None", tem is not None)
            check("temporal encoding shape", tuple(tem.shape) == (2, 4, 8), f"got {tuple(tem.shape)}")
            check("temporal encoding pads zeroed", torch.allclose(tem[0, 3], torch.zeros_like(tem[0, 3]), atol=1e-6))
            check("encoder output not None", enc is not None)
            check("encoder output shape", tuple(enc.shape) == (2, 4, 8), f"got {tuple(enc.shape)}")
            check("encoder output finite", torch.isfinite(enc).all().item())
            check("encoder output pads zeroed", torch.allclose(enc[0, 3], torch.zeros_like(enc[0, 3]), atol=1e-6))
    except Exception as exc:
        skip_checks(7, f"Encoder raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5: full Transformer forward
    # ==========================================================
    print("[Test 5/7] Transformer forward predictions")
    try:
        if not torch.cuda.is_available():
            skip_checks(6, "source Transformer encoder stores position_vec on CUDA")
        else:
            device = torch.device("cuda")
            model = Transformer(num_types=3, d_model=8, d_rnn=6, d_inner=16, n_layers=1, n_head=2, d_k=4, d_v=4, dropout=0.0).to(device)
            event_type = torch.tensor([[1, 2, 3, 1], [2, 1, 3, 2]], dtype=torch.long, device=device)
            event_time = torch.tensor([[0.1, 0.4, 0.9, 1.0], [0.2, 0.7, 1.3, 1.4]], dtype=torch.float, device=device)
            enc_out, (type_pred, time_pred) = model(event_type, event_time)
            check("Transformer enc_out not None", enc_out is not None)
            check("Transformer enc_out shape", tuple(enc_out.shape) == (2, 4, 8), f"got {tuple(enc_out.shape)}")
            check("type prediction shape", tuple(type_pred.shape) == (2, 4, 3), f"got {tuple(type_pred.shape)}")
            check("time prediction shape", tuple(time_pred.shape) == (2, 4, 1), f"got {tuple(time_pred.shape)}")
            check("Transformer outputs finite", torch.isfinite(enc_out).all().item() and torch.isfinite(type_pred).all().item())
            check("intensity parameters exist", model.alpha.requires_grad and model.beta.requires_grad)
    except Exception as exc:
        skip_checks(6, f"Transformer forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6: THP softplus and likelihood components
    # ==========================================================
    print("[Test 6/7] intensity softplus and log-likelihood")
    try:
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.num_types = 3
                self.linear = nn.Linear(8, 3)
                self.alpha = nn.Parameter(torch.tensor(-0.1))
                self.beta = nn.Parameter(torch.tensor(1.0))

        model = TinyModel()
        data = torch.randn(2, 4, 8)
        time = torch.tensor([[0.1, 0.4, 0.9, 1.3], [0.2, 0.5, 0.8, 1.0]], dtype=torch.float)
        types = torch.tensor([[1, 2, 3, 0], [3, 1, 2, 0]], dtype=torch.long)
        sp = softplus(torch.tensor([[-100.0, 0.0, 100.0]]), torch.tensor(1.0))
        event_ll, non_event_ll = log_likelihood(model, data, time, types)
        check("softplus output shape", tuple(sp.shape) == (1, 3), f"got {tuple(sp.shape)}")
        check("softplus finite", torch.isfinite(sp).all().item())
        check("softplus clamps large values", sp[0, 2].item() < 21.0)
        check("event likelihood shape", tuple(event_ll.shape) == (2,), f"got {tuple(event_ll.shape)}")
        check("non-event likelihood shape", tuple(non_event_ll.shape) == (2,), f"got {tuple(non_event_ll.shape)}")
        check("likelihood finite", torch.isfinite(event_ll).all().item() and torch.isfinite(non_event_ll).all().item())
        check("non-event likelihood nonnegative", torch.all(non_event_ll >= 0).item())
    except Exception as exc:
        skip_checks(7, f"likelihood functions raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 7: prediction helper losses
    # ==========================================================
    print("[Test 7/7] prediction helper losses")
    try:
        type_prediction = torch.randn(2, 4, 3)
        event_type = torch.tensor([[1, 2, 3, 0], [2, 1, 3, 0]], dtype=torch.long)
        ce = nn.CrossEntropyLoss(ignore_index=-1, reduction="none")
        loss, correct = type_loss(type_prediction, event_type, ce)
        time_prediction = torch.randn(2, 4, 1)
        event_time = torch.tensor([[0.1, 0.4, 0.9, 1.3], [0.2, 0.5, 0.8, 1.0]], dtype=torch.float)
        se = time_loss(time_prediction.clone(), event_time)
        smooth = LabelSmoothingLoss(0.1, 3, ignore_index=-1)
        smooth_loss = smooth(torch.randn(2, 3, 3), torch.tensor([[0, 1, -1], [2, 0, 1]], dtype=torch.long))
        check("type loss scalar", loss.dim() == 0)
        check("type loss finite", torch.isfinite(loss).item())
        check("correct prediction count scalar", correct.dim() == 0)
        check("time loss scalar", se.dim() == 0)
        check("time loss finite", torch.isfinite(se).item())
        check("label smoothing output shape", tuple(smooth_loss.shape) == (2, 3), f"got {tuple(smooth_loss.shape)}")
        check("label smoothing masks ignore index", smooth_loss[0, 2].item() == 0.0)
    except Exception as exc:
        skip_checks(7, f"prediction helper losses raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some TODO functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
