# ============================================================
# ground_truth.py - TranCS Core Model Components
# Source: Computer_Code/TranCS-main
#
# Contains ONLY the TranCS neural architecture components and
# direct dependencies. No training, evaluation, data loading,
# preprocessing, baseline, optimizer, or scheduler code.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# --- [Original file: src/models/Modules.py] ---

class SeqEncoder_LSTM(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size, n_layers, embedding):
        super(SeqEncoder_LSTM, self).__init__()
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = embedding

        self.lstm = nn.LSTM(emb_size, hidden_size, dropout=0, batch_first=True, bidirectional=False)

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        return (weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_(),
                weight.new(self.n_layers, batch_size, self.hidden_size).zero_().requires_grad_())

    def forward(self, inputs, input_lens=None, hidden=None):
        """
        [TODO] Encode a padded token batch with an LSTM while preserving original batch order.

        Input:
            inputs: (batch, seq_len) - token ids for a sequence modality.
            input_lens: (batch,) or None - true sequence lengths before padding.
            hidden: tuple or None - initial LSTM hidden and cell states shaped
                `(num_layers, batch, hidden_size)`.

        Output: tuple `(hids, (h_n, c_n))`.
            hids: (batch, max_valid_len, hidden_size) when lengths are provided, otherwise
                (batch, seq_len, hidden_size).
            h_n: (batch, hidden_size) - final hidden state for each original example.
            c_n: (batch, hidden_size) - final cell state for each original example.

"""
        pass


# --- [Original file: src/models/TranCS.py] ---

class TranEmbeder(nn.Module):
    def __init__(self, config):
        super(TranEmbeder, self).__init__()

        self.conf = config

        self.n_tran_words = config['n_tran_words']
        self.n_doc_words = config['n_doc_words']
        self.n_tran_doc_words = config['n_tran_doc_words']

        self.code_nn = config['code_nn']
        self.doc_nn = config['doc_nn']
        self.mode = config['mode']

        self.margin = config['margin']
        self.dropout = config['dropout']

        self.emb_size = config['emb_size']
        self.n_hidden = config['n_hidden']
        self.n_layers_LSTM = config['n_layers_LSTM']
        self.doc_n_layers = config['n_layers_LSTM']

        self.tran_with_attention = config['tran_with_attention']
        self.doc_with_attention = config['doc_with_attention']
        self.tran_transform = config['tran_transform']
        self.doc_transform = config['doc_transform']

        self.transform_every_modal = config['transform_every_modal']
        self.transform_attn_out = config['transform_attn_out']

        self.use_tanh = config['use_tanh']

        if self.transform_every_modal:
            self.linear_single_modal = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                     nn.Tanh(),
                                                     nn.Linear(self.n_hidden, self.n_hidden))
        if self.transform_attn_out:
            self.linear_attn_out = nn.Sequential(nn.Linear(self.n_hidden, self.n_hidden),
                                                 nn.Tanh(),
                                                 nn.Linear(self.n_hidden, self.n_hidden))

        self.tran_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.tran_attn_scalar = nn.Linear(self.n_hidden, 1)

        self.doc_attn = nn.Linear(self.n_hidden, self.n_hidden)
        self.doc_attn_scalar = nn.Linear(self.n_hidden, 1)

        self.embedding = nn.Embedding(self.n_tran_doc_words, self.emb_size, padding_idx=0)
        self.init_xavier_linear(self.embedding, init_bias=False)

        self.init_encoder()

    def init_xavier_linear(self, linear, init_bias=True, gain=1, init_normal_std=1e-4):
        torch.nn.init.xavier_uniform_(linear.weight, gain)
        if init_bias:
            if linear.bias is not None:
                linear.bias.data.normal_(std=init_normal_std)

    def init_encoder(self):
        self.tran_encoder = SeqEncoder_LSTM(self.n_tran_words, self.emb_size, self.n_hidden,
                                            self.n_layers_LSTM, self.embedding)

        self.doc_encoder = SeqEncoder_LSTM(self.n_doc_words, self.emb_size, self.n_hidden, self.doc_n_layers,
                                           self.embedding)

    def LSTM_encoding(self, tran, tran_len):
        """
        [TODO] Encode one batch of translation-rule tokens into a fixed-size representation.

        Input:
            tran: (batch, seq_len) - translation-rule token ids.
            tran_len: (batch,) - valid lengths for each translation-rule sequence.

        Output: (batch, hidden_size) - translation-rule representation.

"""
        pass

    def tran_sequence_encoding(self, tran, tran_len):
        output = self.LSTM_encoding(tran, tran_len)
        return output

    def tran_block_encoding(self, tran, tran_block_len):
        """
        [TODO] Encode multi-block translation rules and pool blocks into one code representation.

        Input:
            tran: (batch, num_blocks, block_seq_len) - translation-rule token ids per block.
            tran_block_len: (batch, num_blocks) - valid token counts for each block.

        Output: (batch, hidden_size) - pooled translation/code representation.

"""
        pass

    def code_encoding(self, tran, tran_len, tran_block_len):
        return self.tran_block_encoding(tran, tran_block_len)

    def doc_encoding(self, doc, doc_len):
        """
        [TODO] Encode documentation tokens into a fixed-size query representation.

        Input:
            doc: (batch, doc_seq_len) - documentation token ids.
            doc_len: (batch,) - valid lengths for each documentation sequence.

        Output: (batch, hidden_size) - documentation representation.

"""
        pass

    def forward(self, tran, tran_len, tran_block_len, doc_anchor, doc_anchor_len, doc_neg, doc_neg_len):
        """
        [TODO] Compute the TranCS triplet ranking loss for code search.

        Input:
            tran: (batch, num_blocks, block_seq_len) - translation-rule code tokens.
            tran_len: (batch, num_blocks) - legacy/code lengths passed through the code path.
            tran_block_len: (batch, num_blocks) - valid lengths for each translation block.
            doc_anchor: (batch, doc_seq_len) - matching documentation tokens.
            doc_anchor_len: (batch,) - matching documentation lengths.
            doc_neg: (batch, doc_seq_len) - non-matching documentation tokens.
            doc_neg_len: (batch,) - non-matching documentation lengths.

        Output: scalar tensor - mean margin ranking loss for the batch.

"""
        pass


# ============================================================
# __main__: Automated test suite for 5 ablated functions
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

    def tiny_config():
        return {
            'gpu_id': 0,
            'code_nn': 'LSTM',
            'doc_nn': 'LSTM',
            'mode': 'Block',
            'tran_with_attention': 1,
            'doc_with_attention': 1,
            'tran_transform': 0,
            'doc_transform': 1,
            'transform_every_modal': 0,
            'transform_attn_out': 0,
            'use_tanh': 0,
            'emb_size': 8,
            'margin': 0.6,
            'sim_measure': 'cos',
            'dropout': 0.0,
            'batch_size': 2,
            'n_layers_LSTM': 1,
            'n_hidden': 8,
            'doc_len': 5,
            'tran_len': 6,
            'tran_seq_len': 4,
            'tran_block_len': 3,
            'n_tran_words': 50,
            'n_doc_words': 50,
            'n_tran_doc_words': 60,
        }

    print("=" * 70)
    print("TranCS: automated benchmark for context-aware translation/code search")
    print("Automated Test Suite - 5 ablated functions, 31 checks")
    print("=" * 70)
    print()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/5: SeqEncoder_LSTM.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/5] SeqEncoder_LSTM.forward - sorted packed LSTM restores batch order")
    try:
        conf = tiny_config()
        embedding = nn.Embedding(conf['n_tran_doc_words'], conf['emb_size'], padding_idx=0)
        encoder = SeqEncoder_LSTM(conf['n_tran_words'], conf['emb_size'], conf['n_hidden'], conf['n_layers_LSTM'], embedding).to(device)
        inputs = torch.tensor([[4, 5, 6, 0], [7, 8, 0, 0], [9, 10, 11, 12]], dtype=torch.long, device=device)
        lens = torch.tensor([3, 2, 4], dtype=torch.long, device=device)
        hidden = encoder.init_hidden(inputs.size(0))
        hids, state = encoder(inputs, lens, hidden)
        check("SeqEncoder output not None", hids is not None and state is not None)
        if hids is not None and state is not None:
            h_n, c_n = state
            check("SeqEncoder hids shape", tuple(hids.shape) == (3, 4, 8), f"got {tuple(hids.shape)}")
            check("SeqEncoder hidden shape", tuple(h_n.shape) == (3, 8), f"got {tuple(h_n.shape)}")
            check("SeqEncoder cell shape", tuple(c_n.shape) == (3, 8), f"got {tuple(c_n.shape)}")
            check("SeqEncoder finite", torch.isfinite(hids).all().item() and torch.isfinite(h_n).all().item())
            loss = h_n.sum() + hids.sum()
            loss.backward()
            check("SeqEncoder embedding grad", embedding.weight.grad is not None and torch.isfinite(embedding.weight.grad).all().item())
        else:
            skip_checks(5, "SeqEncoder_LSTM.forward returned None")
    except Exception as exc:
        skip_checks(6, f"SeqEncoder_LSTM.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/5: TranEmbeder.LSTM_encoding
    # ==========================================================
    print("-" * 60)
    print("[Test 2/5] TranEmbeder.LSTM_encoding - translation attention respects lengths")
    try:
        conf = tiny_config()
        model = TranEmbeder(conf).to(device)
        model.eval()
        tran = torch.tensor([[3, 4, 5, 0], [6, 7, 0, 0]], dtype=torch.long, device=device)
        tran_len = torch.tensor([3, 2], dtype=torch.long, device=device)
        output = model.LSTM_encoding(tran, tran_len)
        check("LSTM_encoding output not None", output is not None)
        if output is not None:
            check("LSTM_encoding output shape", tuple(output.shape) == (2, 8), f"got {tuple(output.shape)}")
            check("LSTM_encoding output finite", torch.isfinite(output).all().item())
            check("LSTM_encoding attention scalar trainable", model.tran_attn_scalar.weight.requires_grad)
            with torch.no_grad():
                variant = tran.clone()
                variant[0, 3] = 29
                variant[1, 2:] = torch.tensor([31, 32], dtype=torch.long)
                output_variant = model.LSTM_encoding(variant, tran_len)
            check("LSTM_encoding ignores padded tokens", torch.allclose(output.detach(), output_variant, atol=1e-5))
        else:
            skip_checks(4, "LSTM_encoding returned None")
    except Exception as exc:
        skip_checks(5, f"TranEmbeder.LSTM_encoding raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/5: TranEmbeder.tran_block_encoding
    # ==========================================================
    print("-" * 60)
    print("[Test 3/5] TranEmbeder.tran_block_encoding - block-wise translation max pooling")
    try:
        conf = tiny_config()
        model = TranEmbeder(conf).to(device)
        model.eval()
        tran = torch.tensor([
            [[3, 4, 0, 0], [5, 6, 7, 0], [8, 9, 10, 11]],
            [[12, 13, 14, 0], [15, 16, 0, 0], [17, 18, 19, 0]],
        ], dtype=torch.long, device=device)
        block_len = torch.tensor([[2, 3, 4], [3, 2, 3]], dtype=torch.long, device=device)
        output = model.tran_block_encoding(tran, block_len)
        check("tran_block output not None", output is not None)
        if output is not None:
            check("tran_block output shape", tuple(output.shape) == (2, 8), f"got {tuple(output.shape)}")
            check("tran_block output finite", torch.isfinite(output).all().item())
            manual = torch.stack([model.LSTM_encoding(tran[i], block_len[i]) for i in range(2)])
            manual = F.max_pool2d(manual, kernel_size=(conf['tran_block_len'], 1), stride=1).squeeze(1)
            check("tran_block max-pool semantic", torch.allclose(output, manual, atol=1e-6))
            check("tran_block embedding trainable", model.embedding.weight.requires_grad)
        else:
            skip_checks(4, "tran_block_encoding returned None")
    except Exception as exc:
        skip_checks(5, f"TranEmbeder.tran_block_encoding raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/5: TranEmbeder.doc_encoding
    # ==========================================================
    print("-" * 60)
    print("[Test 4/5] TranEmbeder.doc_encoding - documentation attention masks padding")
    try:
        conf = tiny_config()
        model = TranEmbeder(conf).to(device)
        model.eval()
        doc = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 0]], dtype=torch.long, device=device)
        doc_len = torch.tensor([3, 4], dtype=torch.long, device=device)
        output = model.doc_encoding(doc, doc_len)
        check("doc_encoding output not None", output is not None)
        if output is not None:
            check("doc_encoding output shape", tuple(output.shape) == (2, 8), f"got {tuple(output.shape)}")
            check("doc_encoding output finite", torch.isfinite(output).all().item())
            check("doc_encoding attention scalar trainable", model.doc_attn_scalar.weight.requires_grad)
            with torch.no_grad():
                variant = doc.clone()
                variant[0, 3:] = torch.tensor([25, 26], dtype=torch.long)
                variant[1, 4] = 27
                output_variant = model.doc_encoding(variant, doc_len)
            check("doc_encoding ignores padded tokens", torch.allclose(output.detach(), output_variant, atol=1e-5))
        else:
            skip_checks(4, "doc_encoding returned None")
    except Exception as exc:
        skip_checks(5, f"TranEmbeder.doc_encoding raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/5: TranEmbeder.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 5/5] TranEmbeder.forward - triplet margin ranking loss")
    try:
        conf = tiny_config()
        model = TranEmbeder(conf).to(device)
        model.train()
        tran = torch.tensor([
            [[3, 4, 0, 0], [5, 6, 7, 0], [8, 9, 10, 11]],
            [[12, 13, 14, 0], [15, 16, 0, 0], [17, 18, 19, 0]],
        ], dtype=torch.long, device=device)
        block_len = torch.tensor([[2, 3, 4], [3, 2, 3]], dtype=torch.long, device=device)
        tran_len = block_len
        doc_anchor = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 8, 9, 0]], dtype=torch.long, device=device)
        doc_anchor_len = torch.tensor([3, 4], dtype=torch.long, device=device)
        doc_neg = torch.tensor([[20, 21, 22, 0, 0], [23, 24, 25, 26, 0]], dtype=torch.long, device=device)
        doc_neg_len = torch.tensor([3, 4], dtype=torch.long, device=device)
        loss = model(tran, tran_len, block_len, doc_anchor, doc_anchor_len, doc_neg, doc_neg_len)
        check("forward loss not None", loss is not None)
        if loss is not None:
            check("forward loss scalar", loss.ndim == 0, f"got shape {tuple(loss.shape)}")
            check("forward loss finite", torch.isfinite(loss).item())
            with torch.no_grad():
                code_repr = model.code_encoding(tran, tran_len, block_len)
                anchor_repr = model.doc_encoding(doc_anchor, doc_anchor_len)
                neg_repr = model.doc_encoding(doc_neg, doc_neg_len)
                expected = (conf['margin'] - F.cosine_similarity(code_repr, anchor_repr) + F.cosine_similarity(code_repr, neg_repr)).clamp(min=1e-6).mean()
            check("forward triplet-margin semantic", torch.allclose(loss.detach(), expected, atol=1e-5), f"expected {expected.item()}, got {loss.item()}")
            check("forward loss nonnegative", loss.item() >= 0.0)
        else:
            skip_checks(4, "forward returned None")
    except Exception as exc:
        skip_checks(5, f"TranEmbeder.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The TranCS core model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
