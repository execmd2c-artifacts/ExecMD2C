# ============================================================
# ground_truth.py - DRW Core Model Components
# Source: NLP/DRW-main
#
# Contains ONLY model wrappers and direct watermark/loss/detection components.
# No dataset, training loop, evaluation loop, saved-weight I/O, or remote model loading.
# ============================================================

# --- Third-party imports ---
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from astropy.timeseries import LombScargle


# --- [Original file: model.py] ---
class BertClassifier(nn.Module):

    def __init__(self, model, num_class=6, hidden=768, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = model
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, num_class)

    def forward(self, input_id, mask):
        embedded = self.dropout(self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)[1])
        predictions = self.linear(embedded)
        return predictions


class BertTokenClassifier(nn.Module):
    def __init__(self, bert, num_class, dropout=0.5):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.linear = nn.Linear(embedding_dim, num_class)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        # embedded = [batch size, sent len, emb dim]
        predictions = self.linear(self.dropout(embedded))
        # predictions = [batch size, sent len, output dim]
        return predictions


class BERTPoSTagger(nn.Module):
    def __init__(self, bert, output_dim, dropout):
        super().__init__()
        self.bert = bert
        embedding_dim = bert.config.to_dict()['hidden_size']
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        # text = [sent len, batch size]
        text = text.permute(1, 0)
        # text = [batch size, sent len]
        embedded = self.dropout(self.bert(text)[0])
        # embedded = [batch size, seq len, emb dim]
        embedded = embedded.permute(1, 0, 2)
        # embedded = [sent len, batch size, emb dim]
        predictions = self.fc(self.dropout(embedded))
        # predictions = [sent len, batch size, output dim]
        return predictions


# --- [Original file: wm.py] ---
def softmax_signal_wm(output, embs, task, vec, key, skey, k, epsilon, num_classes=18, shape='cosine',
                      padding=0.0, device='cpu', tid=0, wmidx=[], sub=1.0):
    """
    TODO: Reproduce DRW's soft-label watermark signal injection.

    Input:
        output: logits shaped (batch, num_classes) for sentence tasks, or
            (batch, seq_len, num_classes) for token tasks.
        embs: token ids shaped (batch, seq_len) used to derive the watermark phase input.
        task: task name; token tasks use every token, sentence/GLUE tasks use token position tid.
        vec: lookup matrix indexed by token ids, shaped (vocab_size, key_dim) or compatible.
        key: watermark key tensor producing the main phase signal.
        skey: selection key tensor producing the optional subsampling gate.
        k: frequency multiplier for the cosine watermark.
        epsilon: total watermark perturbation budget.
        num_classes: class count N.
        wmidx: optional list of classes that may receive the selective watermark branch.
        sub: selection threshold for the skey gate.

    Output:
        Watermarked soft-label distribution shaped (effective_batch, num_classes), where effective_batch
        is batch for sentence tasks and batch * seq_len for token tasks.

"""
    pass


# --- [Original file: utils.py] ---
def build_periodogram(xy_array, n_freqs=200000, k=0.5):
    """
    TODO: Reproduce the Lomb-Scargle periodogram construction for watermark detection.

    Input:
        xy_array: (num_points, 2) array where column 0 is the watermark phase input and column 1 is
            the observed softmax/probability trace for a class.
        n_freqs: number of frequency grid points.
        k: watermark angular frequency used to choose diagnostic model parameters.

    Output:
        freqs_array: (n_freqs, 2) array with frequency grid in column 0 and spectral power in column 1.
        thetas: model parameter vector for the frequency nearest k / (2*pi), or three zeros for empty input.

"""
    pass


def get_spectrum_window(freqs, powers, k, halfwidth=0.001, avg=True):
    """
    TODO: Reproduce the watermark-frequency window statistic.

    Input:
        freqs: (num_freqs,) frequency grid in cycles.
        powers: (num_freqs,) periodogram power values aligned with freqs.
        k: target angular watermark frequency.
        halfwidth: angular half-width around k for the detection window.
        avg: if True return average in-window power; otherwise return summed in-window power.

    Output:
        Tuple of two scalars: in-window statistic and ratio against out-of-window background power.

"""
    pass


class KLLoss(nn.Module):
    """
    Custom loss function performing KL loss on soft labels
    """

    def __init__(self, num_classes=10):
        super(KLLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, predicted, target, label=None):
        """
        TODO: Reproduce the soft-label cross-entropy loss used for distillation.

        Input:
            predicted: (num_points, num_classes) student logits.
            target: (num_points, num_classes) teacher soft-label distribution.
            label: optional (num_points,) hard labels where -100 marks ignored positions.

        Output:
            Scalar tensor equal to the mean target-weighted class cross-entropy over non-ignored points.

"""
        pass


# ============================================================
# __main__

# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

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
    print("DRW - benchmark suite")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1/4: softmax_signal_wm sequence classification branch
    # ------------------------------------------------------------
    try:
        output = torch.tensor([[2.0, 0.1, -0.3], [0.2, 1.5, -0.4]], dtype=torch.float32)
        embs = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.long)
        vec = torch.linspace(-1.0, 1.0, steps=10)
        key = torch.tensor([[0.5, -0.25, 0.75]], dtype=torch.float32)
        skey = torch.tensor([[0.2, 0.4, -0.6]], dtype=torch.float32)
        wm = softmax_signal_wm(output, embs, "imdb", vec, key, skey, k=3.0, epsilon=0.2, num_classes=3, tid=1)
        sm = F.softmax(output, dim=1)
        check("softmax_signal_wm cls output not None", wm is not None)
        if wm is not None:
            check("softmax_signal_wm cls shape", tuple(wm.shape) == (2, 3), str(tuple(wm.shape)))
            check("softmax_signal_wm cls finite", torch.isfinite(wm).all().item())
            check("softmax_signal_wm cls row mass positive", bool((wm.sum(dim=1) > 0).all().item()))
            check("softmax_signal_wm cls changes teacher softmax", not torch.allclose(wm, sm, atol=1e-5))
        else:
            skip_checks(4, "softmax_signal_wm returned None")
    except Exception as exc:
        skip_checks(5, f"softmax_signal_wm cls raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 2/4: softmax_signal_wm token branch with selected watermark classes
    # ------------------------------------------------------------
    try:
        token_output = torch.tensor([[[1.6, 0.2, -0.4], [0.1, 1.1, -0.2]], [[-0.2, 0.3, 1.4], [1.0, 0.5, -0.1]]], dtype=torch.float32)
        token_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
        vec = torch.linspace(-1.0, 1.0, steps=6)
        key = torch.tensor([[0.5, -0.25, 0.75]], dtype=torch.float32)
        skey = torch.tensor([[0.2]], dtype=torch.float32)
        wm = softmax_signal_wm(token_output, token_ids, "ner", vec, key, skey, k=2.0, epsilon=0.15, num_classes=3, wmidx=[1], sub=0.9)
        token_sm = F.softmax(token_output.reshape(-1, 3), dim=1)
        check("softmax_signal_wm token output not None", wm is not None)
        if wm is not None:
            check("softmax_signal_wm token flattened shape", tuple(wm.shape) == (4, 3), str(tuple(wm.shape)))
            check("softmax_signal_wm token finite", torch.isfinite(wm).all().item())
            check("softmax_signal_wm token nonnegative", bool((wm >= 0).all().item()))
            selected = torch.argmax(token_output.reshape(-1, 3), dim=1) == 1
            selection_ok = torch.allclose(wm[~selected], token_sm[~selected], atol=1e-6) and not torch.allclose(wm[selected], token_sm[selected], atol=1e-5)
            check("softmax_signal_wm token applies wmidx gate", selection_ok)
        else:
            skip_checks(4, "softmax_signal_wm token returned None")
    except Exception as exc:
        skip_checks(5, f"softmax_signal_wm token raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 3/4: KLLoss.forward
    # ------------------------------------------------------------
    try:
        criterion = KLLoss(num_classes=3)
        predicted = torch.tensor([[1.2, 0.2, -0.4], [0.1, 1.4, -0.2], [-0.5, 0.3, 1.0]], dtype=torch.float32, requires_grad=True)
        target = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=torch.float32)
        labels = torch.tensor([0, -100, 2], dtype=torch.long)
        loss = criterion(predicted, target, labels)
        check("KLLoss output not None", loss is not None)
        if loss is not None:
            check("KLLoss scalar shape", tuple(loss.shape) == (), str(tuple(loss.shape)))
            check("KLLoss finite", torch.isfinite(loss).item())
            check("KLLoss nonnegative", loss.item() >= 0.0)
            loss.backward()
            check("KLLoss respects ignore mask gradients", predicted.grad is not None and predicted.grad[1].abs().sum().item() == 0.0 and predicted.grad[[0, 2]].abs().sum().item() > 0)
        else:
            skip_checks(4, "KLLoss returned None")
    except Exception as exc:
        skip_checks(5, f"KLLoss raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 4/4: build_periodogram and get_spectrum_window
    # ------------------------------------------------------------
    try:
        x = np.linspace(0.0, 2.0 * np.pi, 64)
        k = 6.0
        xy = np.stack([x, 0.5 + 0.2 * np.cos(k * x)], axis=1)
        freqs_array, thetas = build_periodogram(xy, n_freqs=256, k=k)
        win_power, win_ratio = get_spectrum_window(freqs_array[:, 0], freqs_array[:, 1], k=k, halfwidth=0.5, avg=True)
        check("periodogram output not None", freqs_array is not None and thetas is not None)
        if freqs_array is not None and thetas is not None:
            check("periodogram shape", tuple(freqs_array.shape) == (256, 2), str(tuple(freqs_array.shape)))
            check("periodogram finite", np.isfinite(freqs_array).all())
            check("spectrum window finite", np.isfinite(win_power) and np.isfinite(win_ratio))
            check("spectrum window positive near watermark", win_power > 0.0 and win_ratio >= 0.0)
        else:
            skip_checks(4, "periodogram returned None")
    except Exception as exc:
        skip_checks(5, f"periodogram raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"RESULT: passed={passed} failed={failed}")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
