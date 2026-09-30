"""
ground_truth.py for TS-TCC core model components.

Source-consolidated from:
- models/model.py
- models/attention.py
- models/TC.py
- models/loss.py

Only the temporal convolutional encoder, Transformer temporal-context module,
temporal contrasting module, and contrastive loss are included. Data loading,
augmentation, non-model orchestration, state-file handling, metrics,
logging, and CLI code are intentionally excluded.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn


# --- [Original file: models/model.py] ---
class base_Model(nn.Module):
    def __init__(self, configs):
        super(base_Model, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x_in):
        x = self.conv_block1(x_in)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x_flat = x.reshape(x.shape[0], -1)
        logits = self.logits(x_flat)
        return logits, x


# --- [Original file: models/attention.py] ---
########################################################################################

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class Seq_Transformer(nn.Module):
    def __init__(self, *, patch_size, dim, depth, heads, mlp_dim, channels=1, dropout=0.1):
        super().__init__()
        patch_dim = channels * patch_size
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.c_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout)
        self.to_c_token = nn.Identity()


    def forward(self, forward_seq):
        x = self.patch_to_embedding(forward_seq)
        b, n, _ = x.shape
        c_tokens = repeat(self.c_token, '() n d -> b n d', b=b)
        x = torch.cat((c_tokens, x), dim=1)
        x = self.transformer(x)
        c_t = self.to_c_token(x[:, 0])
        return c_t


# --- [Original file: models/TC.py] ---
class TC(nn.Module):
    def __init__(self, configs, device):
        super(TC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.timestep = configs.TC.timesteps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.timestep)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device
        
        self.projection_head = nn.Sequential(
            nn.Linear(configs.TC.hidden_dim, configs.final_out_channels // 2),
            nn.BatchNorm1d(configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(configs.final_out_channels // 2, configs.final_out_channels // 4),
        )

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features_aug1, features_aug2):
        z_aug1 = features_aug1  # features are (batch_size, #channels, seq_len)
        seq_len = z_aug1.shape[2]
        z_aug1 = z_aug1.transpose(1, 2)

        z_aug2 = features_aug2
        z_aug2 = z_aug2.transpose(1, 2)

        batch = z_aug1.shape[0]
        t_samples = torch.randint(seq_len - self.timestep, size=(1,)).long().to(self.device)  # randomly pick time stamps

        nce = 0  # average over timestep and batch
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = z_aug2[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = z_aug1[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.timestep
        return nce, self.projection_head(c_t)


# --- [Original file: models/loss.py] ---
class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


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

    class TinyTCConfig:
        hidden_dim = 16
        timesteps = 4

    class TinyConfig:
        input_channels = 2
        kernel_size = 8
        stride = 1
        final_out_channels = 32
        num_classes = 3
        dropout = 0.0
        features_len = 8
        TC = TinyTCConfig()

    print("=" * 70)
    print("TS-TCC core model component benchmark")
    print("=" * 70)

    device = torch.device("cpu")
    configs = TinyConfig()
    batch_size = 4
    seq_len = 48

    print("-" * 60)
    print("[Group 1] base_Model temporal convolutional encoder")
    try:
        model = base_Model(configs).to(device)
        x = torch.randn(batch_size, configs.input_channels, seq_len, device=device)
        logits, features = model(x)
        check("base_Model logits not None", logits is not None)
        check("base_Model features not None", features is not None)
        check("base_Model logits shape", tuple(logits.shape) == (batch_size, configs.num_classes),
              f"got {tuple(logits.shape)}")
        check("base_Model feature shape", tuple(features.shape) == (batch_size, configs.final_out_channels, configs.features_len),
              f"got {tuple(features.shape)}")
        check("base_Model outputs finite", torch.isfinite(logits).all().item() and torch.isfinite(features).all().item())
        logits.sum().backward()
        check("base_Model conv gradient flows", model.conv_block1[0].weight.grad is not None and model.conv_block1[0].weight.grad.abs().sum().item() > 0)
    except Exception as exc:
        skip_checks(6, f"base_Model raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 2] Attention")
    try:
        attn = Attention(dim=16, heads=4, dropout=0.0).to(device)
        tokens = torch.randn(batch_size, 6, 16, device=device)
        out = attn(tokens)
        check("Attention output not None", out is not None)
        check("Attention output shape", tuple(out.shape) == (batch_size, 6, 16), f"got {tuple(out.shape)}")
        check("Attention output finite", torch.isfinite(out).all().item())
        mask = torch.ones(batch_size, 5, dtype=torch.bool, device=device)
        masked_out = attn(tokens, mask=mask)
        check("Attention mask branch shape", tuple(masked_out.shape) == (batch_size, 6, 16), f"got {tuple(masked_out.shape)}")
        check("Attention mask branch finite", torch.isfinite(masked_out).all().item())
    except Exception as exc:
        skip_checks(5, f"Attention raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 3] Transformer and Seq_Transformer")
    try:
        transformer = Transformer(dim=16, depth=2, heads=4, mlp_dim=32, dropout=0.0).to(device)
        x_tokens = torch.randn(batch_size, 5, 16, device=device)
        y_tokens = transformer(x_tokens)
        check("Transformer output not None", y_tokens is not None)
        check("Transformer output shape", tuple(y_tokens.shape) == (batch_size, 5, 16), f"got {tuple(y_tokens.shape)}")
        check("Transformer output finite", torch.isfinite(y_tokens).all().item())

        seq_transformer = Seq_Transformer(patch_size=32, dim=16, depth=2, heads=4, mlp_dim=32, dropout=0.0).to(device)
        forward_seq = torch.randn(batch_size, 5, 32, device=device)
        c_t = seq_transformer(forward_seq)
        check("Seq_Transformer context not None", c_t is not None)
        check("Seq_Transformer context shape", tuple(c_t.shape) == (batch_size, 16), f"got {tuple(c_t.shape)}")
        check("Seq_Transformer context finite", torch.isfinite(c_t).all().item())
        check("Seq_Transformer has class token", tuple(seq_transformer.c_token.shape) == (1, 1, 16))
    except Exception as exc:
        skip_checks(7, f"Transformer/Seq_Transformer raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 4] Temporal contrasting module")
    try:
        tc = TC(configs, device).to(device)
        features_aug1 = torch.randn(batch_size, configs.final_out_channels, configs.features_len, device=device)
        features_aug2 = torch.randn(batch_size, configs.final_out_channels, configs.features_len, device=device)
        nce, projection = tc(features_aug1, features_aug2)
        check("TC nce not None", nce is not None)
        check("TC projection not None", projection is not None)
        check("TC nce scalar", nce.dim() == 0)
        check("TC projection shape", tuple(projection.shape) == (batch_size, configs.final_out_channels // 4),
              f"got {tuple(projection.shape)}")
        check("TC outputs finite", torch.isfinite(nce).item() and torch.isfinite(projection).all().item())
        check("TC predictor count", len(tc.Wk) == configs.TC.timesteps)
    except Exception as exc:
        skip_checks(6, f"TC raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 5] NTXentLoss correlated mask")
    try:
        criterion = NTXentLoss(device, batch_size, temperature=0.2, use_cosine_similarity=True)
        mask = criterion.mask_samples_from_same_repr
        check("NTXent mask not None", mask is not None)
        check("NTXent mask shape", tuple(mask.shape) == (2 * batch_size, 2 * batch_size), f"got {tuple(mask.shape)}")
        check("NTXent mask dtype bool", mask.dtype == torch.bool)
        mask_int = mask.to(torch.int64)
        check("NTXent mask diagonal false", mask_int.diag().sum().item() == 0)
        check("NTXent mask positive-pair offsets false",
              torch.diag(mask_int, batch_size).sum().item() == 0 and torch.diag(mask_int, -batch_size).sum().item() == 0)
        expected_negatives = 2 * batch_size - 2
        negatives_per_row = mask_int.sum(dim=1).tolist()
        check("NTXent mask negatives per row", all(v == expected_negatives for v in negatives_per_row),
              f"got {negatives_per_row}")
    except Exception as exc:
        skip_checks(6, f"NTXent mask raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 6] NTXentLoss forward")
    try:
        zis = torch.randn(batch_size, 8, device=device)
        zjs = torch.randn(batch_size, 8, device=device)
        criterion_cos = NTXentLoss(device, batch_size, temperature=0.2, use_cosine_similarity=True)
        loss_cos = criterion_cos(zis, zjs)
        criterion_dot = NTXentLoss(device, batch_size, temperature=0.2, use_cosine_similarity=False)
        loss_dot = criterion_dot(zis, zjs)
        sim = criterion_dot.similarity_function(torch.cat([zjs, zis], dim=0), torch.cat([zjs, zis], dim=0))
        check("NTXent cosine loss not None", loss_cos is not None)
        check("NTXent cosine loss scalar", loss_cos.dim() == 0)
        check("NTXent cosine loss finite", torch.isfinite(loss_cos).item())
        check("NTXent dot loss finite", torch.isfinite(loss_dot).item())
        check("NTXent dot similarity shape", tuple(sim.shape) == (2 * batch_size, 2 * batch_size), f"got {tuple(sim.shape)}")
        check("NTXent losses positive", loss_cos.item() > 0 and loss_dot.item() > 0)
    except Exception as exc:
        skip_checks(6, f"NTXent forward raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some target functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
