# ============================================================
# GTM-Transformer core components. Source: models/GTM.py
# Excludes pretrained image/text backbones, Lightning training, and data code.
# ============================================================
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=52):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class FusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, use_img, use_text, dropout=0.2):
        super(FusionNetwork, self).__init__()
        self.img_pool = nn.AdaptiveAvgPool2d((1,1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        self.use_img = use_img
        self.use_text = use_text
        input_dim = embedding_dim + (embedding_dim*use_img) + (embedding_dim*use_text)
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim), nn.Linear(input_dim, input_dim, bias=False), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(input_dim, hidden_dim))

    def forward(self, img_encoding, text_encoding, dummy_encoding):
        """[TODO] Fuse selected image/text embeddings with calendar features.

        Input: image (batch, 2048, h, w), text (batch, embedding_dim), dummy (batch, embedding_dim).
        Output: (batch, hidden_dim).
"""
        pass


class GTrendEmbedder(nn.Module):
    def __init__(self, forecast_horizon, embedding_dim, use_mask, trend_len, num_trends, gpu_num):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.input_linear = TimeDistributed(nn.Linear(num_trends, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=trend_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.use_mask = use_mask
        self.gpu_num = gpu_num

    def _generate_encoder_mask(self, size, forecast_horizon):
        """[TODO] Build the segmented trend-attention mask.
        Input: sequence length and forecast horizon. Output: (size, size) additive attention mask.
"""
        pass

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, gtrends):
        """[TODO] Encode multivariate Google Trends with positional transformer attention.
        Input: (batch, num_trends, trend_len). Output: (trend_len, batch, embedding_dim).
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
    print("=" * 70)
    print("GTM-Transformer Automated Test Suite - 3 ablated targets")
    print("=" * 70)
    device = torch.device("cuda:0")
    print("-" * 70)
    print("[Test 1/3] GTrendEmbedder._generate_encoder_mask")
    try:
        module = GTrendEmbedder(4, 8, 1, 12, 3, 0).to(device)
        output = module._generate_encoder_mask(12, 4)
        check("trend mask not None", output is not None)
        if output is not None:
            check("trend mask shape", tuple(output.shape) == (12, 12))
            check("trend mask finite-or-negative-infinity", bool((torch.isfinite(output) | torch.isneginf(output)).all()))
            split = math.gcd(12, 4)
            check("trend mask keeps gcd-sized blocks", torch.equal(output[:split, :split], torch.zeros(split, split, device=device)) and torch.isneginf(output[0, split]))
        else: skip_checks(3, "trend mask returned None")
    except Exception as exc:
        print(f"  [trend mask] ERROR - {exc}")
        skip_checks(4, "trend mask raised an exception")
    print("-" * 70)
    print("[Test 2/3] GTrendEmbedder.forward")
    try:
        module = GTrendEmbedder(4, 8, 1, 12, 3, 0).to(device)
        module.eval()
        output = module(torch.randn(2, 3, 12, device=device))
        check("trend encoder output not None", output is not None)
        if output is not None:
            check("trend encoder output shape", tuple(output.shape) == (12, 2, 8))
            check("trend encoder output finite", torch.isfinite(output).all().item())
            check("trend encoder gradients", (output.sum().backward() is None) and module.input_linear.module.weight.grad is not None)
        else: skip_checks(3, "trend encoder returned None")
    except Exception as exc:
        print(f"  [trend encoder] ERROR - {exc}")
        skip_checks(4, "trend encoder raised an exception")
    print("-" * 70)
    print("[Test 3/3] FusionNetwork.forward")
    try:
        module = FusionNetwork(8, 6, 1, 1, dropout=0.0).to(device)
        module.eval()
        output = module(torch.randn(2, 2048, 2, 2, device=device), torch.randn(2, 8, device=device), torch.randn(2, 8, device=device))
        check("fusion output not None", output is not None)
        if output is not None:
            check("fusion output shape", tuple(output.shape) == (2, 6))
            check("fusion output finite", torch.isfinite(output).all().item())
            output.sum().backward()
            check("fusion modality projection gradient", module.img_linear.weight.grad is not None and module.img_linear.weight.grad.abs().sum().item() > 0)
        else: skip_checks(3, "fusion output returned None")
    except Exception as exc:
        print(f"  [fusion] ERROR - {exc}")
        skip_checks(4, "fusion raised an exception")
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
