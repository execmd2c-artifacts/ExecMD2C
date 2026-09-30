# ============================================================
# ground_truth.py - SREA Core Model Components (Self-contained)
# Source: src/models/model.py, src/models/MultiTaskClassification.py,
#         src/utils/SREA_utils.py
#
# Contains the convolutional autoencoder, shared multi-task model, and
# SREA-specific centroid/re-labeling mechanisms. Training, datasets,
# evaluation, plotting, and CLI code are intentionally excluded.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- [Original file: src/models/model.py] ---
class MetaAE(nn.Module):
    def __init__(self, name='AE'):
        super(MetaAE, self).__init__()

        self.encoder = None
        self.decoder = None

        self.name = name

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_embedding(self, x):
        # Not Used for now
        emb = self.encoder(x)
        emb = emb / torch.sqrt(torch.sum(emb ** 2, 2, keepdim=True))
        return emb

    def get_name(self):
        return self.name


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, dropout=0.2, normalization='none'):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.conv, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0, output_padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        self.convtraspose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride,
                                               output_padding=output_padding,
                                               padding=padding)
        if normalization == 'batch':
            self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layers = [self.convtraspose, self.norm, self.act, self.dropout]
        # Remove None in layers
        self.net = nn.Sequential(*[x for x in self.layers if x])

    def forward(self, x):
        out = self.net(x)
        return out


class ConvEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, embedding_dim, kernel_size, stride=2, padding=0, dropout=0.2,
                 normalization='none'):
        super().__init__()
        num_blocks = len(num_channels)
        layers = []
        for i in range(num_blocks):
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dropout=dropout,
                          normalization=normalization)]

        self.network = nn.Sequential(*layers)
        self.conv1x1 = nn.Conv1d(num_channels[-1], embedding_dim, 1)

    def forward(self, x):
        x = self.network(x.transpose(2, 1))
        x = F.max_pool1d(x, kernel_size=x.data.shape[2])
        x = self.conv1x1(x)
        return x


def conv_out_len(seq_len, ker_size, stride, padding, dilation, stack):
    for _ in range(stack):
        seq_len = int((seq_len + 2 * padding - dilation * (ker_size - 1) - 1) / stride + 1)
    return seq_len


class ConvDecoder(nn.Module):
    def __init__(self, embedding_dim, num_channels, seq_len, out_dimension, kernel_size, stride=2, padding=0,
                 dropout=0.2, normalization='none'):
        super().__init__()

        num_channels = num_channels[::-1]
        num_blocks = len(num_channels)

        self.compressed_len = conv_out_len(seq_len, kernel_size, stride, padding, 1, num_blocks)

        # Pad sequence to match encoder lenght
        if stride > 1:
            output_padding = []
            seq = seq_len
            for _ in range(num_blocks):
                output_padding.append(seq % 2)
                seq = conv_out_len(seq, kernel_size, stride, padding, 1, 1)
            # bit flip
            if kernel_size % 2 == 1:
                output_padding = [1 - x for x in output_padding[::-1]]
            else:
                output_padding = output_padding[::-1]
        else:
            output_padding = [0] * num_blocks

        layers = []
        for i in range(num_blocks):
            in_channels = embedding_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                ConvTransposeBlock(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   output_padding=output_padding[i], dropout=dropout, normalization=normalization)]
        self.network = nn.Sequential(*layers)
        self.upsample = nn.Linear(1, self.compressed_len)
        self.conv1x1 = nn.Conv1d(num_channels[-1], out_dimension, 1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.network(x)
        x = self.conv1x1(x)
        return x.transpose(2, 1)


class CNNAE(MetaAE):
    def __init__(self, input_size, num_filters, embedding_dim, seq_len, kernel_size, dropout,
                 normalization=None, stride=2, padding=0, name='CNN_AE'):
        super(CNNAE, self).__init__(name=name)

        self.encoder = ConvEncoder(input_size, num_filters, embedding_dim, kernel_size=kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)
        self.decoder = ConvDecoder(embedding_dim, num_filters, seq_len, input_size, kernel_size, stride=stride,
                                   padding=padding, dropout=dropout, normalization=normalization)


# --- [Original file: src/models/MultiTaskClassification.py] ---
class MetaModel(nn.Module):
    def __init__(self, ae, classifier, name='network'):
        super(MetaModel, self).__init__()

        self.encoder = ae.encoder
        self.classifier = classifier
        self.name = name

    def forward(self, x):
        x_enc = self.encoder(x).squeeze()
        x_out = self.classifier(x_enc)
        return x_out.squeeze(-1)

    def get_name(self):
        return self.name


class AEandClass(MetaModel):
    def __init__(self, ae, **kwargs):
        super(AEandClass, self).__init__(ae, **kwargs)
        self.decoder = ae.decoder

    def forward(self, x):
        x_enc = self.encoder(x)
        xhat = self.decoder(x_enc)
        x_out = self.classifier(x_enc.squeeze(-1))
        return xhat, x_out, x_enc


class NonLinClassifier(nn.Module):
    def __init__(self, d_in, n_class, d_hidd=16, activation=nn.ReLU(), dropout=0.1, norm='batch'):
        """
        norm : str : 'batch' 'layer' or None
        """
        super(NonLinClassifier, self).__init__()

        self.dense1 = nn.Linear(d_in, d_hidd)

        if norm == 'batch':
            self.norm = nn.BatchNorm1d(d_hidd)
        elif norm == 'layer':
            self.norm = nn.LayerNorm(d_hidd)
        else:
            self.norm = None

        self.act = activation
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(d_hidd, n_class)

        self.layers = [self.dense1, self.norm, self.act, self.dropout, self.dense2]
        self.net = nn.Sequential(*[x for x in self.layers if x is not None])

    def forward(self, x):
        out = self.net(x)
        return out


# --- [Original file: src/utils/SREA_utils.py] ---
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


class CentroidLoss(nn.Module):
    """
    Centroid loss - Constraint Clustering loss of SREA
    """

    def __init__(self, feat_dim, num_classes, reduction='mean'):
        super(CentroidLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim), requires_grad=True)
        self.reduction = reduction
        self.rho = 1.0

    def forward(self, h, y):
        C = self.centers
        norm_squared = torch.sum((h.unsqueeze(1) - C) ** 2, 2)
        # Attractive
        distance = norm_squared.gather(1, y.unsqueeze(1)).squeeze()
        # Repulsive
        logsum = torch.logsumexp(-torch.sqrt(norm_squared), dim=1)
        loss = reduce_loss(distance + logsum, reduction=self.reduction)
        # Regularization
        reg = self.regularization(reduction='sum')
        return loss + self.rho * reg

    def regularization(self, reduction='sum'):
        C = self.centers
        pairwise_dist = torch.cdist(C, C, p=2) ** 2
        pairwise_dist = pairwise_dist.masked_fill(
            torch.zeros((C.size(0), C.size(0))).fill_diagonal_(1).bool().to(device), float('inf'))
        distance_reg = reduce_loss(-(torch.min(torch.log(pairwise_dist), dim=-1)[0]), reduction=reduction)
        return distance_reg


def create_hard_labels(embedding, centers, y_obs, yhat_hist, w_yhat, w_c, w_obs, classes):
    # TODO: add label temporal dynamics

    # yhat from previous metwork prediction. - Network Ensemble
    steps = yhat_hist.size(-1)
    decay = torch.arange(0, steps, 1).float().to(device)
    decay = torch.exp(-decay / 2)
    yhat_hist = yhat_hist * decay
    yhat = yhat_hist.mean(dim=-1) * w_yhat

    # Label from clustering
    distance_centers = torch.cdist(embedding, centers)
    yc = F.softmin(distance_centers, dim=1).detach() * w_c

    # Observed - given - label (noisy)
    yobs = F.one_hot(y_obs, num_classes=classes).float() * w_obs

    # Label combining
    ystar = (yhat + yc + yobs) / 3
    ystar = torch.argmax(ystar, dim=1)
    return ystar


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
    print("SREA: Self-Re-Labeling with Embedding Analysis")
    print("Automated Test Suite - 4 ablated targets")
    print("=" * 70)

    device = torch.device("cpu")

    print("-" * 70)
    print("[Test 1/4] AEandClass.forward - shared embedding multi-task branches")
    try:
        ae = CNNAE(input_size=3, num_filters=[4, 6], embedding_dim=5, seq_len=17,
                   kernel_size=4, dropout=0.0, normalization='none', stride=2, padding=2)
        model = AEandClass(ae=ae, classifier=NonLinClassifier(5, 4, d_hidd=7, dropout=0.0, norm=None)).to(device)
        model.eval()
        x = torch.randn(2, 17, 3, device=device)
        output = model(x)
        check("AEandClass output not None", output is not None)
        if output is not None:
            xhat, logits, embedding = output
            check("AEandClass reconstruction shape", tuple(xhat.shape) == (2, 17, 3),
                  f"expected (2, 17, 3), got {tuple(xhat.shape)}")
            check("AEandClass logits shape", tuple(logits.shape) == (2, 4),
                  f"expected (2, 4), got {tuple(logits.shape)}")
            check("AEandClass embedding shape", tuple(embedding.shape) == (2, 5, 1),
                  f"expected (2, 5, 1), got {tuple(embedding.shape)}")
            check("AEandClass outputs finite", all(torch.isfinite(value).all().item()
                                                     for value in (xhat, logits, embedding)))
            check("AEandClass branches use returned embedding",
                  torch.allclose(model.decoder(embedding), xhat) and
                  torch.allclose(model.classifier(embedding.squeeze(-1)), logits))
        else:
            skip_checks(5, "AEandClass.forward returned None")
    except Exception as exc:
        print(f"  [AEandClass.forward] ERROR - {exc}")
        skip_checks(6, "AEandClass.forward raised an exception")

    print("-" * 70)
    print("[Test 2/4] CentroidLoss.regularization - distinct-center separation")
    try:
        loss_module = CentroidLoss(feat_dim=2, num_classes=2, reduction='mean').to(device)
        with torch.no_grad():
            loss_module.centers.copy_(torch.tensor([[0.0, 0.0], [3.0, 4.0]], device=device))
        regularization = loss_module.regularization(reduction='sum')
        check("CentroidLoss regularization not None", regularization is not None)
        if regularization is not None:
            check("CentroidLoss regularization scalar", regularization.ndim == 0)
            check("CentroidLoss regularization finite", torch.isfinite(regularization).item())
            expected_regularization = -2.0 * torch.log(torch.tensor(25.0, device=device))
            check("CentroidLoss ignores self-distances",
                  torch.allclose(regularization, expected_regularization, atol=1e-6))
        else:
            skip_checks(3, "CentroidLoss.regularization returned None")
    except Exception as exc:
        print(f"  [CentroidLoss.regularization] ERROR - {exc}")
        skip_checks(4, "CentroidLoss.regularization raised an exception")

    print("-" * 70)
    print("[Test 3/4] CentroidLoss.forward - attraction, repulsion, and gradients")
    try:
        loss_module = CentroidLoss(feat_dim=2, num_classes=3, reduction='mean').to(device)
        h = torch.tensor([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], device=device, requires_grad=True)
        y = torch.tensor([0, 1, 2], device=device)
        output = loss_module(h, y)
        check("CentroidLoss output not None", output is not None)
        if output is not None:
            check("CentroidLoss output scalar", output.ndim == 0)
            check("CentroidLoss output finite", torch.isfinite(output).item())
            output.backward()
            check("CentroidLoss embedding gradient", h.grad is not None and h.grad.abs().sum().item() > 0)
            check("CentroidLoss center gradient",
                  loss_module.centers.grad is not None and loss_module.centers.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "CentroidLoss.forward returned None")
    except Exception as exc:
        print(f"  [CentroidLoss.forward] ERROR - {exc}")
        skip_checks(5, "CentroidLoss.forward raised an exception")

    print("-" * 70)
    print("[Test 4/4] create_hard_labels - prediction, center, and observed-label fusion")
    try:
        embedding = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], device=device)
        centers = torch.tensor([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]], device=device)
        observed = torch.tensor([2, 1, 0], device=device)
        history = torch.zeros(3, 3, 2, device=device)
        history[0, 1, 0] = 1.0
        history[1, 2, 0] = 1.0
        history[2, 0, 0] = 1.0
        output = create_hard_labels(embedding, centers, observed, history, 1.0, 0.0, 0.0, 3)
        check("create_hard_labels output not None", output is not None)
        if output is not None:
            check("create_hard_labels output shape", tuple(output.shape) == (3,),
                  f"expected (3,), got {tuple(output.shape)}")
            check("create_hard_labels output finite", torch.isfinite(output).all().item())
            check("create_hard_labels prediction branch", torch.equal(output, torch.tensor([1, 2, 0], device=device)))
            center_only = create_hard_labels(embedding, centers, observed, history, 0.0, 1.0, 0.0, 3)
            check("create_hard_labels center branch", torch.equal(center_only, torch.tensor([0, 1, 2], device=device)))
            observed_only = create_hard_labels(embedding, centers, observed, history, 0.0, 0.0, 1.0, 3)
            check("create_hard_labels observed branch", torch.equal(observed_only, observed))
            check("create_hard_labels class indices", bool(((output >= 0) & (output < 3)).all().item()))
        else:
            skip_checks(6, "create_hard_labels returned None")
    except Exception as exc:
        print(f"  [create_hard_labels] ERROR - {exc}")
        skip_checks(7, "create_hard_labels raised an exception")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
