import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ground_truth.py - DNA_ECG Core Model Components (Self-contained)
# Source: Adversarial/DNA_ECG-main
#
# Contains ONLY the CNN architecture, Fourier partition filters, and
# decorrelation objective used by Decorrelative Network Architecture.
# No training loops, datasets, evaluation scripts, checkpoints, or attack
# orchestration code are included.
# ============================================================


# --- [Original file: utils/filters.py] ---
# Complementary filter1 and filter2 for Physionet Data
N = 18000
bins = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
bin_length = int(N / 2 / len(bins))
filter1 = bins[0] * np.ones(bin_length, dtype=np.float32)
for i in range(1, len(bins)):
    filter1 = np.concatenate((filter1, bins[i] * np.ones(bin_length)))
filter1 = np.append(filter1, filter1[-1])
filter1 = np.concatenate([filter1, np.flip(filter1[1:-1])])

filter2 = 1 - filter1


# Complementary filter1 and filter2 for CPSC 2018 Data
N = 24000
bins = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
bin_length = int(N / 2 / len(bins))
china_filter1 = bins[0] * np.ones(bin_length, dtype=np.float32)
for i in range(1, len(bins)):
    china_filter1 = np.concatenate((china_filter1, bins[i] * np.ones(bin_length)))
china_filter1 = np.append(china_filter1, china_filter1[-1])
china_filter1 = np.concatenate([china_filter1, np.flip(china_filter1[1:-1])])

china_filter2 = 1 - china_filter1


# --- [Original file: utils/decorrelation_func.py] ---
def rand_project(X, k, D):
    R = torch.randn(D, k).to(X.device)/np.sqrt(D)
    return torch.matmul(X, R)


def get_Y_hat(X, Y, k):
    """
    [TODO] Estimate the part of one feature matrix that is linearly explainable by another.

    Input:
        X: (batch, feature_dim_x) - current model features.
        Y: (batch, feature_dim_y) - features from another model.
        k: int or None - optional projected dimension for X before fitting.

    Output: (batch, feature_dim_y) tensor or None - predicted Y features from X.

"""
    pass


def decorrelation_fn(X, Y_all, k=None, return_R=True):
    """
    [TODO] Compute the decorrelation loss across current and previous model features.

    Input:
        X: (batch, feature_dim...) - features from the model currently being trained.
        Y_all: (num_models, batch, feature_dim...) - feature tensors from previously trained ensemble members.
        k: int or None - optional projection dimension for the predictability estimate.
        return_R: bool - whether to return both loss and average R statistic.

    Output:
        return_R=True: tuple of two scalar tensors, decorrelation loss and average R statistic.
        return_R=False: scalar tensor, decorrelation loss only.

"""
    pass


# --- [Original file: model/CNN.py] ---
class CNN(nn.Module):
    """This is the CNN architecture used in Han, et al. Deep
       learning models for electrocardiograms are susceptible to adversarial attack.
       Definition has been modified to accommodate prefiltering and decorrelation"""
    def __init__(self, num_classes=4, input_channels=1, f_filter=None):
        super(CNN, self).__init__()
        if f_filter is None:
            self.filter = None
        else:
            self.filter = nn.parameter.Parameter(torch.tensor(f_filter, dtype=torch.float32), requires_grad=False)

        self.conv1 = nn.Conv1d(input_channels, 320, kernel_size=24, stride=1, padding=11, bias=True)
        self.bn1 = nn.BatchNorm1d(320)
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=2)
        self.dropout1 = nn.Dropout(0.3)
        self.conv2 = nn.Conv1d(320, 256, kernel_size=16, stride=1, padding=15, dilation=2, bias=True)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.conv3 = nn.Conv1d(256, 256, kernel_size=16, stride=1, padding=30, dilation=4, bias=True)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout3 = nn.Dropout(0.3)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=16, stride=1, padding=30, dilation=4, bias=True)
        self.bn4 = nn.BatchNorm1d(256)
        self.dropout4 = nn.Dropout(0.3)
        self.conv5 = nn.Conv1d(256, 256, kernel_size=16, stride=1, padding=30, dilation=4, bias=True)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.3)
        self.conv6 = nn.Conv1d(256, 128, kernel_size=8, stride=1, padding=14, dilation=4, bias=True)
        self.bn6 = nn.BatchNorm1d(128)
        self.maxpool6 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation = 2)
        self.dropout6 = nn.Dropout(0.3)
        self.conv7 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn7 = nn.BatchNorm1d(128)
        self.dropout7 = nn.Dropout(0.3)
        self.conv8 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn8 = nn.BatchNorm1d(128)
        self.dropout8 = nn.Dropout(0.3)
        self.conv9 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn9 = nn.BatchNorm1d(128)
        self.dropout9 = nn.Dropout(0.3)
        self.conv10 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=21, dilation=6, bias=True)
        self.bn10 = nn.BatchNorm1d(128)
        self.dropout10 = nn.Dropout(0.3)
        self.conv11 = nn.Conv1d(128, 128, kernel_size=8, stride=1, padding=28, dilation=8, bias=True)
        self.bn11 = nn.BatchNorm1d(128)
        self.maxpool11 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=2)
        self.dropout11 = nn.Dropout(0.3)
        self.conv12 = nn.Conv1d(128, 64, kernel_size=8, stride=1, padding=28, dilation=8, bias=True)
        self.bn12 = nn.BatchNorm1d(64)
        self.dropout12 = nn.Dropout(0.3)
        self.conv13 = nn.Conv1d(64, 64, kernel_size=8, stride=1, padding=28, dilation=8, bias=True)
        self.bn13 = nn.BatchNorm1d(64)
        self.dropout13 = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)
        self.features = None
        self.layer_list = list(range(1, 14))  # List of layer indices for DVERGE

    def forward(self, x, return_features=False):
        """
        [TODO] Run the filtered ECG CNN classifier and optionally return decorrelation features.

        Input:
            x: (batch, input_channels, signal_length) - ECG waveform batch.
            return_features: bool - whether to return the pooled penultimate features.

        Output:
            return_features=False: (batch, num_classes) logits.
            return_features=True: tuple of (batch, num_classes) logits and (batch, 64) features.

"""
        pass

    def get_features(self, x, layer_num):
        """
        [TODO] Extract the intermediate convolutional representation used for DVERGE feature distillation.

        Input:
            x: (batch, input_channels, signal_length) - ECG waveform batch.
            layer_num: int - final convolutional layer index to include, from 1 through 13.

        Output: (batch, channels_at_layer, time_at_layer) tensor - normalized feature map at the requested layer.

"""
        pass


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
    print("DNA_ECG: decorrelative ECG model benchmark")
    print("Automated Test Suite - 4 ablated targets")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: CNN.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] CNN.forward - filtered ECG classification and feature return")
    try:
        length = 512
        local_filter = np.resize(filter1, length).astype(np.float32)
        model = CNN(num_classes=4, input_channels=1, f_filter=local_filter).to(device)
        model.eval()
        x = torch.randn(2, 1, length, device=device)
        with torch.no_grad():
            result = model(x, return_features=True)
        check("CNN.forward output not None", result is not None)
        if result is not None:
            logits, features = result
            check("CNN.forward logits shape", tuple(logits.shape) == (2, 4),
                  f"expected (2, 4), got {tuple(logits.shape)}")
            check("CNN.forward features shape", tuple(features.shape) == (2, 64),
                  f"expected (2, 64), got {tuple(features.shape)}")
            check("CNN.forward outputs finite", torch.isfinite(logits).all().item() and torch.isfinite(features).all().item())
            check("CNN.forward stores features", model.features is not None and tuple(model.features.shape) == (2, 64))
            check("CNN.forward filter frozen", model.filter is not None and model.filter.requires_grad is False)
        else:
            skip_checks(5, "CNN.forward returned None")
    except Exception as e:
        skip_checks(6, f"CNN.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: CNN.get_features
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] CNN.get_features - intermediate DVERGE feature extraction")
    try:
        model = CNN(num_classes=4, input_channels=1).to(device)
        model.eval()
        x = torch.randn(2, 1, 512, device=device)
        with torch.no_grad():
            layer1 = model.get_features(x, 1)
            layer2 = model.get_features(x, 2)
            layer13 = model.get_features(x, 13)
        check("CNN.get_features output not None", layer1 is not None and layer2 is not None and layer13 is not None)
        if layer1 is not None and layer2 is not None and layer13 is not None:
            check("CNN.get_features layer1 shape", tuple(layer1.shape) == (2, 320, 511),
                  f"expected (2, 320, 511), got {tuple(layer1.shape)}")
            check("CNN.get_features layer2 shape", tuple(layer2.shape) == (2, 256, 256),
                  f"expected (2, 256, 256), got {tuple(layer2.shape)}")
            check("CNN.get_features layer13 shape", tuple(layer13.shape) == (2, 64, 64),
                  f"expected (2, 64, 64), got {tuple(layer13.shape)}")
            finite = torch.isfinite(layer1).all().item() and torch.isfinite(layer2).all().item() and torch.isfinite(layer13).all().item()
            check("CNN.get_features outputs finite", finite)
        else:
            skip_checks(4, "CNN.get_features returned None")
    except Exception as e:
        skip_checks(5, f"CNN.get_features raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: get_Y_hat
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] get_Y_hat - linear feature predictability estimate")
    try:
        X = torch.randn(8, 3, device=device)
        true_w = torch.tensor([[1.0, -0.5], [0.25, 0.75], [-1.0, 0.5]], device=device)
        Y = X.matmul(true_w) + 0.1
        Yhat = get_Y_hat(X, Y, k=None)
        under = get_Y_hat(torch.randn(2, 4, device=device), torch.randn(2, 3, device=device), k=None)
        check("get_Y_hat output not None", Yhat is not None)
        if Yhat is not None:
            check("get_Y_hat output shape", tuple(Yhat.shape) == (8, 2),
                  f"expected (8, 2), got {tuple(Yhat.shape)}")
            check("get_Y_hat output finite", torch.isfinite(Yhat).all().item())
            check("get_Y_hat recovers linear target", torch.mean(torch.abs(Yhat - Y)).item() < 1e-4)
            check("get_Y_hat underdetermined returns None", under is None)
        else:
            skip_checks(4, "get_Y_hat returned None")
    except Exception as e:
        skip_checks(5, f"get_Y_hat raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: decorrelation_fn
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] decorrelation_fn - decorrelation loss and R statistic")
    try:
        np.random.seed(1)
        X = torch.randn(8, 4, device=device, requires_grad=True)
        Y_related = X.detach().unsqueeze(0).clone()
        loss_related, R_related = decorrelation_fn(X, Y_related, k=None, return_R=True)
        np.random.seed(1)
        Y_random = torch.randn(1, 8, 4, device=device)
        loss_random, R_random = decorrelation_fn(X, Y_random, k=None, return_R=True)
        check("decorrelation_fn output not None", loss_related is not None and R_related is not None)
        if loss_related is not None and R_related is not None:
            scalar_shapes = tuple(loss_related.shape) == () and tuple(R_related.shape) == ()
            check("decorrelation_fn scalar outputs", scalar_shapes)
            finite = torch.isfinite(loss_related).item() and torch.isfinite(R_related).item()
            finite = finite and torch.isfinite(loss_random).item() and torch.isfinite(R_random).item()
            check("decorrelation_fn outputs finite", finite)
            check("decorrelation_fn higher R for related features", R_related.item() > R_random.item())
            loss_related.backward()
            check("decorrelation_fn gradient exists", X.grad is not None and torch.isfinite(X.grad).all().item())
            loss_only = decorrelation_fn(X.detach(), Y_random, k=None, return_R=False)
            check("decorrelation_fn return_R false scalar", loss_only is not None and tuple(loss_only.shape) == ())
        else:
            skip_checks(5, "decorrelation_fn returned None")
    except Exception as e:
        skip_checks(6, f"decorrelation_fn raised {type(e).__name__}: {e}")
    print()

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
