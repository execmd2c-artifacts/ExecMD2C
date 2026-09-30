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
    batch_size, D = X.shape
    if k is not None:
        X = rand_project(X, k, D)
    X = torch.cat([X, torch.ones([batch_size, 1]).to(X.device)], dim=1)
    rows, columns = X.shape
    if rows < columns:
        logging.warning(f'Feature Matrix is underdetermined: {rows} rows and {columns} columns.'
                        f'Returning 0 for decorrelation loss and R^2.')
        return None
    else:
        Yhat = torch.matmul(torch.matmul(X, torch.linalg.pinv(X)), Y)
        return Yhat


def decorrelation_fn(X, Y_all, k=None, return_R=True):
    """Calculates the decorrelation across multiple sets of features from other models
    X (Tensor): Extracted features from the current model that is training. Shape (batch_size, feature_dim)
    Y_all (Tensor): Series of extracted features from previously trained models. Shape (model_num, batch_size, feature_dim)
    k (int): Dimension in which to randomly project features
    return_R (bool): Flag that controls whether to return average R^2 value. If false, only the component needed for the
     loss is returned."""
    eta = 0.00001  # constant for stability
    loss_R = 0
    R = 0
    for Y in Y_all:
        if np.random.uniform() > 0.5:
            X, Y = Y, X
        X = torch.flatten(X, start_dim=1)
        Y = torch.flatten(Y, start_dim=1)
        Yhat = get_Y_hat(X, Y, k)
        if Yhat is None:
            R += torch.tensor(0, dtype=torch.float32, device=X.device)
            loss_R += torch.tensor(0, dtype=torch.float32, device=X.device)
        else:
            SSres = torch.sum(torch.square(Y - Yhat))
            SStot = torch.sum(torch.square(Y - torch.mean(Y, dim=0).unsqueeze(0)))
            R += (1 - SSres/(SStot+eta))
            loss_R += (torch.log(SStot+eta) - torch.log(SSres+eta))
  #      print('SSres:{} SStot:{} R:{}'.format(SSres, SStot, R))
    loss_R /= len(Y_all)
    R /= len(Y_all)
    if return_R:
        return loss_R, R
    else:
        return loss_R


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
        if self.filter is not None:
            x = torch.fft.fft(x)
            x = torch.multiply(x, self.filter)
            x = torch.fft.ifft(x).real
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.dropout4(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.dropout5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.maxpool6(x)
        x = self.dropout6(x)
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.dropout7(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.dropout8(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.dropout9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = F.relu(x)
        x = self.dropout10(x)
        x = self.conv11(x)
        x = self.bn11(x)
        x = F.relu(x)
        x = self.maxpool11(x)
        x = self.dropout11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = F.relu(x)
        x = self.dropout12(x)
        x = self.conv13(x)
        x = self.bn13(x)
        x = F.relu(x)
        x = self.dropout13(x)
        self.features = torch.mean(x, dim=2)
        x = self.fc(self.features)
        if return_features:
            return x, self.features
        else:
            return x

    def get_features(self, x, layer_num):
        """This method extracts the features distilled from batch x at layer number layer_num,
        which is used for DVERGE training"""
        if self.filter is not None:
            x = torch.fft.fft(x)
            x = torch.multiply(x, self.filter)
            x = torch.fft.ifft(x).real
        for i in range(1, layer_num+1):
            if i != 1:
                x = F.relu(x)
                if i in [2, 7, 12]:
                    x = getattr(self, f'maxpool{i-1}')(x)
                x = getattr(self, f'dropout{i-1}')(x)
            x = getattr(self, f'conv{i}')(x)
            x = getattr(self, f'bn{i}')(x)
        return x


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
