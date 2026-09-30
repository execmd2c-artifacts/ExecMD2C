# ============================================================
# ground_truth.py - SLA2P Core Model Components (Self-contained)
# Source:
#   Miscellaneous/SLA2P-TKDE-2024/models/fcn_pytorch.py
#   Miscellaneous/SLA2P-TKDE-2024/sla2p.py
#
# Contains ONLY model components and direct scoring/perturbation dependencies.
# No training loops, datasets, evaluation loops, checkpoint I/O, or CLI code.
# ============================================================

import numpy as np
import torch
import torch.nn as nn


# --- [Original file: models/fcn_pytorch.py] ---
__all__ = ['fcn']

class fcn_6(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(fcn_6, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, in_features_num * 8)
        self.bn3 = nn.BatchNorm1d(in_features_num * 8)
        self.act3 = nn.LeakyReLU(inplace=True)

        self.fc4 = nn.Linear(in_features_num * 8, in_features_num * 16)
        self.bn4 = nn.BatchNorm1d(in_features_num * 16)
        self.act4 = nn.LeakyReLU(inplace=True)

        self.fc5 = nn.Linear(in_features_num * 16, in_features_num * 32)
        self.bn5 = nn.BatchNorm1d(in_features_num * 32)
        self.act5 = nn.LeakyReLU(inplace=True)



        self.fc6 = nn.Linear(in_features_num * 32, class_num)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.act4(self.bn4(self.fc4(x)))
        x = self.act5(self.bn5(self.fc5(x)))

        x = self.fc6(x)

        return x

class fcn_4(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(fcn_4, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, in_features_num * 8)
        self.bn3 = nn.BatchNorm1d(in_features_num * 8)
        self.act3 = nn.LeakyReLU(inplace=True)



        self.fc4 = nn.Linear(in_features_num * 8, class_num)

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.act3(self.bn3(self.fc3(x)))
        x = self.fc4(x)

        return x
class fcn_drop(nn.Module):
    def __init__(self, in_features_num, class_num, droprate=0.1):
        super(fcn_drop, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, class_num)
        self.dropout=nn.Dropout(droprate)


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.fc3(x)

        return x
class fcn(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(fcn, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num * 4, class_num)


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

class one_linear(nn.Module):
    def __init__(self, in_features_num, class_num):
        super(one_linear, self).__init__()

        # self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        # self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        # self.act1 = nn.LeakyReLU(inplace=True)

        # self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        # self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        # self.act2 = nn.LeakyReLU(inplace=True)

        self.fc3 = nn.Linear(in_features_num , class_num)


    def forward(self, x):

        # x = self.act1(self.bn1(self.fc1(x)))
        # x = self.act2(self.bn2(self.fc2(x)))
        x = self.fc3(x)

        return x

class fcn_feature(nn.Module):
    def __init__(self, in_features_num):
        super(fcn_feature, self).__init__()

        self.fc1 = nn.Linear(in_features_num, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.fc2 = nn.Linear(in_features_num * 2, in_features_num * 4)
        self.bn2 = nn.BatchNorm1d(in_features_num * 4)
        self.act2 = nn.LeakyReLU(inplace=True)

        # self.fc3 = nn.Linear(in_features_num * 4, in_features_num * 2)
        # self.bn3 = nn.BatchNorm1d(in_features_num * 2)
        # self.act3 = nn.LeakyReLU(inplace=True)
        # 
        # self.fc4 = nn.Linear(in_features_num * 2, in_features_num )
        # self.bn4 = nn.BatchNorm1d(in_features_num )
        # self.act4 = nn.LeakyReLU(inplace=True)

        # self.fc3 = nn.Linear(in_features_num * 4, class_num)


    def forward(self, x):

        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        # x = self.act3(self.bn3(self.fc3(x)))
        # x = self.act4(self.bn4(self.fc4(x)))
        # x = self.fc3(x)

        return x
class Decoder(nn.Module):
    def __init__(self, in_features_num, original_features_num):
        super(Decoder, self).__init__()
        self.fc2 = nn.Linear(in_features_num * 2, in_features_num)
        self.bn2 = nn.BatchNorm1d(in_features_num)
        self.act2 = nn.LeakyReLU(inplace=True)

        self.fc1 = nn.Linear(in_features_num * 4, in_features_num * 2)
        self.bn1 = nn.BatchNorm1d(in_features_num * 2)
        self.act1 = nn.LeakyReLU(inplace=True)

        self.final_rec_layer = nn.Linear(in_features_num, original_features_num)

    def forward(self, x):
        x = self.act1(self.bn1(self.fc1(x)))
        x = self.act2(self.bn2(self.fc2(x)))
        x = self.final_rec_layer(x)

        return x


class Regressor(nn.Module):
    def __init__(self, original_features_num, feature_num=512):
        super(Regressor, self).__init__()
        self.fcn_f = fcn_feature(feature_num)
        self.feature_extractor=nn.Linear(original_features_num, feature_num)
        self.bn_fe=nn.BatchNorm1d(feature_num)

        self.fc = nn.Linear(feature_num*4*2, original_features_num*feature_num)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x1, x2):
        """
        [TODO] Align original features and projected features through shared nonlinear feature branches.

        Input:
            x1: (batch, original_features_num) - original delegate embeddings.
            x2: (batch, feature_num) - projected pseudo-class features.

        Output: (batch, original_features_num * feature_num) - regressed feature-mapping vector.

"""
        pass

class Regressor_combine(nn.Module):
    def __init__(self, original_features_num, in_features_num = 512, num_classes=128):
        super(Regressor_combine, self).__init__()
        self.fcn_f = fcn_feature(in_features_num)
        self.feature_extractor=nn.Linear(original_features_num, in_features_num)
        self.bn_fe=nn.BatchNorm1d(in_features_num)

        self.fc = nn.Linear(in_features_num*8, num_classes)

        self.decoder = Decoder(in_features_num, original_features_num)
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x1, x2):
        """
        [TODO] Produce paired feature embeddings, pseudo-class logits, and reconstruction.

        Input:
            x1: (batch, original_features_num) - original delegate embeddings.
            x2: (batch, in_features_num) - projected pseudo-class features.

        Output:
            x1_feature: (batch, in_features_num * 4) - encoded original-feature branch.
            x2_feature: (batch, in_features_num * 4) - encoded projected-feature branch.
            logits: (batch, num_classes) - pseudo-class classifier output.
            reconstruction: (batch, original_features_num) - reconstruction of original features.

"""
        pass


# --- [Original file: sla2p.py] ---
def softmax(input_tensor):
    act = nn.Softmax(dim=1)
    return act(input_tensor).numpy()

def neg_entropy(score):
    if len(score.shape) != 1:
        score = np.squeeze(score)
    return score@np.log2(score+1e-16)

def dist_calc(feats1, feats2):
    nb_data1 = feats1.shape[0]
    nb_data2 = feats2.shape[0]
    omega = np.dot(np.sum(feats1 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data2)))
    omega += np.dot(np.sum(feats2 ** 2, axis=1)[:, np.newaxis], np.ones(shape=(1, nb_data1))).T
    omega -= 2 * np.dot(feats1, feats2.T)
    return omega


def l2_loss(score,y):
    """
    [TODO] Compute the SLA/SLA2P pseudo-class probability mismatch score.

    Input:
        score: (classes,) or (batch, classes) - predicted pseudo-class probability vector(s).
        y: (classes,) or broadcast-compatible target - one-hot pseudo-class target.

    Output: scalar or (batch,) - mean squared mismatch over the pseudo-class dimension.

"""
    pass


def test_self_supervised_perturb(testloader, model, criterion, noisemagnitude):
    """
    [TODO] Run SLA2P adversarial perturbation inference for self-supervised logits.

    Input:
        testloader: iterable yielding input batches of shape (batch, feature_dim).
        model: pseudo-class classifier mapping (batch, feature_dim) to (batch, classes).
        criterion: classification loss used to obtain gradients against predicted pseudo-labels.
        noisemagnitude: float - step size for adding the adversarial input-gradient disturbance.

    Output: (num_samples, classes) - logits after perturbing every input batch.

"""
    pass


# ============================================================
# __main__: Automated test suite for 4 ablated functions
# ============================================================

if __name__ == "__main__":
    import contextlib
    from torch.utils.data import DataLoader

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

    @contextlib.contextmanager
    def cuda_passthrough_if_needed():
        if torch.cuda.is_available():
            yield
            return
        tensor_cuda = torch.Tensor.cuda
        module_cuda = nn.Module.cuda
        torch.Tensor.cuda = lambda self, *args, **kwargs: self
        nn.Module.cuda = lambda self, *args, **kwargs: self
        try:
            yield
        finally:
            torch.Tensor.cuda = tensor_cuda
            nn.Module.cuda = module_cuda

    print("=" * 70)
    print("SLA2P: Self-Supervised Anomaly Detection With Adversarial Perturbation")
    print("Automated reproduction benchmark - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==============================================================
    # Test 1/4: l2_loss
    # ==============================================================
    print("-" * 60)
    print("[Test 1/4] l2_loss - pseudo-class probability error score")
    try:
        score = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], dtype=np.float32)
        target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        loss = l2_loss(score, target)
        check("l2_loss output not None", loss is not None)
        if loss is not None:
            expected = ((score - target) ** 2).mean(axis=-1)
            check("l2_loss output shape", np.shape(loss) == (2,), f"expected (2,), got {np.shape(loss)}")
            check("l2_loss output finite", np.isfinite(loss).all())
            check("l2_loss computes mean squared error per sample", np.allclose(loss, expected, atol=1e-7))
        else:
            skip_checks(3, "l2_loss returned None")
    except Exception as e:
        skip_checks(4, f"l2_loss raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 2/4: Regressor.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 2/4] Regressor.forward - original/projected feature alignment")
    try:
        model = Regressor(original_features_num=3, feature_num=4).to(device)
        model.train()
        x1 = torch.randn(3, 3, device=device)
        x2 = torch.randn(3, 4, device=device)
        y = model(x1, x2)
        check("Regressor output not None", y is not None)
        if y is not None:
            check("Regressor output shape", tuple(y.shape) == (3, 12), f"expected (3, 12), got {tuple(y.shape)}")
            check("Regressor output finite", torch.isfinite(y).all().item())
            y.sum().backward()
            grad_input_branch = model.feature_extractor.weight.grad is not None and model.feature_extractor.weight.grad.abs().sum().item() > 0
            grad_projection_branch = model.fcn_f.fc1.weight.grad is not None and model.fcn_f.fc1.weight.grad.abs().sum().item() > 0
            check("Regressor gradients reach both feature branches", grad_input_branch and grad_projection_branch)
        else:
            skip_checks(3, "Regressor.forward returned None")
    except Exception as e:
        skip_checks(4, f"Regressor.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 3/4: Regressor_combine.forward
    # ==============================================================
    print("-" * 60)
    print("[Test 3/4] Regressor_combine.forward - feature, class, and reconstruction outputs")
    try:
        model = Regressor_combine(original_features_num=3, in_features_num=4, num_classes=5).to(device)
        model.train()
        x1 = torch.randn(3, 3, device=device)
        x2 = torch.randn(3, 4, device=device)
        result = model(x1, x2)
        check("Regressor_combine returns four outputs", isinstance(result, tuple) and len(result) == 4)
        if isinstance(result, tuple) and len(result) == 4 and all(item is not None for item in result):
            z1, z2, logits, rec = result
            shape_ok = tuple(z1.shape) == (3, 16) and tuple(z2.shape) == (3, 16) and tuple(logits.shape) == (3, 5) and tuple(rec.shape) == (3, 3)
            finite_ok = all(torch.isfinite(item).all().item() for item in result)
            (logits.sum() + rec.sum()).backward()
            decoder_grad = model.decoder.final_rec_layer.weight.grad is not None and model.decoder.final_rec_layer.weight.grad.abs().sum().item() > 0
            classifier_grad = model.fc.weight.grad is not None and model.fc.weight.grad.abs().sum().item() > 0
            check("Regressor_combine output shapes", shape_ok)
            check("Regressor_combine outputs finite", finite_ok)
            check("Regressor_combine drives classifier and decoder branches", decoder_grad and classifier_grad)
        else:
            skip_checks(3, "Regressor_combine.forward returned None or malformed output")
    except Exception as e:
        skip_checks(4, f"Regressor_combine.forward raised {type(e).__name__}: {e}")
    print()

    # ==============================================================
    # Test 4/4: test_self_supervised_perturb
    # ==============================================================
    print("-" * 60)
    print("[Test 4/4] test_self_supervised_perturb - adversarial uncertainty inference")
    try:
        model = fcn(in_features_num=4, class_num=3)
        model.train()
        criterion = nn.CrossEntropyLoss()
        samples = torch.randn(5, 4)
        loader = DataLoader(samples, batch_size=2, shuffle=False)
        with cuda_passthrough_if_needed():
            model = model.cuda()
            with torch.no_grad():
                clean_logits = model(samples.float().cuda()).detach().cpu()
            perturbed_logits = test_self_supervised_perturb(loader, model, criterion, noisemagnitude=0.05)
        check("perturbation inference output not None", perturbed_logits is not None)
        if perturbed_logits is not None:
            changed = not torch.allclose(perturbed_logits, clean_logits, atol=1e-6)
            check("perturbation inference output shape", tuple(perturbed_logits.shape) == (5, 3), f"expected (5, 3), got {tuple(perturbed_logits.shape)}")
            check("perturbation inference output finite", torch.isfinite(perturbed_logits).all().item())
            check("perturbation inference changes logits after gradient step", changed)
        else:
            skip_checks(3, "test_self_supervised_perturb returned None")
    except Exception as e:
        skip_checks(4, f"test_self_supervised_perturb raised {type(e).__name__}: {e}")
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
