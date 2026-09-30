"""
ground_truth.py for Meta-rPPG core model components.

Source-consolidated from:
- model/sub_model.py
- model/loss.py

Only the neural rPPG feature encoder, rPPG estimator, synthetic-gradient
generator, ordinal regression layer, and direct loss components are included.
Data preparation, parameter-update procedures, validation routines, saved-state
I/O, plotting utilities, and CLI configuration are intentionally excluded.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- [Original file: model/sub_model.py] ---
class Synthetic_Gradient_Generator(nn.Module):
   def __init__(self, input_channel, isTrain, device):
      super(Synthetic_Gradient_Generator, self).__init__()
      self.layer1 = nn.Sequential(
          nn.Conv1d(60, 40, kernel_size=3, padding=1),
          nn.BatchNorm1d(40),
          nn.ReLU()
      )
      self.layer2 = nn.Sequential(
          nn.Conv1d(40, 20, kernel_size=3, padding=1),
          nn.BatchNorm1d(20),
          nn.ReLU()
      )
      self.layer3 = nn.Sequential(
          nn.ConvTranspose1d(20, 40, kernel_size=3, padding=1),
          nn.BatchNorm1d(40),
          nn.ReLU()
      )
      self.layer4 = nn.Sequential(
          nn.ConvTranspose1d(40, 60, kernel_size=3, padding=1)
      )

   def forward(self, x): 
      """
      TODO: Generate synthetic gradients for Meta-rPPG adaptation.

      Input:
         x: Tensor with shape (batch, 60, feature_length). For extractor
            adaptation feature_length is 120; for estimator adaptation it may
            match the ordinal probability length.
      Output:
         Tensor with shape (batch, 60, feature_length), aligned with x so it can
         be used as a synthetic gradient target.

"""
      pass


class Convolutional_Encoder(nn.Module):
   def __init__(self, input_channel, isTrain, device):
      super(Convolutional_Encoder, self).__init__()
      self.conv = nn.Conv3d
      self.conv1 = self.conv(input_channel, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv2 = self.conv(32, 48, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv3 = self.conv(48, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv4 = self.conv(64, 80, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
      self.conv5 = self.conv(80, 120, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))

      self.bn1 = nn.BatchNorm3d(32)
      self.bn2 = nn.BatchNorm3d(48)
      self.bn3 = nn.BatchNorm3d(64)
      self.bn4 = nn.BatchNorm3d(80)
      self.bn5 = nn.BatchNorm3d(120)

      self.cnn = {'c1': self.conv1, 'c2': self.conv2, 'c3': self.conv3, 'c4': self.conv4,
                  'c5': self.conv5, 'b1': self.bn1, 'b2': self.bn2, 'b3': self.bn3, 
                  'b4': self.bn4, 'b5': self.bn5}

      self.relu = nn.ReLU(inplace=True)
   
   def forward(self, x):
      """
      TODO: Encode a face video window into per-frame rPPG features.

      Input:
         x: Video tensor with shape (batch, win_size, 3, height, width).
      Output:
         Tensor with shape (batch, win_size, 120).

"""
      pass
   
   def return_grad(self):
      # pdb.set_trace()
      c1 = self.conv1.weight.grad.data.clone()
      c2 = self.conv2.weight.grad.data.clone()
      c3 = self.conv3.weight.grad.data.clone()
      c4 = self.conv4.weight.grad.data.clone()
      c5 = self.conv5.weight.grad.data.clone()
      b1 = self.bn1.weight.grad.data.clone()
      b2 = self.bn2.weight.grad.data.clone()
      b3 = self.bn3.weight.grad.data.clone()
      b4 = self.bn4.weight.grad.data.clone()
      b5 = self.bn5.weight.grad.data.clone()

      return {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4, 'c5': c5,
              'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5}
   

class rPPG_Estimator(nn.Module):
   def __init__(self, input_channel, num_layers, isTrain, device, num_classes=40, h=None, c=None):
      super(rPPG_Estimator, self).__init__()
      self.lstm = nn.LSTM(input_size=120, hidden_size=60,
                          num_layers=num_layers, batch_first=True, bidirectional=True)
      self.fc = nn.Linear(120, 80)
      self.h, self.c = h, c
      self.orl = OrdinalRegressionLayer()
   
   def forward(self, x):
      """
      TODO: Estimate rPPG ordinal decisions from encoded temporal features.

      Input:
         x: Tensor with shape (batch, 60, 120), containing per-frame encoded
            rPPG features.
      Output:
         Tuple (decision, prob):
            decision has shape (batch, 60), containing decoded ordinal levels.
            prob has shape (batch, 60, 40), containing ordinal probabilities.

"""
      pass
   def feed_hc(self, data):
      # pdb.set_trace()
      self.h = data[0].data
      self.c = data[1].data
      # pdb.set_trace()

   def return_grad(self):
      fc_grad = self.fc.weight.grad.data.clone()
      lstm_list = self.lstm._all_weights
      lstm_dict = {}
      for sublist in lstm_list:
         for name in sublist:
            # pdb.set_trace()
            lstm_dict[name] = self.lstm._parameters[name].grad.data.clone()
      return {'fc': fc_grad, 'lstm': lstm_dict}


class OrdinalRegressionLayer(nn.Module):
   def __init__(self):
      super(OrdinalRegressionLayer, self).__init__()

   def forward(self, x):
      """
      TODO: Convert paired ordinal logits into probabilities and decoded levels.

      Input:
         x: Tensor with shape (batch, 60, 80), where each frame has 40 pairs
            of ordinal logits.
      Output:
         Tuple (decode_c, ord_c1):
            decode_c has shape (batch, 60, 1).
            ord_c1 has shape (batch, 60, 40).

"""
      pass


# --- [Original file: model/loss.py] ---
class ordLoss(nn.Module):
   """
   Ordinal loss is defined as the average of pixelwise ordinal loss F(h, w, X, O)
   over the entire image domain:
   """

   def __init__(self):
      super(ordLoss, self).__init__()
      self.loss = 0.0

   def forward(self, orig_ord_labels, orig_target):
      """
      TODO: Compute the Meta-rPPG ordinal negative log-likelihood.

      Inputs:
         orig_ord_labels: Tensor with shape (batch, 60, 40), containing
            positive-threshold probabilities from the ordinal regression layer.
         orig_target: Integer tensor with shape (batch, 60), containing
            discretized target ordinal levels.
      Output:
         Scalar loss tensor.

"""
      pass


class KLDivLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(KLDivLoss, self).__init__()
        self.criterion = torch.nn.KLDivLoss(reduction=reduction)
      #   self.weight = weight

    def forward(self, outputs, targets):
      out = outputs.clone()
      tar = targets.clone()
      out.uniform_(0, 1)
      tar.uniform_(0, 1)
      # loss = self.criterion(F.log_softmax(out, -1), tar)
      loss = self.criterion(F.log_softmax(outputs, dim=1), F.softmax(targets, dim=1))

      return loss


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
    print("Meta-rPPG core model component benchmark")
    print("=" * 70)

    # ==========================================================
    # Test 1: Convolutional_Encoder.forward
    # ==========================================================
    print("[Test 1/6] 3D convolutional rPPG encoder")
    try:
        encoder = Convolutional_Encoder(input_channel=3, isTrain=True, device=torch.device("cpu"))
        encoder.eval()
        video = torch.randn(2, 60, 3, 32, 32)
        encoded = encoder(video)
        check("encoder output not None", encoded is not None)
        if encoded is not None:
            check("encoder output shape", tuple(encoded.shape) == (2, 60, 120), f"got {tuple(encoded.shape)}")
            check("encoder output finite", torch.isfinite(encoded).all().item())
            check("encoder preserves time length", encoded.shape[1] == video.shape[1])
            check("encoder conv5 out channels", encoder.conv5.out_channels == 120)
            loss = encoded.sum()
            loss.backward()
            check("encoder gradient reaches first conv", encoder.conv1.weight.grad is not None)
        else:
            skip_checks(5, "encoder returned None")
    except Exception as exc:
        skip_checks(6, f"encoder raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2: OrdinalRegressionLayer.forward
    # ==========================================================
    print("[Test 2/6] ordinal regression layer")
    try:
        orl = OrdinalRegressionLayer()
        logits = torch.randn(2, 60, 80)
        decision, prob = orl(logits)
        check("ordinal decision not None", decision is not None)
        if decision is not None:
            check("ordinal decision shape", tuple(decision.shape) == (2, 60, 1), f"got {tuple(decision.shape)}")
            check("ordinal probability shape", tuple(prob.shape) == (2, 60, 40), f"got {tuple(prob.shape)}")
            check("ordinal probability finite", torch.isfinite(prob).all().item())
            check("ordinal probability range", (prob >= 0).all().item() and (prob <= 1).all().item())
            decision_float = decision.float()
            check("ordinal decision integer-like", torch.allclose(decision_float, decision_float.round()))
            check("ordinal decision bounds", decision.min().item() >= 0 and decision.max().item() <= 40)
        else:
            skip_checks(6, "ordinal layer returned None")
    except Exception as exc:
        skip_checks(7, f"ordinal layer raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3: rPPG_Estimator.forward
    # ==========================================================
    print("[Test 3/6] LSTM rPPG estimator")
    try:
        estimator = rPPG_Estimator(input_channel=120, num_layers=2, isTrain=True, device=torch.device("cpu"))
        features = torch.randn(2, 60, 120)
        decision, prob = estimator(features)
        check("estimator decision not None", decision is not None)
        if decision is not None:
            check("estimator decision shape", tuple(decision.shape) == (2, 60), f"got {tuple(decision.shape)}")
            check("estimator probability shape", tuple(prob.shape) == (2, 60, 40), f"got {tuple(prob.shape)}")
            check("estimator probability finite", torch.isfinite(prob).all().item())
            check("estimator decision bounds", decision.min().item() >= 0 and decision.max().item() <= 40)
            h = torch.zeros(4, 2, 60)
            c = torch.zeros(4, 2, 60)
            estimator.feed_hc([h, c])
            decision_h, prob_h = estimator(features)
            check("estimator hidden-state path shape", tuple(prob_h.shape) == (2, 60, 40), f"got {tuple(prob_h.shape)}")
        else:
            skip_checks(5, "estimator returned None")
    except Exception as exc:
        skip_checks(6, f"estimator raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4: Synthetic_Gradient_Generator.forward
    # ==========================================================
    print("[Test 4/6] synthetic gradient generator")
    try:
        grad_net = Synthetic_Gradient_Generator(input_channel=120, isTrain=True, device=torch.device("cpu"))
        x = torch.randn(2, 60, 120, requires_grad=True)
        grad = grad_net(x)
        check("synthetic gradient not None", grad is not None)
        if grad is not None:
            check("synthetic gradient shape", tuple(grad.shape) == (2, 60, 120), f"got {tuple(grad.shape)}")
            check("synthetic gradient finite", torch.isfinite(grad).all().item())
            check("synthetic gradient residual width", grad_net.layer3[0].out_channels == 40)
            grad.sum().backward()
            check("synthetic gradient input grad exists", x.grad is not None)
            check("synthetic gradient first conv grad exists", grad_net.layer1[0].weight.grad is not None)
        else:
            skip_checks(5, "synthetic gradient returned None")
    except Exception as exc:
        skip_checks(6, f"synthetic gradient generator raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5: ordLoss and KLDivLoss
    # ==========================================================
    print("[Test 5/6] ordinal and KL losses")
    try:
        criterion = ordLoss()
        prob = torch.full((2, 60, 40), 0.75)
        target = torch.randint(low=0, high=40, size=(2, 60))
        loss = criterion(prob, target)
        kl = KLDivLoss(reduction="batchmean")
        logits_a = torch.randn(2, 40)
        logits_b = torch.randn(2, 40)
        kl_loss = kl(logits_a, logits_b)
        check("ord loss scalar", loss.dim() == 0)
        check("ord loss finite", torch.isfinite(loss).item())
        check("ord loss nonnegative", loss.item() >= 0)
        check("KL loss scalar", kl_loss.dim() == 0)
        check("KL loss finite", torch.isfinite(kl_loss).item())
    except Exception as exc:
        skip_checks(5, f"loss functions raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6: Encoder -> Estimator -> Synthetic gradient pipeline
    # ==========================================================
    print("[Test 6/6] core Meta-rPPG forward pipeline")
    try:
        encoder = Convolutional_Encoder(input_channel=3, isTrain=True, device=torch.device("cpu"))
        estimator = rPPG_Estimator(input_channel=120, num_layers=1, isTrain=True, device=torch.device("cpu"))
        grad_net = Synthetic_Gradient_Generator(input_channel=120, isTrain=True, device=torch.device("cpu"))
        video = torch.randn(1, 60, 3, 32, 32)
        inter = encoder(video)
        decision, prob = estimator(inter)
        inter_grad = grad_net(inter.detach())
        pred_grad = grad_net(prob.detach())
        check("pipeline intermediate shape", tuple(inter.shape) == (1, 60, 120), f"got {tuple(inter.shape)}")
        check("pipeline decision shape", tuple(decision.shape) == (1, 60), f"got {tuple(decision.shape)}")
        check("pipeline probability shape", tuple(prob.shape) == (1, 60, 40), f"got {tuple(prob.shape)}")
        check("extractor synthetic gradient shape", tuple(inter_grad.shape) == (1, 60, 120), f"got {tuple(inter_grad.shape)}")
        check("estimator synthetic gradient shape", tuple(pred_grad.shape) == (1, 60, 40), f"got {tuple(pred_grad.shape)}")
        check("pipeline finite outputs", torch.isfinite(inter).all().item() and torch.isfinite(prob).all().item())
    except Exception as exc:
        skip_checks(6, f"core pipeline raised {type(exc).__name__}: {exc}")
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
