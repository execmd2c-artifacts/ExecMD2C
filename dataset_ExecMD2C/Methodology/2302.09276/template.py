"""
ground_truth.py for Football Match Event Forecast NMSTPP core model components.

Source-consolidated from:
- NMSTPP.py

Only the Transformer-based neural marked spatio-temporal point process model
and its direct positional-encoding dependency are included. CSV preparation,
batch iteration, parameter-update loops, full-sequence rollout scripts,
plotting, score computation, and saved-weight I/O are intentionally excluded.
"""

import numpy as np
import torch
from torch import nn


window_size = 40
hidden_dim = 1024
mutihead_attention = 1
scale_grad_by_freq = True

action = ['p', '_', 'd', 'x', 's']
zone = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
other = ['zone_s', 'zone_deltay', 'zone_deltax', 'zone_sg', 'zone_thetag']

action_emb_in = len(action)
action_emb_out = len(action)
zone_emb_in = len(zone)
zone_emb_out = len(zone)
other_lin_in = len(other) + 1
other_lin_out = len(other) + 1
input_features_len = action_emb_out + zone_emb_out + other_lin_out
hist_dim = 1 + 20 + 5
device = "cpu"


def positional_encoding(src):
  """
  TODO: Build the source sinusoidal positional encoding tensor.

  Input:
      src: tensor with shape (sequence_length, feature_dim) or an equivalent
      leading sequence dimension and feature dimension matching the embedded
      NMSTPP input.
  Output:
      float tensor with the same shape as src.

  Fill a zero-like tensor with alternating sine and cosine positional signals
  across feature dimensions, using the sequence index as position and a
  dimension-dependent frequency scale. Preserve the source behavior for the
  first position: even dimensions start at zero and odd dimensions start at one.
  """
  pass


class NMSTPP(nn.Module):
    def __init__(self):  #pick up all specification vars from the global environment
        super(NMSTPP, self).__init__()
        # for action one-hot
        self.emb_act = nn.Embedding(action_emb_in,action_emb_out,scale_grad_by_freq=scale_grad_by_freq)
        # for zone one-hot
        self.emb_zone = nn.Embedding(zone_emb_in,zone_emb_out,scale_grad_by_freq=scale_grad_by_freq)
        # for continuous features
        self.lin0 = nn.Linear(other_lin_in,other_lin_out,bias=True)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_features_len,nhead=mutihead_attention,batch_first=True,dim_feedforward=hidden_dim).to(device)

        self.lin_relu = nn.Linear(input_features_len,input_features_len)
        self.lin_deltaT = nn.Linear(input_features_len,1)
        self.lin_zone = nn.Linear(input_features_len+1,20)
        self.lin_action = nn.Linear(input_features_len+1+20,5)
        self.NN_deltaT = nn.ModuleList()
        self.NN_zone = nn.ModuleList()
        self.NN_action = nn.ModuleList()
        for num_layer_deltaT in range(1):
            self.NN_deltaT.append(nn.Linear(input_features_len,input_features_len))
            #self.NN_deltaT.append(nn.ReLU())
            #self.NN_deltaT.append(nn.Dropout(p))
        for num_layer_zone in range(1):
            self.NN_zone.append(nn.Linear(input_features_len+1,input_features_len+1))
            #self.NN_zone.append(nn.ReLU())
            #self.NN_zone.append(nn.Dropout(p))
        for num_layer_action in range(2):
            self.NN_action.append(nn.Linear(input_features_len+1+20,input_features_len+1+20))
            #self.NN_action.append(nn.ReLU())
            #self.NN_action.append(nn.Dropout(p))


        print(self)

    def forward(self, X):
        """
        TODO: Run the NMSTPP event-history forecasting forward pass.

        Input:
            X: tensor of shape (batch, window_size, 8). The first channel stores
            action ids in [0, 4], the second stores zone ids in [0, 19], and the
            remaining six channels store continuous temporal/spatial features
            including delta time.
        Output:
            tensor of shape (batch, 26), concatenating one delta-time prediction,
            twenty zone logits, and five action logits in that exact order.

        Embed the categorical action and zone histories, project the continuous
        feature channels, combine all feature groups per time step, add the
        positional encoding, and pass the sequence through the Transformer
        encoder. Use the last encoded time step as the history summary. Then
        preserve the source cascade: predict delta time first, condition the zone
        logits on delta time plus the Transformer summary, and condition the
        action logits on zone logits, delta time, and the same summary before
        concatenating all three heads.
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

    print("Running Football NMSTPP core model benchmark checks...")

    try:
        src = torch.zeros(40, input_features_len)
        pe = positional_encoding(src)
        check("positional encoding not None", pe is not None)
        if pe is not None:
            check("positional encoding shape", tuple(pe.shape) == (40, input_features_len), f"got {tuple(pe.shape)}")
            check("positional encoding finite", torch.isfinite(pe).all().item())
            check("positional encoding first even zero", torch.isclose(pe[0, 0], torch.tensor(0.0), atol=1e-6).item())
            check("positional encoding first odd one", torch.isclose(pe[0, 1], torch.tensor(1.0), atol=1e-6).item())
            check("positional encoding varies over positions", not torch.allclose(pe[0], pe[1]))
        else:
            skip_checks(5, "positional_encoding returned None")
    except Exception as exc:
        skip_checks(6, f"positional encoding checks raised {type(exc).__name__}: {exc}")

    try:
        model = NMSTPP().to(device)
        model.eval()
        check("model action embedding shape", tuple(model.emb_act.weight.shape) == (5, 5), f"got {tuple(model.emb_act.weight.shape)}")
        check("model zone embedding shape", tuple(model.emb_zone.weight.shape) == (20, 20), f"got {tuple(model.emb_zone.weight.shape)}")
        check("model continuous projection shape", model.lin0.in_features == 6 and model.lin0.out_features == 6)
        check("transformer d_model", model.encoder_layer.self_attn.embed_dim == input_features_len)
        check("delta zone action cascade widths", model.lin_deltaT.out_features == 1 and model.lin_zone.in_features == input_features_len + 1 and model.lin_action.in_features == input_features_len + 1 + 20)
        check("head module depths", len(model.NN_deltaT) == 1 and len(model.NN_zone) == 1 and len(model.NN_action) == 2)
    except Exception as exc:
        skip_checks(6, f"model structure checks raised {type(exc).__name__}: {exc}")

    try:
        model = NMSTPP().to(device)
        model.eval()
        x = torch.zeros(3, window_size, 8, device=torch.device(device))
        x[:, :, 0] = torch.randint(0, action_emb_in, (3, window_size), device=torch.device(device)).float()
        x[:, :, 1] = torch.randint(0, zone_emb_in, (3, window_size), device=torch.device(device)).float()
        x[:, :, 2:] = torch.randn(3, window_size, 6, device=torch.device(device))
        with torch.no_grad():
            out = model(x)
        check("forward output not None", out is not None)
        if out is not None:
            check("forward output shape", tuple(out.shape) == (3, hist_dim), f"got {tuple(out.shape)}")
            check("forward output finite", torch.isfinite(out).all().item())
            check("deltaT head shape", tuple(out[:, 0:1].shape) == (3, 1), f"got {tuple(out[:, 0:1].shape)}")
            check("zone logits shape", tuple(out[:, 1:21].shape) == (3, 20), f"got {tuple(out[:, 1:21].shape)}")
            check("action logits shape", tuple(out[:, 21:].shape) == (3, 5), f"got {tuple(out[:, 21:].shape)}")
            zone_probs = torch.softmax(out[:, 1:21], dim=1)
            action_probs = torch.softmax(out[:, 21:], dim=1)
            check("softmax heads normalize", torch.allclose(zone_probs.sum(dim=1), torch.ones(3, device=torch.device(device)), atol=1e-5) and torch.allclose(action_probs.sum(dim=1), torch.ones(3, device=torch.device(device)), atol=1e-5))
        else:
            skip_checks(6, "NMSTPP.forward returned None")
    except Exception as exc:
        skip_checks(7, f"forward checks raised {type(exc).__name__}: {exc}")

    try:
        model = NMSTPP().to(device)
        model.eval()
        x = torch.zeros(2, window_size, 8, device=torch.device(device))
        x[:, :, 0] = torch.randint(0, action_emb_in, (2, window_size), device=torch.device(device)).float()
        x[:, :, 1] = torch.randint(0, zone_emb_in, (2, window_size), device=torch.device(device)).float()
        x[:, :, 2:] = torch.randn(2, window_size, 6, device=torch.device(device))
        out = model(x)
        objective = out[:, 0].sum() + out[:, 1:21].sum() + out[:, 21:].sum()
        objective.backward()
        check("action embedding gradient", model.emb_act.weight.grad is not None and torch.isfinite(model.emb_act.weight.grad).all().item())
        check("zone embedding gradient", model.emb_zone.weight.grad is not None and torch.isfinite(model.emb_zone.weight.grad).all().item())
        check("continuous projection gradient", model.lin0.weight.grad is not None and torch.isfinite(model.lin0.weight.grad).all().item())
        check("transformer gradient", model.encoder_layer.self_attn.in_proj_weight.grad is not None and torch.isfinite(model.encoder_layer.self_attn.in_proj_weight.grad).all().item())
        check("cascade head gradients", model.lin_deltaT.weight.grad is not None and model.lin_zone.weight.grad is not None and model.lin_action.weight.grad is not None)
    except Exception as exc:
        skip_checks(5, f"gradient path checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
