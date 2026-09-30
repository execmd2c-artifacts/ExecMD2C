# ============================================================
# ground_truth.py - Type4Py Core Model Components
# Source: Computer_Code/type4py-main
#
# Contains ONLY the neural model definitions for deep similarity
# learning-based type inference and their direct dependencies.
# No training loops, KNN indexing, datasets, prediction pipelines,
# server code, CLI wrappers, or checkpoint I/O.
# ============================================================

import torch
import torch.nn as nn


# --- [Original file: type4py/learn.py] ---

class Type4Py(nn.Module):
    """
    Complete model
    """
    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int,
                 num_layers: int, output_size: int, dropout_rate: float):
        super(Type4Py, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)
        self.linear = nn.Linear(self.hidden_size * 2 * 2 + self.aval_type_size, self.output_size)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_id, x_tok, x_type):
        # Using dropout on input sequences
        x_id = self.dropout(x_id)
        x_tok = self.dropout(x_tok)

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]

        x = torch.cat((x_id, x_tok, x_type), 1)

        x = self.linear(x)
        return x


class Type4PyWOI(nn.Module):
    """
    Type4Py without the identifier RNN
    """
    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int,
                 num_layers: int, output_size: int, dropout_rate: float):
        super(Type4PyWOI, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2 + self.aval_type_size, self.output_size)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_tok, x_type):

        # Using dropout on input sequences
        x_tok = self.dropout(x_tok)

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_tok.flatten_parameters()

        x_tok, _ = self.lstm_tok(x_tok)

        # Decode the hidden state of the last time step
        x_tok = x_tok[:, -1, :]

        x = torch.cat((x_tok, x_type), 1)

        x = self.linear(x)
        return x


class Type4PyWOC(nn.Module):
    """
    Type4Py without code context
    """
    def __init__(self, input_size: int, hidden_size: int, aval_type_size: int,
                 num_layers: int, output_size: int, dropout_rate: float):
        super(Type4PyWOC, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.aval_type_size = aval_type_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2 + self.aval_type_size, self.output_size)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_id, x_type):

        # Using dropout on input sequences
        x_id = self.dropout(x_id)

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]

        x = torch.cat((x_id, x_type), 1)

        x = self.linear(x)
        return x


class Type4PyWOV(nn.Module):
    """
    Type4Py model without visible type hints
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int,
                 dropout_rate: float):
        super(Type4PyWOV, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm_id = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                               bidirectional=True)
        self.lstm_tok = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True,
                                bidirectional=True)

        self.linear = nn.Linear(self.hidden_size * 2 * 2, self.output_size)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x_id, x_tok):

        # Using dropout on input sequences
        x_id = self.dropout(x_id)
        x_tok = self.dropout(x_tok)

        # Flattens LSTMs weights for data-parallelism in multi-GPUs config
        self.lstm_id.flatten_parameters()
        self.lstm_tok.flatten_parameters()

        x_id, _ = self.lstm_id(x_id)
        x_tok, _ = self.lstm_tok(x_tok)

        # Decode the hidden state of the last time step
        x_id = x_id[:, -1, :]
        x_tok = x_tok[:, -1, :]

        x = torch.cat((x_id, x_tok), 1)

        x = self.linear(x)
        return x


class TripletModel(nn.Module):
    """
    A model with Triplet loss for similarity learning
    """
    def __init__(self, model: nn.Module):
        super(TripletModel, self).__init__()
        self.model = model

    def forward(self, a, p, n):
        """
        A triplet consists of anchor, positive examples and negative examples
        """
        # return self.model(*(s.to(DEVICE) for s in a)), \
        #        self.model(*(s.to(DEVICE) for s in p)), \
        #        self.model(*(s.to(DEVICE) for s in n))

        return self.model(*(s for s in a)), \
               self.model(*(s for s in p)), \
               self.model(*(s for s in n))


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

    def sample_inputs(batch=3, id_len=4, tok_len=5, input_size=6, aval_type_size=7):
        x_id = torch.randn(batch, id_len, input_size)
        x_tok = torch.randn(batch, tok_len, input_size)
        x_type = torch.randn(batch, aval_type_size)
        return x_id, x_tok, x_type

    print("=" * 70)
    print("Type4Py: automated benchmark for deep similarity type embeddings")
    print("Automated Test Suite - 5 ablated functions, 31 checks")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/5: Type4Py.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/5] Type4Py.forward - complete identifier/context/type-hint embedder")
    try:
        model = Type4Py(input_size=6, hidden_size=4, aval_type_size=7, num_layers=1, output_size=5, dropout_rate=0.0).to(device)
        model.eval()
        x_id, x_tok, x_type = sample_inputs()
        output = model(x_id, x_tok, x_type)
        check("Type4Py output not None", output is not None)
        if output is not None:
            check("Type4Py output shape", tuple(output.shape) == (3, 5), f"got {tuple(output.shape)}")
            check("Type4Py output finite", torch.isfinite(output).all().item())
            with torch.no_grad():
                id_seq, _ = model.lstm_id(x_id)
                tok_seq, _ = model.lstm_tok(x_tok)
                expected_input = torch.cat((id_seq[:, -1, :], tok_seq[:, -1, :], x_type), 1)
                expected = model.linear(expected_input)
            check("Type4Py last-step concat semantic", torch.allclose(output, expected, atol=1e-6))
            output.sum().backward()
            check("Type4Py id LSTM grad", model.lstm_id.weight_ih_l0.grad is not None and torch.isfinite(model.lstm_id.weight_ih_l0.grad).all().item())
            check("Type4Py tok LSTM grad", model.lstm_tok.weight_ih_l0.grad is not None and torch.isfinite(model.lstm_tok.weight_ih_l0.grad).all().item())
        else:
            skip_checks(5, "Type4Py.forward returned None")
    except Exception as exc:
        skip_checks(6, f"Type4Py.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/5: Type4PyWOI.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 2/5] Type4PyWOI.forward - context plus visible-type embedder")
    try:
        model = Type4PyWOI(input_size=6, hidden_size=4, aval_type_size=7, num_layers=1, output_size=5, dropout_rate=0.0).to(device)
        model.eval()
        _, x_tok, x_type = sample_inputs()
        output = model(x_tok, x_type)
        check("Type4PyWOI output not None", output is not None)
        if output is not None:
            check("Type4PyWOI output shape", tuple(output.shape) == (3, 5), f"got {tuple(output.shape)}")
            check("Type4PyWOI output finite", torch.isfinite(output).all().item())
            with torch.no_grad():
                tok_seq, _ = model.lstm_tok(x_tok)
                expected = model.linear(torch.cat((tok_seq[:, -1, :], x_type), 1))
            check("Type4PyWOI concat semantic", torch.allclose(output, expected, atol=1e-6))
            output.sum().backward()
            check("Type4PyWOI tok LSTM grad", model.lstm_tok.weight_ih_l0.grad is not None and torch.isfinite(model.lstm_tok.weight_ih_l0.grad).all().item())
        else:
            skip_checks(4, "Type4PyWOI.forward returned None")
    except Exception as exc:
        skip_checks(5, f"Type4PyWOI.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/5: Type4PyWOC.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 3/5] Type4PyWOC.forward - identifier plus visible-type embedder")
    try:
        model = Type4PyWOC(input_size=6, hidden_size=4, aval_type_size=7, num_layers=1, output_size=5, dropout_rate=0.0).to(device)
        model.eval()
        x_id, _, x_type = sample_inputs()
        output = model(x_id, x_type)
        check("Type4PyWOC output not None", output is not None)
        if output is not None:
            check("Type4PyWOC output shape", tuple(output.shape) == (3, 5), f"got {tuple(output.shape)}")
            check("Type4PyWOC output finite", torch.isfinite(output).all().item())
            with torch.no_grad():
                id_seq, _ = model.lstm_id(x_id)
                expected = model.linear(torch.cat((id_seq[:, -1, :], x_type), 1))
            check("Type4PyWOC concat semantic", torch.allclose(output, expected, atol=1e-6))
            output.sum().backward()
            check("Type4PyWOC id LSTM grad", model.lstm_id.weight_ih_l0.grad is not None and torch.isfinite(model.lstm_id.weight_ih_l0.grad).all().item())
        else:
            skip_checks(4, "Type4PyWOC.forward returned None")
    except Exception as exc:
        skip_checks(5, f"Type4PyWOC.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/5: Type4PyWOV.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 4/5] Type4PyWOV.forward - identifier plus context embedder without visible types")
    try:
        model = Type4PyWOV(input_size=6, hidden_size=4, num_layers=1, output_size=5, dropout_rate=0.0).to(device)
        model.eval()
        x_id, x_tok, _ = sample_inputs()
        output = model(x_id, x_tok)
        check("Type4PyWOV output not None", output is not None)
        if output is not None:
            check("Type4PyWOV output shape", tuple(output.shape) == (3, 5), f"got {tuple(output.shape)}")
            check("Type4PyWOV output finite", torch.isfinite(output).all().item())
            with torch.no_grad():
                id_seq, _ = model.lstm_id(x_id)
                tok_seq, _ = model.lstm_tok(x_tok)
                expected = model.linear(torch.cat((id_seq[:, -1, :], tok_seq[:, -1, :]), 1))
            check("Type4PyWOV concat semantic", torch.allclose(output, expected, atol=1e-6))
            output.sum().backward()
            check("Type4PyWOV id LSTM grad", model.lstm_id.weight_ih_l0.grad is not None and torch.isfinite(model.lstm_id.weight_ih_l0.grad).all().item())
            check("Type4PyWOV tok LSTM grad", model.lstm_tok.weight_ih_l0.grad is not None and torch.isfinite(model.lstm_tok.weight_ih_l0.grad).all().item())
        else:
            skip_checks(5, "Type4PyWOV.forward returned None")
    except Exception as exc:
        skip_checks(6, f"Type4PyWOV.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/5: TripletModel.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 5/5] TripletModel.forward - shared embedder over anchor/positive/negative")
    try:
        base = Type4Py(input_size=6, hidden_size=4, aval_type_size=7, num_layers=1, output_size=5, dropout_rate=0.0).to(device)
        base.eval()
        model = TripletModel(base).to(device)
        a = sample_inputs()
        p = sample_inputs()
        n = sample_inputs()
        outputs = model(a, p, n)
        check("TripletModel output not None", outputs is not None)
        if outputs is not None:
            anchor, positive, negative = outputs
            check("TripletModel tuple length", isinstance(outputs, tuple) and len(outputs) == 3)
            check("TripletModel anchor shape", tuple(anchor.shape) == (3, 5), f"got {tuple(anchor.shape)}")
            check("TripletModel positive shape", tuple(positive.shape) == (3, 5), f"got {tuple(positive.shape)}")
            check("TripletModel negative shape", tuple(negative.shape) == (3, 5), f"got {tuple(negative.shape)}")
            with torch.no_grad():
                expected_anchor = base(*a)
            check("TripletModel shared-model semantic", torch.allclose(anchor, expected_anchor, atol=1e-6))
            criterion = nn.TripletMarginLoss(margin=0.4)
            loss = criterion(anchor, positive, negative)
            loss.backward()
            check("TripletModel differentiable", base.linear.weight.grad is not None and torch.isfinite(base.linear.weight.grad).all().item())
        else:
            skip_checks(7, "TripletModel.forward returned None")
    except Exception as exc:
        skip_checks(8, f"TripletModel.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The Type4Py core model code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
