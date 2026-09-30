"""
ground_truth.py for MRQA domain-agnostic QA core model components.

Source-consolidated from:
- model.py
- utils.py (kl_coef only)

Only the BERT span-QA model wrapper, adversarial domain discriminator, and the
direct KL annealing helper are included. Data iteration, run managers, evaluation
scripts, distributed launchers, CLI wrappers, saved-weight I/O, and official
metric code are intentionally excluded.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel, BertConfig


def kl_coef(i):
    """
    TODO: Compute the scalar KL annealing coefficient used by adversarial QA.

    Input:
        i: integer or float global step.
    Output:
        scalar coefficient in the interval [0, 1].

    Produce a smooth monotonic schedule that is near zero at early steps,
    equals one half at the transition step used by the source implementation,
    and approaches one for late training steps. Preserve the source curve shape
    and numerical behavior for scalar Python inputs.
    """
    pass


class DomainDiscriminator(nn.Module):
    def __init__(self, num_classes=6, input_size=768 * 2,
                 hidden_size=768, num_layers=3, dropout=0.1):
        super(DomainDiscriminator, self).__init__()
        self.num_layers = num_layers
        hidden_layers = []
        for i in range(num_layers):
            if i == 0:
                input_dim = input_size
            else:
                input_dim = hidden_size
            hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(), nn.Dropout(dropout)
            ))
        hidden_layers.append(nn.Linear(hidden_size, num_classes))
        self.hidden_layers = nn.ModuleList(hidden_layers)

    def forward(self, x):
        """
        TODO: Run the domain discriminator and return class log-probabilities.

        Input:
            x: hidden representation tensor of shape (batch, input_size).
        Output:
            tensor of shape (batch, num_classes) containing normalized
            log-probabilities over source domains.

        Pass the representation through the configured hidden layers according
        to the source loop semantics, apply the final domain classifier, and
        normalize across the class dimension so exponentiated rows sum to one.
        """
        pass


class DomainQA(nn.Module):
    def __init__(self, bert_name_or_config, num_classes=6, hidden_size=768,
                 num_layers=3, dropout=0.1, dis_lambda=0.5, concat=False, anneal=False):
        super(DomainQA, self).__init__()
        if isinstance(bert_name_or_config, BertConfig):
            self.bert = BertModel(bert_name_or_config)
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased")

        self.config = self.bert.config

        self.qa_outputs = nn.Linear(hidden_size, 2)
        # init weight
        self.qa_outputs.weight.data.normal_(mean=0.0, std=0.02)
        self.qa_outputs.bias.data.zero_()
        if concat:
            input_size = 2 * hidden_size
        else:
            input_size = hidden_size
        self.discriminator = DomainDiscriminator(num_classes, input_size, hidden_size, num_layers, dropout)

        self.num_classes = num_classes
        self.dis_lambda = dis_lambda
        self.anneal = anneal
        self.concat = concat
        self.sep_id = 102

    # only for prediction
    def forward(self, input_ids, token_type_ids, attention_mask,
                start_positions=None, end_positions=None, labels=None,
                dtype=None, global_step=22000):
        """
        TODO: Dispatch DomainQA among QA-loss, discriminator-loss, and prediction modes.

        Inputs:
            input_ids, token_type_ids, attention_mask: BERT inputs of shape
            (batch, sequence_length).
            start_positions, end_positions: optional answer span labels.
            labels: optional domain labels of shape (batch,).
            dtype: selects "qa", "dis", or prediction behavior.
        Output:
            QA scalar loss for dtype "qa"; discriminator scalar loss for dtype
            "dis"; otherwise a pair of start/end logits, each shaped
            (batch, sequence_length).

        Preserve the source dispatch behavior: QA mode calls the QA objective,
        discriminator mode requires domain labels and calls the discriminator
        objective, and prediction mode runs BERT plus the span head without
        computing losses.
        """
        pass

    def forward_qa(self, input_ids, token_type_ids, attention_mask, start_positions, end_positions, global_step):
        """
        TODO: Compute the adversarially regularized span-QA objective.

        Inputs:
            input_ids, token_type_ids, attention_mask: BERT inputs of shape
            (batch, sequence_length).
            start_positions, end_positions: span labels of shape (batch,) or
            (batch, 1).
            global_step: step value used when KL annealing is enabled.
        Output:
            scalar total loss.

        Run BERT to obtain token representations. Build the domain hidden vector
        from CLS alone or from CLS concatenated with the selected SEP
        representation. Encourage the discriminator output to match a uniform
        domain distribution, optionally scaling this term with the annealing
        schedule. In parallel, compute start/end span logits from token
        representations, squeeze extra label dimensions, clamp out-of-range span
        labels to the ignored index, average the start/end classification losses,
        and return the QA loss plus the weighted domain-uniform regularizer.
        """
        pass

    def forward_discriminator(self, input_ids, token_type_ids, attention_mask, labels):
        """
        TODO: Compute the domain discriminator objective with frozen BERT features.

        Inputs:
            input_ids, token_type_ids, attention_mask: BERT inputs of shape
            (batch, sequence_length).
            labels: integer domain labels of shape (batch,).
        Output:
            scalar discriminator loss.

        Extract CLS or CLS+SEP representations from BERT without allowing BERT
        gradients in this branch, detach the resulting hidden representation,
        classify it with the domain discriminator, and compute the negative
        log-likelihood loss against the provided domain labels.
        """
        pass

    def get_sep_embedding(self, input_ids, sequence_output):
        """
        TODO: Select the SEP-related representation used by concat mode.

        Inputs:
            input_ids: token ids of shape (batch, sequence_length).
            sequence_output: BERT token representations of shape
            (batch, sequence_length, hidden_size).
        Output:
            tensor of shape (batch, hidden_size).

        Preserve the source indexing semantics: compute a per-example index from
        how many separator tokens are present, then gather that token
        representation from the BERT sequence output for each batch item.
        """
        pass


if __name__ == "__main__":
    import numpy as np

    torch.manual_seed(42)

    passed = 0
    failed = 0

    def tensor_isfinite(tensor):
        if hasattr(torch, "isfinite"):
            return torch.isfinite(tensor).all().item()
        data = tensor.detach() if hasattr(tensor, "detach") else tensor.data
        return bool(np.isfinite(data.cpu().numpy()).all())

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

    def make_config():
        return BertConfig(
            vocab_size_or_config_json_file=305,
            hidden_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=64,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
        )

    def fake_inputs(batch=2, seq_len=12):
        input_ids = torch.randint(0, 200, (batch, seq_len), dtype=torch.long)
        input_ids[:, 0] = 101
        input_ids[:, 1] = 102
        token_type_ids = torch.zeros(batch, seq_len, dtype=torch.long)
        token_type_ids[:, 4:] = 1
        attention_mask = torch.ones(batch, seq_len, dtype=torch.long)
        start_positions = torch.tensor([2, 4], dtype=torch.long)[:batch]
        end_positions = torch.tensor([3, 5], dtype=torch.long)[:batch]
        labels = torch.tensor([0, 2], dtype=torch.long)[:batch]
        return input_ids, token_type_ids, attention_mask, start_positions, end_positions, labels

    print("Running MRQA DomainQA core model benchmark checks...")

    try:
        early = kl_coef(0)
        mid = kl_coef(3500)
        late = kl_coef(22000)
        check("kl coef finite", all(math.isfinite(v) for v in [early, mid, late]))
        check("kl coef midpoint", abs(mid - 0.5) < 1e-6, str(mid))
        check("kl coef monotonic", early < mid < late)
        check("kl coef bounded", 0.0 <= early <= 1.0 and 0.0 <= late <= 1.0)
    except Exception as exc:
        skip_checks(4, f"kl_coef checks raised {type(exc).__name__}: {exc}")

    try:
        discriminator = DomainDiscriminator(num_classes=4, input_size=32, hidden_size=16, num_layers=3, dropout=0.0)
        x = torch.randn(3, 32)
        log_prob = discriminator(x)
        check("discriminator output not None", log_prob is not None)
        if log_prob is not None:
            check("discriminator shape", tuple(log_prob.shape) == (3, 4), f"got {tuple(log_prob.shape)}")
            check("discriminator finite", tensor_isfinite(log_prob))
            prob_sum = torch.exp(log_prob).sum(dim=1)
            check("discriminator log softmax", torch.allclose(prob_sum, torch.ones_like(prob_sum), atol=1e-5))
            check("discriminator layer count", len(discriminator.hidden_layers) == 4)
        else:
            skip_checks(4, "DomainDiscriminator returned None")
    except Exception as exc:
        skip_checks(5, f"DomainDiscriminator checks raised {type(exc).__name__}: {exc}")

    try:
        model = DomainQA(make_config(), num_classes=3, hidden_size=32, num_layers=2, dropout=0.0, dis_lambda=0.2)
        model.eval()
        input_ids, token_type_ids, attention_mask, start_positions, end_positions, labels = fake_inputs()
        with torch.no_grad():
            start_logits, end_logits = model(input_ids, token_type_ids, attention_mask)
        check("prediction start not None", start_logits is not None)
        check("prediction end not None", end_logits is not None)
        if start_logits is not None and end_logits is not None:
            check("prediction start shape", tuple(start_logits.shape) == (2, 12), f"got {tuple(start_logits.shape)}")
            check("prediction end shape", tuple(end_logits.shape) == (2, 12), f"got {tuple(end_logits.shape)}")
            check("prediction finite", tensor_isfinite(start_logits) and tensor_isfinite(end_logits))
            check("qa head shape", model.qa_outputs.in_features == 32 and model.qa_outputs.out_features == 2)
        else:
            skip_checks(4, "DomainQA prediction returned None")
    except Exception as exc:
        skip_checks(6, f"DomainQA prediction checks raised {type(exc).__name__}: {exc}")

    try:
        model = DomainQA(make_config(), num_classes=3, hidden_size=32, num_layers=2, dropout=0.0, dis_lambda=0.2, concat=True, anneal=True)
        model.train()
        input_ids, token_type_ids, attention_mask, start_positions, end_positions, labels = fake_inputs()
        old_lambda = model.dis_lambda
        loss = model(input_ids, token_type_ids, attention_mask, start_positions.unsqueeze(-1), end_positions.unsqueeze(-1), labels, dtype="qa", global_step=3500)
        check("qa loss not None", loss is not None)
        if loss is not None:
            check("qa loss scalar", loss.dim() == 0, f"got {tuple(loss.shape)}")
            check("qa loss finite", tensor_isfinite(loss))
            check("qa anneal updated lambda", abs(model.dis_lambda - old_lambda * kl_coef(3500)) < 1e-6)
            loss.backward()
            check("qa bert gradient path", any(p.grad is not None for p in model.bert.parameters() if p.requires_grad))
            check("qa head gradient path", model.qa_outputs.weight.grad is not None)
        else:
            skip_checks(5, "DomainQA.forward dtype='qa' returned None")
    except Exception as exc:
        skip_checks(6, f"DomainQA QA-loss checks raised {type(exc).__name__}: {exc}")

    try:
        model = DomainQA(make_config(), num_classes=3, hidden_size=32, num_layers=2, dropout=0.0, dis_lambda=0.2, concat=True)
        model.train()
        input_ids, token_type_ids, attention_mask, start_positions, end_positions, labels = fake_inputs()
        loss = model(input_ids, token_type_ids, attention_mask, start_positions, end_positions, labels, dtype="dis")
        check("dis loss not None", loss is not None)
        if loss is not None:
            check("dis loss scalar", loss.dim() == 0, f"got {tuple(loss.shape)}")
            check("dis loss finite", tensor_isfinite(loss))
            loss.backward()
            check("discriminator gradient path", any(p.grad is not None for p in model.discriminator.parameters()))
            check("dis bert frozen by no_grad", all(p.grad is None for p in model.bert.parameters()))
        else:
            skip_checks(4, "DomainQA.forward dtype='dis' returned None")
    except Exception as exc:
        skip_checks(5, f"DomainQA discriminator checks raised {type(exc).__name__}: {exc}")

    try:
        model = DomainQA(make_config(), num_classes=3, hidden_size=32, num_layers=2, dropout=0.0, concat=True)
        input_ids = torch.tensor([[101, 102, 7, 8], [101, 9, 102, 10]], dtype=torch.long)
        sequence_output = torch.arange(2 * 4 * 32, dtype=torch.float32).view(2, 4, 32)
        sep_embedding = model.get_sep_embedding(input_ids, sequence_output)
        check("sep embedding shape", tuple(sep_embedding.shape) == (2, 32), f"got {tuple(sep_embedding.shape)}")
        check("sep embedding finite", tensor_isfinite(sep_embedding))
        expected = sequence_output[torch.arange(2), torch.tensor([1, 1])]
        check("sep embedding source semantics", torch.equal(sep_embedding, expected))
    except Exception as exc:
        skip_checks(3, f"SEP embedding checks raised {type(exc).__name__}: {exc}")

    total = passed + failed
    print(f"Checks passed: {passed}/{total}")
    if failed:
        raise SystemExit(1)
