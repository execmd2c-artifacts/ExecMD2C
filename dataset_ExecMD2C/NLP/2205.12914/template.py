# ============================================================
# ground_truth.py - NID_ACLARR2022 Core Model Components
# Source: NLP/NID_ACLARR2022-main
#
# Contains ONLY model, contrastive objective, and direct MTP/CLNN view utilities.
# Runtime pipelines, corpus plumbing, cluster scoring, saved-weight I/O, and remote execution are omitted.
# ============================================================

# --- Third-party imports ---
import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForMaskedLM


# --- [Original file: utils/tools.py] ---
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


#https://github.com/huggingface/transformers/blob/master/src/transformers/data/data_collator.py#L70
def mask_tokens(inputs, tokenizer,\
    special_tokens_mask=None, mlm_probability=0.15):
        """
        TODO: Reproduce MTP masked-token preparation for MLM pre-training.

        Input:
            inputs: (batch, seq_len) integer token ids.
            tokenizer: tokenizer-like object with mask_token, convert_tokens_to_ids, length, and special-token mask support.
            special_tokens_mask: optional (batch, seq_len) mask where True positions must never be masked.
            mlm_probability: probability for sampling ordinary tokens as MLM prediction targets.

        Output:
            masked_inputs: (batch, seq_len) token ids after MLM corruption.
            labels: (batch, seq_len) original token ids at masked positions and -100 elsewhere.

"""
        pass


class view_generator:
    def __init__(self, tokenizer, rtr_prob, seed):
        set_seed(seed)
        self.tokenizer = tokenizer
        self.rtr_prob = rtr_prob
    
    def random_token_replace(self, ids):
        """
        TODO: Reproduce CLNN random token replacement view generation.

        Input:
            ids: (batch, seq_len) integer token ids.

        Output:
            (batch, seq_len) token ids after replacing sampled ordinary tokens with random vocabulary ids.

"""
        pass

    def shuffle_tokens(self, ids):
        """
        TODO: Reproduce CLNN token-shuffle view generation.

        Input:
            ids: (batch, seq_len) integer token ids.

        Output:
            (batch, seq_len) tensor where each sequence keeps special tokens fixed and shuffles only ordinary tokens.

"""
        pass


# --- [Original file: utils/contrastive.py] ---
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        TODO: Reproduce supervised contrastive loss for CLNN.

        Input:
            features: (batch, n_views, feature_dim...) contrastive features for augmented views.
            labels: optional (batch,) class labels for supervised positives.
            mask: optional (batch, batch) positive-pair mask; mutually exclusive with labels.

        Output:
            Scalar contrastive loss.

"""
        pass


# --- [Original file: model.py] ---
class BertForModel(nn.Module):
    def __init__(self,model_name, num_labels, device=None):
        super(BertForModel, self).__init__()
        self.num_labels = num_labels
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.classifier = nn.Linear(768, self.num_labels)
        self.dropout = nn.Dropout(0.1)
        self.backbone.to(self.device)
        self.classifier.to(self.device)

    def forward(self, X, output_hidden_states=False, output_attentions=False):
        """
        TODO: Reproduce the MTP BERT classification forward path.

        Input:
            X: dictionary of BERT inputs with input_ids shaped (batch, seq_len) and optional masks.
            output_hidden_states: whether to include the last-layer CLS embedding.
            output_attentions: whether to include backbone attentions.

        Output:
            Dictionary containing:
                logits: (batch, num_labels) unnormalized class logits.
                hidden_states: optional (batch, hidden_size) CLS embedding before dropout.
                attentions: optional backbone attention outputs.

"""
        pass

    def mlmForward(self, X, Y):
        outputs = self.backbone(**X, labels=Y)
        return outputs.loss

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y)
        return output
class CLBert(nn.Module):
    def __init__(self,model_name, device, feat_dim=128):
        super(CLBert, self).__init__()
        self.model_name = model_name
        self.device = device
        self.backbone = AutoModelForMaskedLM.from_pretrained(self.model_name)
        hidden_size = self.backbone.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, feat_dim)
        )
        self.backbone.to(self.device)
        self.head.to(device)
        
    def forward(self, X, output_hidden_states=False, output_attentions=False, output_logits=False):
        """
        TODO: Reproduce the CLNN BERT projection forward path.

        Input:
            X: dictionary of BERT inputs with input_ids shaped (batch, seq_len) and optional masks.
            output_hidden_states: whether to include raw CLS embeddings.
            output_attentions: whether to include backbone attentions.
            output_logits: preserved source argument; not used by the source forward path.

        Output:
            Dictionary containing:
                features: (batch, feat_dim) L2-normalized projection-head features.
                hidden_states: optional (batch, hidden_size) raw CLS embedding.
                attentions: optional backbone attention outputs.

"""
        pass

    def loss_cl(self, embds, label=None, mask=None, temperature=0.07, base_temperature=0.07):
        """compute contrastive loss"""
        loss = SupConLoss(temperature=temperature, base_temperature=base_temperature)
        output = loss(embds, labels=label, mask=mask)
        return output

# ============================================================
# __main__: Automated test suite for 6 ablated functions
# ============================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

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

    class TinyTokenizer:
        mask_token = "[MASK]"
        def __len__(self):
            return 30522
        def convert_tokens_to_ids(self, token):
            return 49 if token == self.mask_token else 1
        def get_special_tokens_mask(self, val, already_has_special_tokens=True):
            arr = val.tolist() if hasattr(val, "tolist") else list(val)
            return [1 if x in (0, 101, 102) else 0 for x in arr]

    class TinyBackbone(nn.Module):
        def __init__(self, hidden_size=768, seq_len=4, with_attention=True):
            super().__init__()
            self.config = type("Cfg", (), {"hidden_size": hidden_size})()
            self.proj = nn.Linear(1, hidden_size)
            self.with_attention = with_attention
        def forward(self, input_ids=None, attention_mask=None, output_hidden_states=False, output_attentions=False, labels=None, **kwargs):
            x = input_ids.float().unsqueeze(-1)
            hidden = self.proj(x)
            out = type("Out", (), {})()
            out.hidden_states = (hidden + 0.1, hidden + 0.2)
            out.attentions = (torch.ones(input_ids.size(0), 1, input_ids.size(1), input_ids.size(1)),)
            out.attention = out.attentions
            out.loss = hidden.mean()
            return out

    print("=" * 70)
    print("NID_ACLARR2022 - benchmark suite")
    print("=" * 70)

    # ------------------------------------------------------------
    # Test 1/6: mask_tokens
    # ------------------------------------------------------------
    try:
        tokenizer = TinyTokenizer()
        ids = torch.tensor([[101, 4, 5, 0, 102], [101, 6, 7, 8, 102]], dtype=torch.long)
        masked, labels = mask_tokens(ids.clone(), tokenizer, mlm_probability=1.0)
        check("mask_tokens output not None", masked is not None and labels is not None)
        if masked is not None and labels is not None:
            check("mask_tokens shapes", tuple(masked.shape) == (2, 5) and tuple(labels.shape) == (2, 5))
            check("mask_tokens keeps special/pad labels ignored", bool((labels[:, 0] == -100).all() and (labels[:, -1] == -100).all() and labels[0, 3].item() == -100))
            check("mask_tokens masks ordinary tokens", bool((labels[0, 1:3] != -100).all() and (labels[1, 1:4] != -100).all()))
            check("mask_tokens finite integer ids", masked.dtype == torch.long and labels.dtype == torch.long)
        else:
            skip_checks(4, "mask_tokens returned None")
    except Exception as exc:
        skip_checks(5, f"mask_tokens raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 2/6: view_generator.random_token_replace
    # ------------------------------------------------------------
    try:
        tokenizer = TinyTokenizer()
        gen = view_generator(tokenizer, rtr_prob=0.25, seed=7)
        ids = torch.tensor([[101, 4, 5, 6, 102], [101, 7, 8, 9, 102]], dtype=torch.long)
        out = gen.random_token_replace(ids.clone())
        check("random_token_replace output not None", out is not None)
        if out is not None:
            check("random_token_replace shape", tuple(out.shape) == (2, 5), str(tuple(out.shape)))
            check("random_token_replace no mask token remains", not bool((out == tokenizer.convert_tokens_to_ids(tokenizer.mask_token)).any().item()))
            check("random_token_replace keeps special tokens", bool((out[:, 0] == 101).all() and (out[:, -1] == 102).all()))
            check("random_token_replace integer range", out.dtype == torch.long and int(out.max()) < len(tokenizer))
        else:
            skip_checks(4, "random_token_replace returned None")
    except Exception as exc:
        skip_checks(5, f"random_token_replace raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 3/6: view_generator.shuffle_tokens
    # ------------------------------------------------------------
    try:
        tokenizer = TinyTokenizer()
        gen = view_generator(tokenizer, rtr_prob=0.25, seed=3)
        ids = torch.tensor([[101, 4, 5, 6, 102], [101, 7, 8, 9, 102]], dtype=torch.long)
        out = gen.shuffle_tokens(ids.clone())
        check("shuffle_tokens output not None", out is not None)
        if out is not None:
            check("shuffle_tokens shape", tuple(out.shape) == (2, 5), str(tuple(out.shape)))
            check("shuffle_tokens keeps special tokens", bool((out[:, 0] == 101).all() and (out[:, -1] == 102).all()))
            check("shuffle_tokens preserves row token multisets", sorted(out[0, 1:4].tolist()) == [4, 5, 6] and sorted(out[1, 1:4].tolist()) == [7, 8, 9])
            check("shuffle_tokens returns integer ids", out.dtype == torch.long)
        else:
            skip_checks(4, "shuffle_tokens returned None")
    except Exception as exc:
        skip_checks(5, f"shuffle_tokens raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 4/6: SupConLoss.forward
    # ------------------------------------------------------------
    try:
        features = torch.randn(4, 2, 6, requires_grad=True)
        labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
        loss_fn = SupConLoss(temperature=0.2, base_temperature=0.2)
        loss = loss_fn(features, labels=labels)
        check("SupConLoss output not None", loss is not None)
        if loss is not None:
            check("SupConLoss scalar shape", tuple(loss.shape) == (), str(tuple(loss.shape)))
            check("SupConLoss finite", torch.isfinite(loss).item())
            check("SupConLoss nonnegative", loss.item() >= 0.0)
            loss.backward()
            check("SupConLoss gradients reach features", features.grad is not None and features.grad.abs().sum().item() > 0)
        else:
            skip_checks(4, "SupConLoss returned None")
    except Exception as exc:
        skip_checks(5, f"SupConLoss raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 5/6: BertForModel.forward
    # ------------------------------------------------------------
    try:
        model = BertForModel.__new__(BertForModel)
        nn.Module.__init__(model)
        model.num_labels = 3
        model.model_name = "tiny"
        model.device = "cpu"
        model.backbone = TinyBackbone(hidden_size=768)
        model.classifier = nn.Linear(768, 3)
        model.dropout = nn.Dropout(0.0)
        X = {"input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)}
        out = model.forward(X, output_hidden_states=True, output_attentions=True)
        check("BertForModel forward output not None", out is not None)
        if out is not None:
            check("BertForModel logits shape", tuple(out["logits"].shape) == (2, 3), str(tuple(out["logits"].shape)))
            check("BertForModel hidden state shape", tuple(out["hidden_states"].shape) == (2, 768), str(tuple(out["hidden_states"].shape)))
            check("BertForModel logits finite", torch.isfinite(out["logits"]).all().item())
            check("BertForModel attentions propagated", "attentions" in out and len(out["attentions"]) == 1)
        else:
            skip_checks(4, "BertForModel.forward returned None")
    except Exception as exc:
        skip_checks(5, f"BertForModel.forward raised {type(exc).__name__}: {exc}")

    # ------------------------------------------------------------
    # Test 6/6: CLBert.forward
    # ------------------------------------------------------------
    try:
        model = CLBert.__new__(CLBert)
        nn.Module.__init__(model)
        model.model_name = "tiny"
        model.device = "cpu"
        model.backbone = TinyBackbone(hidden_size=12)
        model.head = nn.Sequential(
            nn.Linear(12, 12),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(12, 5),
        )
        X = {"input_ids": torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=torch.long)}
        out = model.forward(X, output_hidden_states=True, output_attentions=True)
        check("CLBert forward output not None", out is not None)
        if out is not None:
            check("CLBert features shape", tuple(out["features"].shape) == (2, 5), str(tuple(out["features"].shape)))
            check("CLBert features finite", torch.isfinite(out["features"]).all().item())
            check("CLBert features L2-normalized", torch.allclose(out["features"].norm(dim=1), torch.ones(2), atol=1e-5))
            check("CLBert hidden and attentions optional outputs", tuple(out["hidden_states"].shape) == (2, 12) and len(out["attentions"]) == 1)
        else:
            skip_checks(4, "CLBert.forward returned None")
    except Exception as exc:
        skip_checks(5, f"CLBert.forward raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"RESULT: passed={passed} failed={failed}")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
