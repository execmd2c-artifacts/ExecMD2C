# -*- coding: utf-8 -*-
"""Core model components extracted from Cross_Lingual_Checkworthy_Detection.

This file contains only the Transformer-based model architecture components from
model.py. Experiment orchestration, corpus readers, metric scoring, and parameter
update routines are intentionally excluded.
"""

import torch
from transformers import AutoConfig, AutoModel
import torch.nn.functional as F


class SentenceTransformer(torch.nn.Module):
    def __init__(self, args):
        super(SentenceTransformer, self).__init__()
        transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True,
                                                        output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model, config=transformer_config)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.linear = torch.nn.Sequential(
            torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(transformer_config.hidden_size, args.num_labels))

    def mean_pooling(self, token_embeddings, attention_mask):
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def forward(self, input_id, attention_mask):
        # list of input_ids and attention mask
        input_id = input_id.squeeze(dim=1)  # reduce dimension
        attention_mask = attention_mask.squeeze(dim=1)
        output = self.transformer(input_ids=input_id, attention_mask=attention_mask)
        post_encoding = self.mean_pooling(output['last_hidden_state'], attention_mask)
        attentions = output['attentions']
        features = self.dropout(post_encoding)
        logits = self.linear(features)
        predictions = F.softmax(logits, dim=1).detach().cpu().numpy()
        predictions = predictions.flatten()
        predictions = predictions[1] - predictions[0]
        return predictions, logits, attentions, features


class SentenceTransformerAdversarial(SentenceTransformer):
    def __init__(self, args):
        super(SentenceTransformerAdversarial, self).__init__(args)
        transformer_config = AutoConfig.from_pretrained(args.pretrained_model, return_dict=True,
                                                        output_attentions=True, output_hidden_states=True)
        self.transformer = AutoModel.from_pretrained(args.pretrained_model, config=transformer_config)
        self.dropout = torch.nn.Dropout(p=args.dropout)

        self.linear = \
            torch.nn.Sequential(
                torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(transformer_config.hidden_size, args.num_labels))

        self.lang_classifier = \
            torch.nn.Sequential(
                torch.nn.Linear(transformer_config.hidden_size, transformer_config.hidden_size),
                torch.nn.Tanh(),
                torch.nn.Linear(transformer_config.hidden_size, 5))  # we have 5 langs

        self.adversarial = args.adversarial

    def forward(self, input_id, attention_mask):
        # list of input_ids and attention mask
        input_id = input_id.squeeze(dim=1)  # reduce dimension
        attention_mask = attention_mask.squeeze(dim=1)
        output = self.transformer(input_ids=input_id, attention_mask=attention_mask)
        post_encoding = self.mean_pooling(output['last_hidden_state'], attention_mask)
        attentions = output['attentions']
        features= self.dropout(post_encoding)
        logits = self.linear(features)
        predictions = F.softmax(logits, dim=1).detach().cpu().numpy()
        predictions = predictions.flatten()
        predictions = predictions[1] - predictions[0]

        # if self.adversarial:
        #     reverse_features = ReverseLayerF.apply(post_encoding, grl_lambda)
        #     lang_logits = self.lang_classifier(reverse_features)
        # else:
        lang_logits = self.lang_classifier(post_encoding)

        return predictions, logits, attentions, lang_logits, features


TRANSFORMER_MODELS = {'sentence_transformer':
                          SentenceTransformer,
                      'adversarial_sentence_transformer':
                          SentenceTransformerAdversarial}


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
    print("Cross_Lingual_Checkworthy_Detection: transformer claim and language heads")
    print("=" * 70)

    class Args:
        pretrained_model = "tiny-local-transformer"
        dropout = 0.0
        num_labels = 2
        adversarial = False

    class TinyConfig:
        hidden_size = 6

    class TinyTransformer(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.embedding = torch.nn.Embedding(32, config.hidden_size)
            self.proj = torch.nn.Linear(config.hidden_size, config.hidden_size)

        def forward(self, input_ids, attention_mask):
            hidden = self.proj(self.embedding(input_ids))
            attention = torch.ones(input_ids.size(0), 1, input_ids.size(1), input_ids.size(1), device=input_ids.device)
            return {"last_hidden_state": hidden, "attentions": (attention,)}

    original_config_loader = AutoConfig.from_pretrained
    original_model_loader = AutoModel.from_pretrained
    AutoConfig.from_pretrained = staticmethod(lambda *args, **kwargs: TinyConfig())
    AutoModel.from_pretrained = staticmethod(lambda *args, **kwargs: TinyTransformer(kwargs["config"]))

    try:
        print("-" * 70)
        print("[Test 1/4] SentenceTransformer.mean_pooling")
        try:
            model = SentenceTransformer(Args())
            token_embeddings = torch.arange(2 * 4 * 6, dtype=torch.float32).view(2, 4, 6)
            attention_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 1, 0]])
            pooled = model.mean_pooling(token_embeddings, attention_mask)
            expected0 = token_embeddings[0, :2].mean(dim=0)
            expected1 = token_embeddings[1, [0, 2]].mean(dim=0)
            check("mean_pooling output not None", pooled is not None)
            if pooled is not None:
                check("mean_pooling output shape", pooled.shape == (2, 6), str(tuple(pooled.shape)))
                check("mean_pooling output finite", torch.isfinite(pooled).all().item())
                check("mean_pooling respects first mask", torch.allclose(pooled[0], expected0), str(pooled[0]))
                check("mean_pooling respects non-contiguous valid tokens", torch.allclose(pooled[1], expected1), str(pooled[1]))
            else:
                skip_checks(4, "mean_pooling returned None")
        except Exception as exc:
            skip_checks(5, f"mean_pooling raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 2/4] SentenceTransformer.forward")
        try:
            model = SentenceTransformer(Args())
            model.eval()
            input_ids = torch.tensor([[[1, 2, 3, 4]], [[4, 3, 2, 1]]])
            attention_mask = torch.tensor([[[1, 1, 1, 0]], [[1, 1, 0, 0]]])
            predictions, logits, attentions, features = model(input_ids, attention_mask)
            check("SentenceTransformer predictions not None", predictions is not None)
            check("SentenceTransformer logits not None", logits is not None)
            if logits is not None and features is not None:
                check("SentenceTransformer logits shape", logits.shape == (2, 2), str(tuple(logits.shape)))
                check("SentenceTransformer feature shape", features.shape == (2, 6), str(tuple(features.shape)))
                check("SentenceTransformer logits finite", torch.isfinite(logits).all().item())
                check("SentenceTransformer scalar ranking score", getattr(predictions, "shape", None) == (), str(getattr(predictions, "shape", None)))
                check("SentenceTransformer returns attentions tuple", isinstance(attentions, tuple) and len(attentions) == 1, str(type(attentions)))
                logits.sum().backward()
                grad_ok = any(param.grad is not None for param in model.linear.parameters())
                check("SentenceTransformer classifier receives gradients", grad_ok)
            else:
                skip_checks(6, "SentenceTransformer returned incomplete outputs")
        except Exception as exc:
            skip_checks(8, f"SentenceTransformer forward raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 3/4] SentenceTransformerAdversarial.forward")
        try:
            model = SentenceTransformerAdversarial(Args())
            model.eval()
            input_ids = torch.tensor([[[1, 5, 6, 0]], [[2, 3, 0, 0]]])
            attention_mask = torch.tensor([[[1, 1, 1, 0]], [[1, 1, 0, 0]]])
            predictions, logits, attentions, lang_logits, features = model(input_ids, attention_mask)
            check("Adversarial predictions not None", predictions is not None)
            check("Adversarial claim logits not None", logits is not None)
            check("Adversarial language logits not None", lang_logits is not None)
            if logits is not None and lang_logits is not None and features is not None:
                check("Adversarial claim logits shape", logits.shape == (2, 2), str(tuple(logits.shape)))
                check("Adversarial language logits shape", lang_logits.shape == (2, 5), str(tuple(lang_logits.shape)))
                check("Adversarial feature shape", features.shape == (2, 6), str(tuple(features.shape)))
                check("Adversarial outputs finite", torch.isfinite(logits).all().item() and torch.isfinite(lang_logits).all().item())
                check("Adversarial scalar ranking score", getattr(predictions, "shape", None) == (), str(getattr(predictions, "shape", None)))
                check("Adversarial returns attentions tuple", isinstance(attentions, tuple) and len(attentions) == 1, str(type(attentions)))
                (logits.sum() + lang_logits.sum()).backward()
                claim_grad_ok = any(param.grad is not None for param in model.linear.parameters())
                lang_grad_ok = any(param.grad is not None for param in model.lang_classifier.parameters())
                check("Adversarial claim head receives gradients", claim_grad_ok)
                check("Adversarial language head receives gradients", lang_grad_ok)
            else:
                skip_checks(8, "SentenceTransformerAdversarial returned incomplete outputs")
        except Exception as exc:
            skip_checks(11, f"SentenceTransformerAdversarial forward raised {type(exc).__name__}: {exc}")

        print("-" * 70)
        print("[Test 4/4] model registry")
        try:
            check("registry contains sentence transformer", TRANSFORMER_MODELS.get("sentence_transformer") is SentenceTransformer)
            check("registry contains adversarial transformer", TRANSFORMER_MODELS.get("adversarial_sentence_transformer") is SentenceTransformerAdversarial)
            check("registry exposes two model variants", len(TRANSFORMER_MODELS) == 2, str(TRANSFORMER_MODELS.keys()))
        except Exception as exc:
            skip_checks(3, f"registry checks raised {type(exc).__name__}: {exc}")
    finally:
        AutoConfig.from_pretrained = original_config_loader
        AutoModel.from_pretrained = original_model_loader

    print("=" * 70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
