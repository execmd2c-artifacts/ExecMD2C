# ============================================================
# ground_truth.py - PPOCoder Core Algorithm Components
# Source: Computer_Code/PPOCoder-main
#
# Contains the actor/value-head wrapper, response sampling helper,
# PPO/KL optimization logic, and direct tensor utility dependencies.
# No training loop, dataset loading, compiler invocation, or CLI code.
# ============================================================

import collections
import time
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
import torch.optim as optim
from transformers import T5ForConditionalGeneration


# --- [Original file: utils.py] ---

def flatten_dict(nested, sep='/'):
    """Flatten dictionary and concatenate nested keys with separator."""
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k:
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping):
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v
    flat = {}
    rec(nested, '', flat)
    return flat


def stack_dicts(stats_dicts):
    """Stack the values of a dict."""
    results = dict()
    for k in stats_dicts[0]:
        stats_list = [torch.flatten(d[k]) for d in stats_dicts]
        max_len = max([len(l) for l in stats_list])
        stats_list = [torch.cat((l.cpu(), torch.ones(max_len - len(l)))) for l in stats_list]
        results[k] = torch.stack(stats_list)
    return results


def add_suffix(input_dict, suffix):
    """Add suffix to dict keys."""
    return dict((k + suffix, v) for k, v in input_dict.items())


def logprobs_from_logits(logits, labels):
    """
    See: https://github.com/pytorch/pytorch/issues/563#issuecomment-330103591
    """
    logp = F.log_softmax(logits, dim=2)
    logpy = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logpy

    # logpy = torch.gather(logits, 2, labels.unsqueeze(2)).squeeze(-1)
    # logp = F.log_softmax(logpy, dim=-1)
    # return logp


def whiten(values, shift_mean=True):
    """Whiten values."""
    mean, var = torch.mean(values), torch.var(values)
    var = torch.nan_to_num(var, nan=1.0)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened += mean
    return whitened


def clip_by_value(x, tensor_min, tensor_max):
    """
    Tensor extenstion to torch.clamp
    https://github.com/pytorch/pytorch/issues/2793#issuecomment-428784713
    """
    clipped = torch.max(torch.min(x, tensor_max), tensor_min)
    return clipped


def entropy_from_logits(logits):
    """Calculate entropy from logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, axis=-1) - torch.sum(pd * logits, axis=-1)
    return entropy


def average_torch_dicts(list_of_dicts):
    """Average values of a list of dicts wiht torch tensors."""
    average_dict = dict()
    for key in list_of_dicts[0].keys():
        average_dict[key] = torch.mean(torch.stack([d[key] for d in list_of_dicts]), axis=0)
    return average_dict


def stats_to_np(stats_dict):
    """Cast all torch.tensors in dict to numpy arrays."""
    new_dict = dict()
    for k, v in stats_dict.items():
        if isinstance(v, torch.Tensor):
            new_dict[k] = v.detach().cpu().numpy()
        else:
            new_dict[k] = v
        if np.isscalar(new_dict[k]):
            new_dict[k] = float(new_dict[k])
    return new_dict


# --- [Original file: model.py] ---

class CodeT5HeadWithValueModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
        self.first_dropout = nn.Dropout(0.1)
        self.summary = nn.Linear(self.model.model_dim, 1)

    def load_base_model(self, load_model_path):
        self.model.load_state_dict(torch.load(load_model_path))

    def forward(self, input_ids, attention_mask=None, labels=None, decoder_attention_mask=None):
        """
        [TODO] Run the CodeT5 policy and attach a scalar value estimate to every decoded token.

        Input:
            input_ids: (batch, source_len) - token ids for source programs or natural-language prompts.
            attention_mask: (batch, source_len) or None - mask for valid source tokens.
            labels: (batch, target_len) or None - target/response token ids used by the decoder.
            decoder_attention_mask: (batch, target_len) or None - optional target-side mask.

        Output: tuple - (token logits, original model output object, token values).
            token logits: (batch, target_len, vocab_size).
            token values: (batch, target_len).

"""
        pass


def respond_to_batch(model, source_ids, attention_mask, max_target_length=400, top_k=5, top_p=1.0):
    """
    [TODO] Sample target-code token sequences from the wrapped policy model.

    Input:
        model: object - wrapper whose inner policy exposes sequence generation.
        source_ids: (batch, source_len) - source token ids.
        attention_mask: (batch, source_len) - valid-token mask for source_ids.
        max_target_length: int - generated sequence length limit.
        top_k: int - top-k sampling support size.
        top_p: float - nucleus sampling threshold.

    Output: (batch, generated_len) - sampled target token ids.

"""
    pass


# --- [Original file: ppo.py] ---

class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current, n_steps):
        """
        [TODO] Adapt the KL penalty coefficient toward a target KL level.

        Input:
            current: scalar - observed KL value from the latest PPO step.
            n_steps: int - number of samples or steps represented by the update.

        Output: None - updates self.value in place.

"""
        pass


class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current, n_steps):
        pass


class PPOTrainer:

    default_params = {
        "lr": 1e-5,
        "adap_kl_ctrl": True,
        "init_kl_coef": 100,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": 0.1,
        "batch_size": 48,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "device": torch.device("cuda"),
        'adam_eps': 1e-8
    }

    def __init__(self, model, ref_model, **ppo_params):
        self.ppo_params = self.default_params
        self.ppo_params.update(ppo_params)

        self.ref_model = ref_model
        self.model = model
        self.optimizer = AdamW(model.parameters(), lr=self.ppo_params['lr'], eps=self.ppo_params['adam_eps'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, factor=1. / np.cbrt(2), patience=100, verbose=True)
        self.metric = 0

        if self.ppo_params['adap_kl_ctrl']:
            self.kl_ctl = AdaptiveKLController(self.ppo_params['init_kl_coef'],
                                               self.ppo_params['target'],
                                               self.ppo_params['horizon'])
        else:
            self.kl_ctl = FixedKLController(self.ppo_params['init_kl_coef'])

    def batched_forward_pass(self, source_ids, source_mask, response_ids):
        """
        [TODO] Evaluate policy and reference models to collect token log-probs and values.

        Input:
            source_ids: (batch, source_len) - source token ids.
            source_mask: (batch, source_len) - source attention mask.
            response_ids: (batch, target_len) - sampled response token ids.

        Output: tuple - (policy log-probs, reference log-probs, values).
            policy log-probs: (batch, target_len).
            reference log-probs: (batch, target_len).
            values: (batch, target_len).

"""
        pass

    def compute_rewards(self, scores, logprobs, ref_logprobs):
        """
        [TODO] Combine execution-feedback scores with the per-token KL penalty.

        Input:
            scores: (batch, target_len) - external reward signal placed on response tokens.
            logprobs: (batch, target_len) - active policy log-probs for sampled tokens.
            ref_logprobs: (batch, target_len) - reference policy log-probs for sampled tokens.

        Output: tuple - (total rewards, non-score rewards, KL coefficient).
            total rewards: (batch, target_len).
            non-score rewards: (batch, target_len).
            KL coefficient: scalar.

"""
        pass

    def loss(self, old_logprobs, values, rewards, source_ids, source_mask, response_ids, response_ids_ref):  ##MODIFIED
        """
        [TODO] Compute the PPO clipped policy loss, clipped value loss, and diagnostic statistics.

        Input:
            old_logprobs: (batch, target_len) - fixed log-probs from the sampling policy.
            values: (batch, target_len) - fixed critic values from the sampling pass.
            rewards: (batch, target_len) - token-level rewards after execution score and KL penalty.
            source_ids: (batch, source_len) - source token ids for the update pass.
            source_mask: (batch, source_len) - source attention mask.
            response_ids: (batch, target_len) - sampled response token ids.
            response_ids_ref: (batch, target_len) - reference response ids retained for the original interface.

        Output: tuple - (policy loss, scaled value loss, flattened stats).
            policy loss: scalar tensor.
            scaled value loss: scalar tensor.
            flattened stats: dict of scalar/tensor diagnostics, including advantages and probability ratios.

"""
        pass

    def record_step_stats(self, kl_coef, **data):
        """Record training step statistics."""
        kl = data['logprobs'] - data['ref_logprobs']
        mean_kl = torch.mean(torch.sum(kl, axis=-1))
        mean_kl = torch.max(-mean_kl, mean_kl)
        mean_entropy = torch.mean(torch.sum(-data['logprobs'], axis=1))
        mean_non_score_reward = torch.mean(torch.sum(data['non_score_reward'], axis=1))
        stats = {
            'objective/kl': mean_kl,
            'objective/kl_dist': kl,
            'objective/logprobs': data['logprobs'],
            'objective/ref_logprobs': data['ref_logprobs'],
            'objective/kl_coef': kl_coef,
            'objective/entropy': mean_entropy,
            'ppo/mean_non_score_reward': mean_non_score_reward,
        }

        for k, v in data['train_stats'].items():
            stats[f'ppo/{k}'] = torch.mean(v, axis=0)
        stats['ppo/val/var_explained'] = 1 - stats['ppo/val/error'] / stats['ppo/returns/var']
        return stats


# ============================================================
# __main__: Automated test suite for 6 ablated functions
# ============================================================

if __name__ == "__main__":
    import collections.abc

    if not hasattr(collections, "Mapping"):
        collections.Mapping = collections.abc.Mapping

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

    class FakeSeq2Seq(nn.Module):
        def __init__(self, vocab_size=7, model_dim=4):
            super().__init__()
            self.model_dim = model_dim
            self.proj = nn.Linear(model_dim, vocab_size)
            self.last_generate_kwargs = None

        def forward(self, input_ids=None, attention_mask=None, labels=None, decoder_attention_mask=None, output_hidden_states=False):
            batch = input_ids.size(0)
            target_len = labels.size(1)
            base = torch.arange(batch * target_len * self.model_dim, dtype=torch.float32, device=input_ids.device)
            hidden = base.reshape(batch, target_len, self.model_dim) / 10.0
            output = type("FakeOutput", (), {})()
            output.decoder_hidden_states = (hidden - 1.0, hidden)
            output.logits = self.proj(hidden)
            return output

        def generate(self, source_ids, attention_mask=None, do_sample=True, top_k=5, top_p=1.0, max_length=400):
            self.last_generate_kwargs = {
                "attention_mask": attention_mask,
                "do_sample": do_sample,
                "top_k": top_k,
                "top_p": top_p,
                "max_length": max_length,
            }
            return torch.arange(source_ids.size(0) * max_length, device=source_ids.device).reshape(source_ids.size(0), max_length)

    class FakePPOPolicy(nn.Module):
        def __init__(self, vocab_size=6, offset=0.0):
            super().__init__()
            self.vocab_size = vocab_size
            self.scale = nn.Parameter(torch.tensor(0.25 + offset))

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            batch = input_ids.size(0)
            target_len = labels.size(1)
            vocab_axis = torch.arange(self.vocab_size, dtype=torch.float32, device=input_ids.device)
            label_axis = labels.to(torch.float32).unsqueeze(-1)
            logits = (vocab_axis.reshape(1, 1, -1) * self.scale) + (0.03 * label_axis)
            logits = logits.expand(batch, target_len, self.vocab_size)
            vpred = self.scale.expand(batch, target_len) + labels.to(torch.float32) * 0.01
            return logits, object(), vpred

    print("=" * 70)
    print("PPOCoder: automated benchmark for actor/value and PPO update logic")
    print("Automated Test Suite - 6 ablated functions, 37 checks")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/6: CodeT5HeadWithValueModel.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 1/6] CodeT5HeadWithValueModel.forward - logits plus token values")
    try:
        model = CodeT5HeadWithValueModel.__new__(CodeT5HeadWithValueModel)
        nn.Module.__init__(model)
        model.model = FakeSeq2Seq().to(device)
        model.first_dropout = nn.Dropout(0.0)
        model.summary = nn.Linear(model.model.model_dim, 1)
        input_ids = torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]], device=device)
        labels = torch.tensor([[1, 2, 3], [3, 2, 1]], device=device)
        attention_mask = (input_ids != 0).long()
        output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        check("value-head output not None", output is not None)
        if output is not None:
            logits, raw_output, values = output
            check("value-head tuple length", isinstance(output, tuple) and len(output) == 3)
            check("value-head logits shape", tuple(logits.shape) == (2, 3, 7), f"got {tuple(logits.shape)}")
            check("value-head values shape", tuple(values.shape) == (2, 3), f"got {tuple(values.shape)}")
            check("value-head logits finite", torch.isfinite(logits).all().item())
            check("value-head values finite", torch.isfinite(values).all().item())
            (values.sum() + logits.sum()).backward()
            check("value-head summary receives grad", model.summary.weight.grad is not None and model.summary.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(6, "forward returned None")
    except Exception as exc:
        skip_checks(7, f"CodeT5HeadWithValueModel.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/6: respond_to_batch
    # ==========================================================
    print("-" * 60)
    print("[Test 2/6] respond_to_batch - stochastic generation arguments")
    try:
        wrapper = type("Wrapper", (), {})()
        wrapper.model = FakeSeq2Seq().to(device)
        source_ids = torch.tensor([[1, 2, 0], [3, 4, 5]], device=device)
        attention_mask = (source_ids != 0).long()
        preds = respond_to_batch(wrapper, source_ids, attention_mask, max_target_length=5, top_k=3, top_p=0.75)
        check("respond output not None", preds is not None)
        if preds is not None:
            check("respond output shape", tuple(preds.shape) == (2, 5), f"got {tuple(preds.shape)}")
            check("respond output dtype", preds.dtype == torch.long, f"got {preds.dtype}")
            check("respond top_k propagated", wrapper.model.last_generate_kwargs["top_k"] == 3)
            check("respond sampling enabled", wrapper.model.last_generate_kwargs["do_sample"] is True)
        else:
            skip_checks(4, "respond_to_batch returned None")
    except Exception as exc:
        skip_checks(5, f"respond_to_batch raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/6: AdaptiveKLController.update
    # ==========================================================
    print("-" * 60)
    print("[Test 3/6] AdaptiveKLController.update - clipped adaptive KL coefficient")
    try:
        high = AdaptiveKLController(init_kl_coef=0.1, target=2.0, horizon=10)
        high.update(current=10.0, n_steps=5)
        low = AdaptiveKLController(init_kl_coef=0.1, target=2.0, horizon=10)
        low.update(current=0.0, n_steps=5)
        unchanged = AdaptiveKLController(init_kl_coef=0.1, target=2.0, horizon=10)
        unchanged.update(current=2.0, n_steps=5)
        check("adaptive KL high not None", high.value is not None)
        check("adaptive KL clipped high", np.isclose(high.value, 0.11), f"got {high.value}")
        check("adaptive KL clipped low", np.isclose(low.value, 0.09), f"got {low.value}")
        check("adaptive KL target stable", np.isclose(unchanged.value, 0.1), f"got {unchanged.value}")
        check("adaptive KL remains positive", high.value > 0 and low.value > 0 and unchanged.value > 0)
    except Exception as exc:
        skip_checks(5, f"AdaptiveKLController.update raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/6: PPOTrainer.batched_forward_pass
    # ==========================================================
    print("-" * 60)
    print("[Test 4/6] PPOTrainer.batched_forward_pass - policy/ref log-probs and values")
    try:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.model = FakePPOPolicy(vocab_size=6, offset=0.0).to(device)
        trainer.ref_model = FakePPOPolicy(vocab_size=6, offset=0.5).to(device)
        source_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], device=device)
        source_mask = (source_ids != 0).long()
        response_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], device=device)
        logprobs, ref_logprobs, values = trainer.batched_forward_pass(source_ids, source_mask, response_ids)
        check("batched forward output not None", logprobs is not None and ref_logprobs is not None and values is not None)
        if logprobs is not None and ref_logprobs is not None and values is not None:
            check("batched forward logprob shape", tuple(logprobs.shape) == (2, 4), f"got {tuple(logprobs.shape)}")
            check("batched forward ref shape", tuple(ref_logprobs.shape) == (2, 4), f"got {tuple(ref_logprobs.shape)}")
            check("batched forward value shape", tuple(values.shape) == (2, 4), f"got {tuple(values.shape)}")
            check("batched forward detached", not logprobs.requires_grad and not ref_logprobs.requires_grad and not values.requires_grad)
            expected = logprobs_from_logits(trainer.model(source_ids, source_mask, response_ids)[0], response_ids)
            check("batched forward gathers selected tokens", torch.allclose(logprobs, expected, atol=1e-6))
        else:
            skip_checks(5, "batched_forward_pass returned None output")
    except Exception as exc:
        skip_checks(6, f"PPOTrainer.batched_forward_pass raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 5/6: PPOTrainer.compute_rewards
    # ==========================================================
    print("-" * 60)
    print("[Test 5/6] PPOTrainer.compute_rewards - KL penalty plus external score")
    try:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.kl_ctl = FixedKLController(0.2)
        scores = torch.tensor([[0.0, 0.0, 1.0], [0.5, 0.0, -0.5]], device=device)
        logprobs = torch.tensor([[-1.0, -0.5, -0.25], [-0.2, -0.3, -0.4]], device=device)
        ref_logprobs = torch.tensor([[-1.5, -0.25, -0.75], [-0.5, -0.1, -0.6]], device=device)
        rewards, non_score_reward, kl_coef = trainer.compute_rewards(scores, logprobs, ref_logprobs)
        expected_non_score = -0.2 * (logprobs - ref_logprobs)
        expected_rewards = expected_non_score + scores
        check("compute rewards output not None", rewards is not None and non_score_reward is not None)
        if rewards is not None and non_score_reward is not None:
            check("compute rewards shape", tuple(rewards.shape) == (2, 3), f"got {tuple(rewards.shape)}")
            check("compute rewards finite", torch.isfinite(rewards).all().item())
            check("compute non-score KL penalty", torch.allclose(non_score_reward, expected_non_score, atol=1e-6))
            check("compute final reward", torch.allclose(rewards, expected_rewards, atol=1e-6))
            check("compute KL coefficient returned", np.isclose(kl_coef, 0.2), f"got {kl_coef}")
        else:
            skip_checks(5, "compute_rewards returned None output")
    except Exception as exc:
        skip_checks(6, f"PPOTrainer.compute_rewards raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 6/6: PPOTrainer.loss
    # ==========================================================
    print("-" * 60)
    print("[Test 6/6] PPOTrainer.loss - GAE, clipped policy loss, clipped value loss")
    try:
        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.ppo_params = {
            "gamma": 0.9,
            "lam": 0.95,
            "cliprange": 0.2,
            "cliprange_value": 0.2,
            "vf_coef": 0.3,
        }
        trainer.model = FakePPOPolicy(vocab_size=6, offset=0.0).to(device)
        source_ids = torch.tensor([[1, 2, 3], [4, 5, 0]], device=device)
        source_mask = (source_ids != 0).long()
        response_ids = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]], device=device)
        response_ids_ref = response_ids.clone()
        with torch.no_grad():
            old_logits, _, old_values = trainer.model(source_ids, source_mask, response_ids)
            old_logprobs = logprobs_from_logits(old_logits * 0.9, response_ids).detach()
            values = old_values.detach()
        rewards = torch.tensor([[0.1, 0.0, 0.2, 0.5], [0.3, -0.1, 0.0, 0.2]], device=device)
        loss_p, loss_v, stats = trainer.loss(old_logprobs, values, rewards, source_ids, source_mask, response_ids, response_ids_ref)
        check("loss output not None", loss_p is not None and loss_v is not None and stats is not None)
        if loss_p is not None and loss_v is not None and stats is not None:
            check("loss policy scalar", loss_p.ndim == 0, f"got shape {tuple(loss_p.shape)}")
            check("loss value scalar", loss_v.ndim == 0, f"got shape {tuple(loss_v.shape)}")
            check("loss finite", torch.isfinite(loss_p).item() and torch.isfinite(loss_v).item())
            check("loss stats include advantages", "policy/advantages" in stats and tuple(stats["policy/advantages"].shape) == (2, 4))
            check("loss stats include ratio", "policy/ratio" in stats and tuple(stats["policy/ratio"].shape) == (2, 4))
            check("loss stats finite", all(torch.isfinite(v).all().item() for v in stats.values() if isinstance(v, torch.Tensor)))
            (loss_p + loss_v).backward()
            check("loss keeps model differentiable", trainer.model.scale.grad is not None and torch.isfinite(trainer.model.scale.grad).all().item())
        else:
            skip_checks(7, "loss returned None output")
    except Exception as exc:
        skip_checks(8, f"PPOTrainer.loss raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The PPOCoder core code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some [TODO] functions may not be implemented correctly.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
