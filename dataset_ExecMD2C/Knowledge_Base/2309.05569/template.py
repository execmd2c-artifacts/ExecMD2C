"""
Self-contained ITI-GEN benchmark answer key.

This file extracts the core model logic from iti_gen/model.py: the learnable
FairToken parameters, attribute-combination indexing, fair-token prompt
embedding construction, and the two ITI-GEN training losses. Dataset loading,
CLIP loading, model saving, and Stable Diffusion generation are intentionally
excluded as pipeline code rather than reproduction targets.
"""

import itertools as it
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairToken(nn.Module):
    """
    K_m * 3 * 768 parameters
    """
    def __init__(self, num_basis=2, token_length=3, emb_dim=768):
        super().__init__()
        self.para = nn.Parameter(torch.zeros(num_basis, token_length, emb_dim), requires_grad=True)

    def forward(self):
        # TODO: Return the learnable FairToken parameter tensor so callers can
        # insert or add category-specific token embeddings during prompt
        # construction while preserving gradient flow to this parameter.
        pass


class ITI_GEN(object):
    """
    Construct iti-gen model and training pipeline
    """
    def setup_fairtoken_model(self):
        """
        Initialize the fairtoken models
        :return:
        """
        # TODO: Build one FairToken module per attribute. Each module must have
        # one basis vector set for every category in that attribute, use
        # self.args.token_length tokens, move the module to self.args.device,
        # and store the modules in self.fairtoken_model in attribute order.
        pass

    def setup_index(self):
        """
        Eg. for 2*6*9 combinations, self.index should contain three lists (for each attribute).
        Each list contain the index that one attribute's values differ while the other attributes' values are the same.
        Eg. for 2nd list -> Skin_Tone, we should include the index that gender, age are the same, while only skin_tone value is different.
        Total self.tokenized_text_queries contain 108 combinations, as shown below:
        0: + 0 0    54: - 0 0
        1: + 0 1    55: - 0 1
        2: + 0 2    56: - 0 2
        3: + 0 3    57: - 0 3
        4: + 0 4    58: - 0 4
        5: + 0 5    59: - 0 5
        6: + 0 6    60: - 0 6
        7: + 0 7    61: - 0 7
        8: + 0 8    62: - 0 8
        9: + 1 0    63: - 1 0
        ..          ..
        53: + 5 8   107: - 5 8
        the second list should contain 18 lists, as shown below:
        [
            [0, 9, 18, 27, 36, 45],
            [1, 10, 19, 28, 37, 46],
            ...
            [8, 17, 26, 35, 44, 53],
            [54, 63, 72, 81, 90, 99],
            ...
            [62, 71, 80, 89, 98, 107]
        ]
        """
        # TODO: Populate self.index with one list per attribute. For each
        # attribute, group prompt-combination indices where all other
        # attributes are fixed and only the current attribute category varies.
        # The ordering must match the Cartesian-product layout used by
        # self.cate_num_list and self.total_combination.
        pass

    def construct_fair_text_features(self, text_queries_embedding, last_word_idx):
        """
        insert fair_tokens to the original text queries' embeddings and obtain the
        corresponding text feature
        :return:
        """
        # TODO: Detach the base text-query embeddings, insert FairTokens into
        # the token window beginning at last_word_idx, and feed the edited
        # embeddings through the CLIP text transformer blocks and final layer
        # norm. The first attribute replaces the base token slice; every later
        # attribute adds its FairToken slice to the current value.
        pass

    def cos_loss(self, image_features, text_features, each_index, data_cls):
        """
        compute the cosine similarity loss, 1-cos(image, text)
        :return:
        """
        # TODO: Compute image-text cosine logits, mask each image so it only
        # contributes to text prompts with the same category value for the
        # current attribute, average valid similarities per prompt combination,
        # convert them to 1-cos losses, and normalize by the number of index
        # groups for this attribute.
        pass

    def iti_gen_loss(self, image_features, text_features, cate_num, each_index, data_cls):
        """
        compute two iti_gen losses
        :return:
        """
        # TODO: For every pair of categories in the current attribute, compare
        # the normalized image-feature direction against the matching
        # text-feature direction inside every fixed-context index group. Also
        # compute the semantic consistency hinge against self.ori_text_feature.
        # Average each pair over groups, then scale the accumulated pair losses
        # by the ITI-GEN category-count normalization.
        pass


class _IdentityLayerNorm(nn.Module):
    def forward(self, x):
        return x


def _make_core_model(cate_num_list=(2, 3), token_length=2, emb_dim=4, lam=0.25):
    model = object.__new__(ITI_GEN)
    model.args = SimpleNamespace(token_length=token_length, device="cpu", lam=lam)
    model.cate_num_list = list(cate_num_list)
    model.attr_num = len(model.cate_num_list)
    model.total_combination = 1
    for cate_num in model.cate_num_list:
        model.total_combination *= cate_num
    model.clip_layers_num = 0
    model.clip_model = SimpleNamespace(dtype=torch.float32, ln_final=_IdentityLayerNorm())
    model.setup_index()
    model.fairtoken_model = [
        FairToken(num_basis=cate_num, token_length=token_length, emb_dim=emb_dim)
        for cate_num in model.cate_num_list
    ]
    return model


def _as_lists(index_groups):
    return [[item.tolist() for item in attr_groups] for attr_groups in index_groups]


if __name__ == "__main__":
    torch.manual_seed(7)

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
    print("ITI-GEN benchmark: fair-token indexing and losses")
    print("Automated Test Suite - 6 ablated functions")
    print("=" * 70)
    print()

    print("-" * 60)
    print("[Test 1/4] setup_fairtoken_model - per-attribute FairToken modules")
    setup_checks = 4
    try:
        model = object.__new__(ITI_GEN)
        model.args = SimpleNamespace(token_length=5, device="cpu")
        model.cate_num_list = [2, 4]
        model.setup_fairtoken_model()
        check("fairtoken module count", len(model.fairtoken_model) == 2)
        check("first fairtoken shape", tuple(model.fairtoken_model[0]().shape) == (2, 5, 768))
        check("second fairtoken shape", tuple(model.fairtoken_model[1]().shape) == (4, 5, 768))
        check("fairtoken requires grad", model.fairtoken_model[0]().requires_grad)
    except Exception as exc:
        skip_checks(setup_checks, f"setup_fairtoken_model raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 2/4] setup_index - Cartesian attribute-difference groups")
    index_checks = 2
    try:
        model = _make_core_model(cate_num_list=(2, 3, 2), token_length=1, emb_dim=2)
        expected = [
            [[0, 6], [1, 7], [2, 8], [3, 9], [4, 10], [5, 11]],
            [[0, 2, 4], [1, 3, 5], [6, 8, 10], [7, 9, 11]],
            [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        ]
        has_index = hasattr(model, "index")
        check("index attribute exists", has_index)
        if has_index:
            check("index groups match expected layout", _as_lists(model.index) == expected)
        else:
            skip_checks(index_checks - 1, "setup_index did not create self.index")
    except Exception as exc:
        skip_checks(index_checks, f"setup_index raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 3/4] construct_fair_text_features - insertion and gradient flow")
    insertion_checks = 7
    try:
        model = _make_core_model(cate_num_list=(2, 3), token_length=2, emb_dim=4)
        with torch.no_grad():
            model.fairtoken_model[0].para.copy_(torch.arange(16, dtype=torch.float32).view(2, 2, 4) + 10.0)
            model.fairtoken_model[1].para.copy_(torch.arange(24, dtype=torch.float32).view(3, 2, 4) + 100.0)

        last_word_idx = 3
        text_queries_embedding = torch.arange(6 * 7 * 4, dtype=torch.float32).view(6, 7, 4) / 1000.0
        original = text_queries_embedding.clone()
        output = model.construct_fair_text_features(text_queries_embedding, last_word_idx)
        check("fair text output not None", output is not None)
        if output is None:
            skip_checks(insertion_checks - 1, "construct_fair_text_features returned None")
        else:
            check("fair text output shape", tuple(output.shape) == (6, 7, 4), f"got {tuple(output.shape)}")
            check("prefix tokens unchanged", torch.allclose(output[:, :last_word_idx, :], original[:, :last_word_idx, :]))
            check("suffix tokens unchanged", torch.allclose(output[:, last_word_idx + 2:, :], original[:, last_word_idx + 2:, :]))
            slices_match = True
            for row in range(6):
                first_attr_category = row // 3
                second_attr_category = row % 3
                expected_slice = (
                    model.fairtoken_model[0].para[first_attr_category]
                    + model.fairtoken_model[1].para[second_attr_category]
                )
                slices_match = slices_match and torch.allclose(
                    output[row, last_word_idx:last_word_idx + 2, :], expected_slice
                )
            check("fair tokens inserted by attribute category", slices_match)
            loss = output[:, last_word_idx:last_word_idx + 2, :].sum()
            loss.backward()
            check("first fairtoken gradient count", torch.allclose(
                model.fairtoken_model[0].para.grad,
                torch.full_like(model.fairtoken_model[0].para, 3.0),
            ))
            check("second fairtoken gradient count", torch.allclose(
                model.fairtoken_model[1].para.grad,
                torch.full_like(model.fairtoken_model[1].para, 2.0),
            ))
    except Exception as exc:
        skip_checks(insertion_checks, f"construct_fair_text_features raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Test 4/4] cos_loss and iti_gen_loss - matching and consistency objectives")
    loss_checks = 8
    try:
        model = _make_core_model(cate_num_list=(2, 3), token_length=2, emb_dim=4, lam=0.25)
        basis = torch.eye(3, dtype=torch.float32)
        labels = torch.tensor([0, 1, 2], dtype=torch.long)
        text_features = torch.vstack([basis[0], basis[1], basis[2], basis[0], basis[1], basis[2]])

        perfect_image_features = basis[labels]
        shifted_image_features = basis[(labels + 1) % 3]
        matching_loss = model.cos_loss(perfect_image_features, text_features, model.index[1], labels)
        shifted_loss = model.cos_loss(shifted_image_features, text_features, model.index[1], labels)
        check("cos_loss matching finite", torch.isfinite(matching_loss).item())
        check("cos_loss matching near zero", matching_loss.item() < 1e-6, f"got {matching_loss.item():.6f}")
        check("cos_loss shifted larger", shifted_loss.item() > matching_loss.item() + 2.0,
              f"matching={matching_loss.item():.6f}, shifted={shifted_loss.item():.6f}")

        model.ori_text_feature = torch.ones_like(text_features)
        direction_loss, consistency_loss = model.iti_gen_loss(
            perfect_image_features, text_features, 3, model.index[1], labels
        )
        check("iti_gen direction finite", torch.isfinite(direction_loss).item())
        check("iti_gen consistency finite", torch.isfinite(consistency_loss).item())
        check("iti_gen direction near zero", direction_loss.item() < 1e-6, f"got {direction_loss.item():.6f}")
        check("iti_gen consistency near zero", consistency_loss.item() < 1e-6, f"got {consistency_loss.item():.6f}")

        model.ori_text_feature = torch.zeros_like(text_features)
        _, weak_consistency_loss = model.iti_gen_loss(
            perfect_image_features, text_features, 3, model.index[1], labels
        )
        check("iti_gen consistency penalizes weak alignment",
              weak_consistency_loss.item() > consistency_loss.item() + 0.1,
              f"weak={weak_consistency_loss.item():.6f}, good={consistency_loss.item():.6f}")
    except Exception as exc:
        skip_checks(loss_checks, f"loss tests raised {type(exc).__name__}: {exc}")
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
