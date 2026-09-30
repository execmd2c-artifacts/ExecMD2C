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
        return self.para


class ITI_GEN(object):
    """
    Construct iti-gen model and training pipeline
    """
    def setup_fairtoken_model(self):
        """
        Initialize the fairtoken models
        :return:
        """
        self.fairtoken_model = []
        for cate_num in self.cate_num_list:
            self.fairtoken_model.append(
                FairToken(num_basis=cate_num, token_length=self.args.token_length).to(device=self.args.device)
            )

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
        # 1, 2, 12. The multiplication accumulation before the current attribute.
        pre_multi_accu = 1
        # 108, 54, 9
        nonpre_multi_accu = self.total_combination # 108
        # 54, 9, 1
        later_multi_accu= int(self.total_combination // self.cate_num_list[0])

        self.index = []
        for attr_idx in range(self.attr_num):
            if attr_idx >= 1:
                pre_multi_accu *= self.cate_num_list[attr_idx - 1]
                nonpre_multi_accu = int(nonpre_multi_accu/self.cate_num_list[attr_idx - 1])
                later_multi_accu = int(later_multi_accu/self.cate_num_list[attr_idx])

            each_attr_index = []
            for i1 in range(pre_multi_accu):
                for i2 in range(later_multi_accu):
                    # construct a tensor with the length equals attribute category's number. skin tone: 6, age: 9...
                    sample = []
                    for i3 in range(self.cate_num_list[attr_idx]):
                        sample.append(0 + i1*nonpre_multi_accu + i2*1 + i3*later_multi_accu)
                    each_attr_index.append(torch.LongTensor(sample))
            self.index.append(each_attr_index)

    def construct_fair_text_features(self, text_queries_embedding, last_word_idx):
        """
        insert fair_tokens to the original text queries' embeddings and obtain the
        corresponding text feature
        :return:
        """
        x = text_queries_embedding.detach() # (108, 77, 768)
        # add FairToken to the corresponding place
        # for 1st attr, replace from the last word index; for 2nd and later, add to them
        for i, each_index in enumerate(self.index):
            for index in each_index:
                if i == 0:
                    x[index, last_word_idx:last_word_idx + self.args.token_length, :] = (
                    self.fairtoken_model[i])()
                else:
                    x[index, last_word_idx:last_word_idx + self.args.token_length, :] += (
                    self.fairtoken_model[i])()
        x = x.permute(1, 0, 2)
        for ll in range(self.clip_layers_num):
            x = self.clip_model.transformer.resblocks[ll](x)
        x = x.permute(1, 0, 2)
        return self.clip_model.ln_final(x).type(self.clip_model.dtype)  # (108, 77, 768)

    def cos_loss(self, image_features, text_features, each_index, data_cls):
        """
        compute the cosine similarity loss, 1-cos(image, text)
        :return:
        """
        # cosine similarity logits
        logits_i = image_features @ text_features.t()  # (n_img, n_text) # bs * 108
        # for each image, we only try to maximize the cosine similarity between the image and corresponding text,
        # which should own the same category value for that attribute with image.
        # Eg, for skin tone, image1's label is type3, so for text1, text2, ..., text6. Only text3 should have skin tone type3,
        # so we mask all the other texts.
        temp_mask = torch.zeros(logits_i.size(0), logits_i.size(1) // len(each_index)).to(self.args.device)  # bs * 108/18 (6)
        range_index = torch.arange(logits_i.size(0)).long()
        temp_mask[range_index, data_cls[range_index]] = 1  # only 0 or 1
        # extend to the whole text
        mask = torch.zeros_like(logits_i)  # bs*108
        for index in each_index:
            mask[:, index] = temp_mask
        mask = mask.to(self.args.device)
        # conduct mask operation
        logits_i = mask * logits_i
        # make summation across all images
        logits_i = logits_i.sum(dim=0)  # 108
        mask = mask.sum(dim=0)  # 108
        for i in range(self.total_combination):
            if mask[i] != 0:
                logits_i[i] = 1 - logits_i[i] / mask[i]
            else:
                logits_i[i] = 0
        # make summation for the cosine similarity loss and divide by the index length
        return logits_i.sum() / len(each_index)

    def iti_gen_loss(self, image_features, text_features, cate_num, each_index, data_cls):
        """
        compute two iti_gen losses
        :return:
        """
        # select all the combinations, eg. skin tone. We take C_6^2
        all_comb = it.combinations(range(cate_num), 2)  # 15 combinations
        comb_len = int(cate_num * (cate_num - 1) / 2)  # 15
        temp_loss_direction = torch.zeros(comb_len)
        temp_loss_con = torch.zeros(comb_len)

        # loop through all combinations
        for comb_idx, (first_class, second_class) in enumerate(all_comb):

            loss_direction, loss_con = None, None
            first_class_img_mean = image_features[data_cls == first_class].mean(dim=0) # 768
            second_class_img_mean = image_features[data_cls == second_class].mean(dim=0) # 768
            delta_img = first_class_img_mean - second_class_img_mean  # (+) - (-)
            delta_img = delta_img / delta_img.norm(dim=0, keepdim=True) # 768

            selected_class= torch.LongTensor([first_class, second_class])
            logits_con = self.ori_text_feature @ text_features.t()  # 108*108

            # for each index, select the corresponding text features and compute loss
            for i, index in enumerate(each_index):
                delta_txt = text_features[index[first_class]] - text_features[index[second_class]]
                delta_txt = delta_txt / delta_txt.norm(dim=0, keepdim=True) # 768
                logits_direction = delta_img @ delta_txt.t() # 1
                if i == 0:
                    loss_direction = (1 - logits_direction.t())  # 1
                    # logits_con[0,:] is the similarity between original text feature and all new text features.
                    # since we select two classes from cate_num of classes, we only compute their similarity
                    # with self.ori_text_feature in this index and compute the mean
                    loss_con = F.relu(self.args.lam - logits_con[0, index[selected_class]].mean())
                else:
                    loss_direction = loss_direction + (1 - logits_direction.t())
                    loss_con = loss_con + F.relu(self.args.lam - logits_con[0, index[selected_class]].mean())
            loss_direction = loss_direction / len(each_index)
            loss_con = loss_con / len(each_index)

            temp_loss_direction[comb_idx] = loss_direction
            temp_loss_con[comb_idx] = loss_con

        # when the category number is 2 for attributes, such as Male (man and woman), we have one combination.
        # when the category number is larger then 2, we have C_N^2 combinations, which N is the cate_num.
        # in order to keep the updating speed for different attributes (even a specific category in those attributes) roughly the same,
        # we keep the combination number/total loss accumulation as N/2.

        divide_num = 2 * comb_len / cate_num
        loss_direction = temp_loss_direction.sum() / divide_num
        loss_con = temp_loss_con.sum() / divide_num
        return loss_direction, loss_con


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
