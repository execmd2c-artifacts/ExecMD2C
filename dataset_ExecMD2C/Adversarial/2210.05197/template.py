# ============================================================
# ground_truth.py - OTTeR Core Retrieval Components
# Source: Adversarial/OTTeR-main
#
# Contains ONLY the three-part table-text retrieval representation
# modules and their direct tensor helper dependency.
# ============================================================

from transformers import AutoModel
import torch.nn as nn
import torch


# --- [Original file: retrieval/models/tb_retriever.py] ---

def pooling_masked_part(hidden, mask, method='mean'):
    """
    [TODO] Pool one masked segment from token hidden states.

    Input:
        hidden: (batch, seq_len, hidden_size) - contextual token representations.
        mask: (batch, seq_len) - 1/0 indicator for the segment to pool.
        method: str - pooling mode, including mean, sum, max, cls, first, and fallback behavior.

    Output: (batch, hidden_size) - one vector per example for the selected segment.

"""
    pass


class SingleRetrieverThreeCatPool(nn.Module):
    def __init__(self, config, args):
        super(SingleRetrieverThreeCatPool, self).__init__()
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.encoder = AutoModel.from_pretrained(args.model_name)
        self.hidden_size = config.hidden_size
        self.part_pooling = args.part_pooling
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        c_cls = self.encode_seq(batch['c_input_ids'], batch['c_mask'], batch['c_type_ids'], part2_mask=batch['c_part2_mask'], part3_mask=batch['c_part3_mask'])
        neg_c_cls = self.encode_seq(batch['neg_input_ids'], batch['neg_mask'], batch['neg_type_ids'], part2_mask=batch['neg_part2_mask'], part3_mask=batch['neg_part3_mask'])
        q_cls = self.encode_q(batch['q_input_ids'], batch['q_mask'], batch['q_type_ids'], part2_mask=batch['q_part2_mask'], part3_mask=batch['q_part3_mask'])
        return {'q': q_cls, 'c': c_cls, 'neg_c': neg_c_cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None):
        """
        [TODO] Encode a table-text block as the OTTeR three-part representation.

        Input:
            input_ids: (batch, seq_len) - token ids for table-text block sequences.
            attention_mask: (batch, seq_len) - valid-token mask for the encoder.
            token_type_ids: optional (batch, seq_len) - segment ids for encoders that use them.
            part2_mask: (batch, seq_len) - mask for the table/header portion.
            part3_mask: (batch, seq_len) - mask for the linked passage/text portion.

        Output: (batch, 3 * hidden_size) - concatenated global, table, and passage vectors.

"""
        pass

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        """
        [TODO] Encode a question sequence as the OTTeR three-part query representation.

        Input:
            input_ids: (batch, seq_len) - token ids for question sequences.
            attention_mask: (batch, seq_len) - valid-token mask for the encoder.
            token_type_ids: optional (batch, seq_len) - segment ids for encoders that use them.
            part2_mask: (batch, seq_len) - mask for the second question-related span.
            part3_mask: (batch, seq_len) - mask for the third question-related span.

        Output: (batch, 3 * hidden_size) - concatenated global and span-pooled query vectors.

"""
        pass

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        # (Batch, dim)
        # logger.info("vector.shape:{}".format(cls_rep[0].shape))
        return {'embed': q_cls}


class SingleEncoderThreeCatPool(nn.Module):
    def __init__(self, config, args):
        super(SingleEncoderThreeCatPool, self).__init__()
        self.config = config
        self.args = args
        self.shared_encoder = args.shared_encoder
        self.no_proj = args.no_proj
        self.part_pooling = args.part_pooling

        self.encoder = AutoModel.from_pretrained(args.model_name)
        if not self.shared_encoder:
            self.encoder_q = AutoModel.from_pretrained(args.model_name)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                     nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], batch['input_type_ids'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}

    def encode_seq(self, input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   output_hidden_states=True,
                   part2_mask=None,
                   part3_mask=None):
        hidden_states = self.encoder(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        # pooled_output = self.dropout(cls_rep[1])
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector

    def encode_q(self, input_ids=None,
                 attention_mask=None,
                 token_type_ids=None,
                 position_ids=None,
                 head_mask=None,
                 output_hidden_states=True,
                 part2_mask=None,
                 part3_mask=None):
        if self.shared_encoder:
            hidden_states = self.encoder(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   output_hidden_states=output_hidden_states)[0]
        else:
            hidden_states = self.encoder_q(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     position_ids=position_ids,
                                     head_mask=head_mask,
                                     output_hidden_states=output_hidden_states)[0]
        cls_rep = hidden_states[:, 0, :]
        if self.no_proj:
            part1 = cls_rep
        else:
            part1 = self.project(cls_rep)
        part2 = pooling_masked_part(hidden_states, part2_mask, method=self.part_pooling)
        part3 = pooling_masked_part(hidden_states, part3_mask, method=self.part_pooling)
        vector = torch.cat([part1, part2, part3], dim=1)
        return vector


class RobertaSingleRetrieverThreeCatPool(SingleRetrieverThreeCatPool):

    def __init__(self, config, args):
        super(RobertaSingleRetrieverThreeCatPool, self).__init__(config, args)

    def forward(self, batch):
        """
        [TODO] Encode the RoBERTa training batch into query, positive context, and negative context embeddings.

        Input:
            batch: dict containing q/c/neg token ids and masks, each with shape (batch, seq_len),
                plus q/c/neg part2 and part3 masks with shape (batch, seq_len).

        Output: dict with:
            q: (batch, 3 * hidden_size) - query embeddings.
            c: (batch, 3 * hidden_size) - positive table-text block embeddings.
            neg_c: (batch, 3 * hidden_size) - negative table-text block embeddings.

"""
        pass

    def evaluate_encode_tb(self, batch):
        c_cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': c_cls}

    def evaluate_encode_que(self, batch):
        q_cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': q_cls}


class RobertaSingleEncoderThreeCatPool(SingleEncoderThreeCatPool):

    def __init__(self, config, args):
        super(RobertaSingleEncoderThreeCatPool, self).__init__(config, args)
        self.encode_table = args.encode_table

    def forward(self, batch):
        if self.encode_table:
            cls = self.encode_seq(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        else:
            cls = self.encode_q(batch['input_ids'], batch['input_mask'], part2_mask=batch['part2_mask'], part3_mask=batch['part3_mask'])
        return {'embed': cls}


# ============================================================
# __main__: Automated test suite for 4 ablated functions
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

    class FakeEncoder(nn.Module):
        def __init__(self, hidden_size=4, vocab_size=64, offset=0.0):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            with torch.no_grad():
                values = torch.arange(vocab_size * hidden_size, dtype=torch.float32).view(vocab_size, hidden_size)
                self.embedding.weight.copy_(values / 100.0 + offset)

        def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                    position_ids=None, head_mask=None, output_hidden_states=True):
            hidden = self.embedding(input_ids)
            if attention_mask is not None:
                hidden = hidden * attention_mask.unsqueeze(-1).float()
            return (hidden,)

    def make_pool_model(shared_encoder=True, no_proj=True, part_pooling='mean'):
        model = SingleRetrieverThreeCatPool.__new__(SingleRetrieverThreeCatPool)
        nn.Module.__init__(model)
        model.shared_encoder = shared_encoder
        model.no_proj = no_proj
        model.hidden_size = 4
        model.part_pooling = part_pooling
        model.encoder = FakeEncoder(hidden_size=4, offset=0.0)
        if not shared_encoder:
            model.encoder_q = FakeEncoder(hidden_size=4, offset=1.0)
        model.project = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4, eps=1e-5))
        return model

    def make_roberta_pool_model(shared_encoder=True, no_proj=True, part_pooling='mean'):
        model = RobertaSingleRetrieverThreeCatPool.__new__(RobertaSingleRetrieverThreeCatPool)
        nn.Module.__init__(model)
        model.shared_encoder = shared_encoder
        model.no_proj = no_proj
        model.hidden_size = 4
        model.part_pooling = part_pooling
        model.encoder = FakeEncoder(hidden_size=4, offset=0.0)
        if not shared_encoder:
            model.encoder_q = FakeEncoder(hidden_size=4, offset=1.0)
        model.project = nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4, eps=1e-5))
        return model

    def sample_masks(device):
        return (
            torch.tensor([[1, 1, 0, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float32, device=device),
            torch.tensor([[0, 0, 1, 1, 1], [1, 0, 0, 1, 1]], dtype=torch.float32, device=device),
        )

    print("=" * 70)
    print("OTTeR Benchmark: Three-Part Table-Text Retriever")
    print("Automated Test Suite - 4 ablated functions")
    print("=" * 70)
    print()

    device = torch.device("cpu")
    print(f"Device: {device}")
    print()

    # ==========================================================
    # Test 1/4: pooling_masked_part
    # ==========================================================
    print("-" * 60)
    print("[Test 1/4] pooling_masked_part - masked segment pooling")
    try:
        hidden = torch.arange(40, dtype=torch.float32, device=device).view(2, 5, 4).requires_grad_(True)
        mask, _ = sample_masks(device)
        pooled_mean = pooling_masked_part(hidden, mask, method='mean')
        pooled_sum = pooling_masked_part(hidden, mask, method='sum')
        pooled_max = pooling_masked_part(hidden, mask, method='max')
        pooled_cls = pooling_masked_part(hidden, mask, method='cls')
        pooled_first = pooling_masked_part(hidden, mask, method='first')
        check("pooling mean not None", pooled_mean is not None)
        if pooled_mean is not None:
            check("pooling mean shape", pooled_mean.shape == (2, 4), f"expected (2, 4), got {tuple(pooled_mean.shape)}")
            check("pooling sum shape", pooled_sum.shape == (2, 4), f"expected (2, 4), got {tuple(pooled_sum.shape)}")
            check("pooling max shape", pooled_max.shape == (2, 4), f"expected (2, 4), got {tuple(pooled_max.shape)}")
            check("pooling cls shape", pooled_cls.shape == (2, 4), f"expected (2, 4), got {tuple(pooled_cls.shape)}")
            check("pooling first shape", pooled_first.shape == (2, 4), f"expected (2, 4), got {tuple(pooled_first.shape)}")
            check("pooling mean finite", torch.isfinite(pooled_mean).all().item())
            check("pooling sum consistency", torch.allclose(pooled_sum, pooled_mean * mask.sum(dim=1, keepdim=True)))
            check("pooling cls consistency", torch.allclose(pooled_cls, hidden[:, 0, :]))
            pooled_mean.sum().backward()
            check("pooling gradient on selected hidden", hidden.grad is not None and hidden.grad[mask.bool()].abs().sum().item() > 0)
            check("pooling no gradient on masked hidden", hidden.grad is not None and hidden.grad[(1 - mask).bool()].abs().sum().item() == 0)
        else:
            skip_checks(10, "pooling_masked_part returned None")
    except Exception as exc:
        skip_checks(11, f"pooling_masked_part raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 2/4: SingleRetrieverThreeCatPool.encode_seq
    # ==========================================================
    print("-" * 60)
    print("[Test 2/4] SingleRetrieverThreeCatPool.encode_seq - table-text triple representation")
    try:
        model = make_pool_model(shared_encoder=True, no_proj=True, part_pooling='mean').to(device)
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long, device=device)
        input_mask = torch.ones(2, 5, dtype=torch.float32, device=device)
        part2_mask, part3_mask = sample_masks(device)
        vector = model.encode_seq(input_ids, input_mask, None, part2_mask=part2_mask, part3_mask=part3_mask)
        check("encode_seq output not None", vector is not None)
        if vector is not None:
            check("encode_seq output shape", vector.shape == (2, 12), f"expected (2, 12), got {tuple(vector.shape)}")
            check("encode_seq output finite", torch.isfinite(vector).all().item())
            hidden = model.encoder(input_ids=input_ids, attention_mask=input_mask)[0]
            check("encode_seq cls slice", torch.allclose(vector[:, :4], hidden[:, 0, :]))
            check("encode_seq table slice", torch.allclose(vector[:, 4:8], pooling_masked_part(hidden, part2_mask, method='mean')))
            check("encode_seq passage slice", torch.allclose(vector[:, 8:12], pooling_masked_part(hidden, part3_mask, method='mean')))
            vector.sum().backward()
            check("encode_seq encoder gradient", model.encoder.embedding.weight.grad is not None and model.encoder.embedding.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(6, "encode_seq returned None")
    except Exception as exc:
        skip_checks(7, f"encode_seq raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 3/4: SingleRetrieverThreeCatPool.encode_q
    # ==========================================================
    print("-" * 60)
    print("[Test 3/4] SingleRetrieverThreeCatPool.encode_q - shared and separate query encoders")
    try:
        input_ids = torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], dtype=torch.long, device=device)
        input_mask = torch.ones(2, 5, dtype=torch.float32, device=device)
        part2_mask, part3_mask = sample_masks(device)
        shared_model = make_pool_model(shared_encoder=True, no_proj=True, part_pooling='sum').to(device)
        separate_model = make_pool_model(shared_encoder=False, no_proj=True, part_pooling='sum').to(device)
        shared_vec = shared_model.encode_q(input_ids, input_mask, None, part2_mask=part2_mask, part3_mask=part3_mask)
        separate_vec = separate_model.encode_q(input_ids, input_mask, None, part2_mask=part2_mask, part3_mask=part3_mask)
        check("encode_q shared output not None", shared_vec is not None)
        check("encode_q separate output not None", separate_vec is not None)
        if shared_vec is not None and separate_vec is not None:
            check("encode_q shared shape", shared_vec.shape == (2, 12), f"expected (2, 12), got {tuple(shared_vec.shape)}")
            check("encode_q separate shape", separate_vec.shape == (2, 12), f"expected (2, 12), got {tuple(separate_vec.shape)}")
            check("encode_q outputs finite", torch.isfinite(shared_vec).all().item() and torch.isfinite(separate_vec).all().item())
            check("encode_q separate encoder branch", not torch.allclose(shared_vec, separate_vec))
            separate_vec.sum().backward()
            check("encode_q query encoder gradient", separate_model.encoder_q.embedding.weight.grad is not None and separate_model.encoder_q.embedding.weight.grad.abs().sum().item() > 0)
        else:
            skip_checks(5, "encode_q returned None")
    except Exception as exc:
        skip_checks(7, f"encode_q raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Test 4/4: RobertaSingleRetrieverThreeCatPool.forward
    # ==========================================================
    print("-" * 60)
    print("[Test 4/4] RobertaSingleRetrieverThreeCatPool.forward - q/c/negative retrieval outputs")
    try:
        model = make_roberta_pool_model(shared_encoder=True, no_proj=True, part_pooling='first').to(device)
        batch = {
            'c_input_ids': torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=torch.long, device=device),
            'c_mask': torch.ones(2, 5, dtype=torch.float32, device=device),
            'c_part2_mask': torch.tensor([[0, 1, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32, device=device),
            'c_part3_mask': torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 0]], dtype=torch.float32, device=device),
            'neg_input_ids': torch.tensor([[2, 3, 4, 5, 6], [7, 8, 9, 10, 11]], dtype=torch.long, device=device),
            'neg_mask': torch.ones(2, 5, dtype=torch.float32, device=device),
            'neg_part2_mask': torch.tensor([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]], dtype=torch.float32, device=device),
            'neg_part3_mask': torch.tensor([[0, 0, 1, 1, 0], [0, 0, 0, 1, 1]], dtype=torch.float32, device=device),
            'q_input_ids': torch.tensor([[3, 4, 5, 6, 7], [8, 9, 10, 11, 12]], dtype=torch.long, device=device),
            'q_mask': torch.ones(2, 5, dtype=torch.float32, device=device),
            'q_part2_mask': torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float32, device=device),
            'q_part3_mask': torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32, device=device),
        }
        outputs = model(batch)
        check("roberta forward output not None", outputs is not None)
        if outputs is not None:
            check("roberta forward keys", set(outputs.keys()) == {'q', 'c', 'neg_c'}, f"got {sorted(outputs.keys())}")
            check("roberta q shape", outputs['q'].shape == (2, 12), f"expected (2, 12), got {tuple(outputs['q'].shape)}")
            check("roberta c shape", outputs['c'].shape == (2, 12), f"expected (2, 12), got {tuple(outputs['c'].shape)}")
            check("roberta neg shape", outputs['neg_c'].shape == (2, 12), f"expected (2, 12), got {tuple(outputs['neg_c'].shape)}")
            check("roberta outputs finite", all(torch.isfinite(v).all().item() for v in outputs.values()))
            scores = torch.mm(outputs['q'], outputs['c'].t())
            neg_scores = (outputs['q'] * outputs['neg_c']).sum(-1).unsqueeze(1)
            logits = torch.cat([scores, neg_scores], dim=-1)
            check("roberta retrieval score shape", logits.shape == (2, 3), f"expected (2, 3), got {tuple(logits.shape)}")
            check("roberta retrieval scores finite", torch.isfinite(logits).all().item())
        else:
            skip_checks(7, "RobertaSingleRetrieverThreeCatPool.forward returned None")
    except Exception as exc:
        skip_checks(8, f"RobertaSingleRetrieverThreeCatPool.forward raised {type(exc).__name__}: {exc}")
    print()

    # ==========================================================
    # Final Score
    # ==========================================================
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
