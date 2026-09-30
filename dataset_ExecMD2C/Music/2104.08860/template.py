"""Ground-truth core retrieval components for the CLIP4Clip benchmark.

This file consolidates the model-side similarity computation used by CLIP4Clip:
masked pooling, loose retrieval similarity, tight cross-transformer similarity,
and similarity-head routing. Corpus handling, fitting loops, update-rule code,
score-reporting utilities, saved artifact I/O, and CLIP weight acquisition logic
are intentionally excluded.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


# --- [Original file: modules/until_module.py] ---

class CrossEn(nn.Module):
    def __init__(self,):
        super(CrossEn, self).__init__()

    def forward(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss


class AllGather(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None,
        )


allgather = AllGather.apply


# --- [Original file: modules/modeling.py] ---

class CLIP4Clip(nn.Module):
    def get_sequence_output(self, input_ids, token_type_ids, attention_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])

        bs_pair = input_ids.size(0)
        sequence_hidden = self.clip.encode_text(input_ids).float()
        sequence_hidden = sequence_hidden.view(bs_pair, -1, sequence_hidden.size(-1))

        return sequence_hidden

    def get_visual_output(self, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        bs_pair = video_mask.size(0)
        visual_hidden = self.clip.encode_image(video, video_frame=video_frame).float()
        visual_hidden = visual_hidden.view(bs_pair, -1, visual_hidden.size(-1))

        return visual_hidden

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False, video_frame=-1):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            video = torch.as_tensor(video).float()
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.view(b * pair * bs * ts, channel, h, w)
            video_frame = bs * ts

        sequence_output = self.get_sequence_output(input_ids, token_type_ids, attention_mask, shaped=True)
        visual_output = self.get_visual_output(video, video_mask, shaped=True, video_frame=video_frame)

        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        """[TODO] Build and run the tight cross-transformer input for one text-video batch.

        Input:
            sequence_output: (batch, text_tokens, hidden) - text embeddings.
            visual_output: (batch, video_frames, hidden) - frame embeddings.
            attention_mask: (batch, text_tokens) - valid text-token mask.
            video_mask: (batch, video_frames) - valid video-frame mask.

        Output:
            cross_output: (batch, text_tokens + video_frames, hidden) - final
                cross-transformer hidden states.
            pooled_output: (batch, hidden) - pooled cross representation.
            concat_mask: (batch, text_tokens + video_frames) - concatenated
                validity mask.

"""
        pass

    def _mean_pooling_for_similarity_sequence(self, sequence_output, attention_mask):
        """[TODO] Mean-pool text token features for retrieval similarity.

        Input:
            sequence_output: (batch, text_tokens, hidden) - per-token text
                features.
            attention_mask: (batch, text_tokens) - valid-token mask.

        Output:
            (batch, hidden) - masked average text representation.

"""
        pass

    def _mean_pooling_for_similarity_visual(self, visual_output, video_mask,):
        """[TODO] Mean-pool video frame features for retrieval similarity.

        Input:
            visual_output: (batch, video_frames, hidden) - per-frame visual
                features.
            video_mask: (batch, video_frames) - valid-frame mask.

        Output:
            (batch, hidden) - masked average video representation.

"""
        pass

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        text_out = self._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        video_out = self._mean_pooling_for_similarity_visual(visual_output, video_mask)

        return text_out, video_out

    def _loose_similarity(self, sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP"):
        """[TODO] Compute loose-type text-video retrieval logits.

        Input:
            sequence_output: (text_batch, 1, hidden) - CLIP text features.
            visual_output: (video_batch, video_frames, hidden) - CLIP frame
                features.
            attention_mask: (text_batch, text_tokens) - text mask, preserved for
                interface consistency.
            video_mask: (video_batch, video_frames) - valid-frame mask.
            sim_header: string - one of "meanP", "seqLSTM", or "seqTransf".

        Output:
            (text_batch, video_batch) - scaled retrieval similarity matrix.

"""
        pass

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        """[TODO] Compute tight cross-transformer retrieval logits for all text-video pairs.

        Input:
            sequence_output: (text_batch, text_tokens, hidden) - text features.
            visual_output: (video_batch, video_frames, hidden) - video features.
            attention_mask: (text_batch, text_tokens) - text validity mask.
            video_mask: (video_batch, video_frames) - video validity mask.

        Output:
            (text_batch, video_batch) - pairwise tight-transformer similarity
            logits.

"""
        pass

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, loose_type=False):
        """[TODO] Dispatch to the correct CLIP4Clip similarity head.

        Input:
            sequence_output: text features, already shaped or flattenable.
            visual_output: video features.
            attention_mask: text mask, shaped like the text batch.
            video_mask: frame mask, shaped like the video batch.
            shaped: bool - whether masks are already flattened to model batch
                shape.
            loose_type: bool - choose loose similarity or tight cross similarity.

        Output:
            retrieve_logits: pairwise text-video similarity matrix.
            contrastive_direction: empty tuple placeholder preserved from the
                original interface.

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

    class DummyClip:
        def __init__(self):
            self.logit_scale = torch.tensor(0.0)

    class DummyTaskConfig:
        world_size = 1
        rank = 0

    class FakeCross:
        def __call__(self, concat_features, concat_type, concat_mask, output_all_encoded_layers=True):
            type_bias = concat_type.to(dtype=concat_features.dtype).unsqueeze(-1) * 0.25
            masked = (concat_features + type_bias) * concat_mask.to(dtype=concat_features.dtype).unsqueeze(-1)
            pooled = masked[:, 0]
            return [masked], pooled

    def make_model():
        model = object.__new__(CLIP4Clip)
        nn.Module.__init__(model)
        model.clip = DummyClip()
        model.task_config = DummyTaskConfig()
        model.sim_header = "meanP"
        model.training = False
        return model

    print("CLIP4Clip benchmark checks")

    print("[Test 1/6] _mean_pooling_for_similarity_sequence")
    try:
        model = make_model()
        sequence_output = torch.tensor([
            [[100.0, 100.0], [1.0, 3.0], [3.0, 5.0]],
            [[50.0, 50.0], [2.0, 4.0], [9.0, 9.0]],
        ])
        attention_mask = torch.tensor([[1, 1, 1], [1, 1, 0]])
        pooled = model._mean_pooling_for_similarity_sequence(sequence_output, attention_mask)
        check("sequence pooling output not None", pooled is not None)
        if pooled is not None:
            check("sequence pooling shape", tuple(pooled.shape) == (2, 2), str(tuple(pooled.shape)))
            check("sequence pooling finite", torch.isfinite(pooled).all().item())
            check("sequence pooling excludes first token",
                  torch.allclose(pooled, torch.tensor([[2.0, 4.0], [2.0, 4.0]])), str(pooled))
        else:
            skip_checks(3, "sequence pooling returned None")
    except Exception as exc:
        skip_checks(4, f"_mean_pooling_for_similarity_sequence raised {type(exc).__name__}: {exc}")

    print("[Test 2/6] _mean_pooling_for_similarity_visual")
    try:
        model = make_model()
        visual_output = torch.tensor([
            [[1.0, 1.0], [3.0, 5.0], [9.0, 9.0]],
            [[7.0, 7.0], [8.0, 8.0], [9.0, 9.0]],
        ])
        video_mask = torch.tensor([[1, 1, 0], [0, 0, 0]])
        pooled = model._mean_pooling_for_similarity_visual(visual_output, video_mask)
        check("visual pooling output not None", pooled is not None)
        if pooled is not None:
            check("visual pooling shape", tuple(pooled.shape) == (2, 2), str(tuple(pooled.shape)))
            check("visual pooling finite", torch.isfinite(pooled).all().item())
            check("visual pooling handles zero-mask sample",
                  torch.allclose(pooled, torch.tensor([[2.0, 3.0], [0.0, 0.0]])), str(pooled))
        else:
            skip_checks(3, "visual pooling returned None")
    except Exception as exc:
        skip_checks(4, f"_mean_pooling_for_similarity_visual raised {type(exc).__name__}: {exc}")

    print("[Test 3/6] _get_cross_output")
    try:
        model = make_model()
        model.cross = FakeCross()
        sequence_output = torch.ones(2, 1, 4)
        visual_output = torch.ones(2, 3, 4) * 2
        attention_mask = torch.tensor([[1], [1]])
        video_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        cross_output, pooled_output, concat_mask = model._get_cross_output(
            sequence_output, visual_output, attention_mask, video_mask)
        check("cross output tuple not None", cross_output is not None and pooled_output is not None and concat_mask is not None)
        if cross_output is not None and pooled_output is not None and concat_mask is not None:
            check("cross output shape", tuple(cross_output.shape) == (2, 4, 4), str(tuple(cross_output.shape)))
            check("pooled output shape", tuple(pooled_output.shape) == (2, 4), str(tuple(pooled_output.shape)))
            check("concat mask preserves text/video masks",
                  torch.equal(concat_mask, torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])), str(concat_mask))
        else:
            skip_checks(3, "cross output returned None")
    except Exception as exc:
        skip_checks(4, f"_get_cross_output raised {type(exc).__name__}: {exc}")

    print("[Test 4/6] _loose_similarity")
    try:
        model = make_model()
        sequence_output = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        visual_output = torch.tensor([
            [[2.0, 0.0], [2.0, 0.0]],
            [[0.0, 3.0], [0.0, 3.0]],
        ])
        attention_mask = torch.ones(2, 1, dtype=torch.long)
        video_mask = torch.ones(2, 2, dtype=torch.long)
        logits = model._loose_similarity(sequence_output, visual_output, attention_mask, video_mask, sim_header="meanP")
        check("loose similarity output not None", logits is not None)
        if logits is not None:
            check("loose similarity shape", tuple(logits.shape) == (2, 2), str(tuple(logits.shape)))
            check("loose similarity finite", torch.isfinite(logits).all().item())
            check("loose similarity favors aligned pairs",
                  logits[0, 0] > logits[0, 1] and logits[1, 1] > logits[1, 0], str(logits))
        else:
            skip_checks(3, "loose similarity returned None")
    except Exception as exc:
        skip_checks(4, f"_loose_similarity raised {type(exc).__name__}: {exc}")

    print("[Test 5/6] _cross_similarity")
    try:
        model = make_model()
        model.cross = FakeCross()
        model.similarity_dense = nn.Linear(4, 1, bias=False)
        with torch.no_grad():
            model.similarity_dense.weight.fill_(1.0)
        sequence_output = torch.tensor([[[1.0, 0.0, 0.0, 0.0]], [[0.0, 1.0, 0.0, 0.0]]])
        visual_output = torch.randn(3, 2, 4)
        attention_mask = torch.ones(2, 1, dtype=torch.long)
        video_mask = torch.ones(3, 2, dtype=torch.long)
        logits = model._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        check("cross similarity output not None", logits is not None)
        if logits is not None:
            check("cross similarity shape", tuple(logits.shape) == (2, 3), str(tuple(logits.shape)))
            check("cross similarity finite", torch.isfinite(logits).all().item())
            check("cross similarity scores every text-video pair", logits.numel() == 6, str(logits.numel()))
        else:
            skip_checks(3, "cross similarity returned None")
    except Exception as exc:
        skip_checks(4, f"_cross_similarity raised {type(exc).__name__}: {exc}")

    print("[Test 6/6] get_similarity_logits")
    try:
        model = make_model()
        model.sim_header = "meanP"
        sequence_output = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        visual_output = torch.tensor([
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ])
        attention_mask = torch.ones(2, 1, dtype=torch.long)
        video_mask = torch.ones(2, 2, dtype=torch.long)
        logits, contrastive_direction = model.get_similarity_logits(
            sequence_output, visual_output, attention_mask, video_mask, shaped=True, loose_type=True)
        check("similarity logits output not None", logits is not None)
        if logits is not None:
            check("similarity logits shape", tuple(logits.shape) == (2, 2), str(tuple(logits.shape)))
            check("similarity logits finite", torch.isfinite(logits).all().item())
            check("similarity logits returns empty contrastive direction", contrastive_direction == (), str(contrastive_direction))
        else:
            skip_checks(3, "similarity logits returned None")
    except Exception as exc:
        skip_checks(4, f"get_similarity_logits raised {type(exc).__name__}: {exc}")

    print(f"RESULT: passed={passed} failed={failed}")
    if failed != 0:
        raise SystemExit(1)
