"""
ground_truth.py for CASA core reasoning components.

Source-consolidated from:
- code/claim_extraction.py
- code/context_sampling.py
- code/revision_under_intervention.py
- code/probability_estimation.py

Only the CASA prompt-construction, claim parsing, context sampling, intervention
revision, generation helper, and probability-estimation decision logic are
included. Corpus loading, external model loading, file writing, CLI/script
orchestration, and remote model requirements are intentionally excluded.
"""

import re

import torch


try:
    from transformers import StoppingCriteria
except Exception:
    class StoppingCriteria:
        ...


EXTRACTION_INSTRUCTION = (
    "Determine which part of the text is the conclusion.\n"
    "Output the number of the conclusion part first, and give an explanation.\n"
    "Format:\nConclusion: [number]\nExplanation: ..."
)
GENERATE_INSTRUCTION = (
    "Generate 3 detailed contexts. Each context is consistent with both the premise and the conclusion. "
    "Each context is in one line."
)
GENERATE_INSTRUCTION_CONTEXT = (
    'Generate 3 detailed contexts. Each context contains "{cur_context}" Each context is consistent with both '
    "the premise and the conclusion. Each context is in one line."
)
REVISE_CONTEXT_INSTRUCTION = "Revise the text to contain the provided statement."
LABEL_MAPPING = ["contradiction", "neutral", "entailment"]


def select_prompt_format(model_name, task):
    """
    [TODO] Select the CASA prompt template for a model family and task stage.

    Input:
        model_name: str - generation model identifier, such as a Llama or Tulu model name.
        task: str - one of "claim_extraction", "context_sampling", or "revision".

    Output: str - prompt format string with the placeholders expected by that stage.

"""
    pass


def segment_argument_text(argument_text):
    """
    [TODO] Segment an argument into numbered conclusion-candidate clauses.

    Input:
        argument_text: str - original informal argument text.

    Output:
        text: str - original text followed by a "Choices:" section with numbered candidate clauses.
        cur_splitted: list[str] - candidate clauses in the same order as the numbered choices.

"""
    pass


def build_claim_extraction_prompts(data, model_name="Llama-2-7b-chat-hf"):
    """
    [TODO] Build conclusion-extraction prompts for CASA records.

    Input:
        data: list[dict] - records containing a "text" argument field.
        model_name: str - model family used to select the prompt format.

    Output:
        prompts: list[str] - formatted extraction prompts.
        idxs: list[int] - original record indices aligned with prompts.
        splitted_text: list[list[str]] - segmented candidate clauses aligned with prompts.

"""
    pass


def parse_claim_extraction_responses(data, responses, idxs, splitted_text, negator):
    """
    [TODO] Parse LLM conclusion-selection responses and update CASA records.

    Input:
        data: list[dict] - original mutable records.
        responses: list[str] - generated claim-extraction outputs.
        idxs: list[int] - record indices aligned with responses.
        splitted_text: list[list[str]] - candidate clauses aligned with responses.
        negator: object - exposes `negate_sentence(str) -> str`.

    Output: list[dict] - records updated with premise, conclusion, neg_premise, and neg_conclusion.

"""
    pass


def build_context_sampling_prompts(data, model_name="Llama-2-7b-chat-hf"):
    """
    [TODO] Build prompts for sampling counterfactual contexts per premise.

    Input:
        data: list[dict] - records containing premise, neg_premise, and neg_conclusion fields.
        model_name: str - model family used to select prompt format.

    Output:
        prompts: list[str] - context-generation prompts.
        idxs: list[list[int]] - pairs [record_index, premise_index] aligned with prompts.

"""
    pass


def parse_context_sampling_responses(data, responses, idxs):
    """
    [TODO] Parse generated counterfactual contexts into CASA records.

    Input:
        data: list[dict] - mutable records.
        responses: list[str] - raw context-generation responses.
        idxs: list[list[int]] - [record_index, premise_index] alignment metadata.

    Output: list[dict] - records updated with nested `neg_context` lists.

"""
    pass


def build_revision_prompts(data, model_name="Llama-2-7b-chat-hf"):
    """
    [TODO] Build intervention prompts that revise sampled contexts to contain a premise.

    Input:
        data: list[dict] - records containing premise and neg_context fields.
        model_name: str - model family used to select prompt format.

    Output:
        revise_context_prompts: list[str] - revision prompts.
        idxs: list[list[int]] - triples [record_index, premise_index, context_index].

"""
    pass


def parse_revision_responses(data, responses, idxs):
    """
    [TODO] Parse revised intervention contexts into the nested CASA context field.

    Input:
        data: list[dict] - mutable records.
        responses: list[str] - revision-generation responses.
        idxs: list[list[int]] - triples [record_index, premise_index, context_index].

    Output: list[dict] - records updated with `contexts[premise_index][context_index]`.

"""
    pass


def estimate_probability_predictions(data, score_fn, label_mapping=None):
    """
    [TODO] Estimate CASA sufficiency from NLI-style context/conclusion scores.

    Input:
        data: list[dict] - records containing conclusion and nested contexts.
        score_fn: callable(context: str, conclusion: str) -> length-3 score/probability tensor-like object.
        label_mapping: optional list[str] - labels ordered as contradiction, neutral, entailment.

    Output: list[dict] - records updated with entailment labels and final prediction.

"""
    pass


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)


@torch.no_grad()
def generate_completions(model, tokenizer, prompts, batch_size=1, stop_id_sequences=None, **generation_kwargs):
    generations = []

    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        tokenized_prompts = tokenizer(batch_prompts, padding="longest", return_tensors="pt", add_special_tokens=False)
        batch_input_ids = tokenized_prompts.input_ids
        attention_mask = tokenized_prompts.attention_mask

        if model.device.type == "cuda":
            batch_input_ids = batch_input_ids.cuda()
            attention_mask = attention_mask.cuda()

        try:
            batch_outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=attention_mask,
                stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
                **generation_kwargs
            )

            if stop_id_sequences:
                for output_idx in range(batch_outputs.shape[0]):
                    for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                        if any(batch_outputs[output_idx, token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in stop_id_sequences):
                            batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                            break

            batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
            batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
            batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
            batch_generations = [
                output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
            ]
        except Exception as e:
            print("Error when generating completions for batch:")
            print(batch_prompts)
            print("Error message:")
            print(e)
            print("Use empty string as the completion.")
            batch_generations = [""] * len(batch_prompts) * num_return_sequences

        generations += batch_generations

    assert len(generations) == len(prompts) * num_return_sequences, "number of generations should be equal to number of prompts * num_return_sequences"
    return generations


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

    class FakeNegator:
        def negate_sentence(self, sentence):
            return "not " + sentence

    print("=" * 70)
    print("CASA core reasoning component benchmark")
    print("=" * 70)

    print("-" * 60)
    print("[Group 1] Prompt formats and argument segmentation")
    try:
        llama_claim = select_prompt_format("Llama-2-7b-chat-hf", "claim_extraction")
        tulu_context = select_prompt_format("Tulu-2", "context_sampling")
        text, fragments = segment_argument_text("The streets are wet because it rained last night so traffic is slow")
        check("Llama prompt selected", llama_claim.startswith("### Instruction:"))
        check("Tulu prompt selected", tulu_context.startswith("<|user|>"))
        check("segmentation choices label", "Choices:" in text and "1." in text)
        check("segmentation fragment count", len(fragments) >= 2, f"got {len(fragments)}")
        check("segmentation keeps long fragments", all(len(x) >= 10 for x in fragments))
    except Exception as exc:
        skip_checks(5, f"prompt/segmentation raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 2] Claim extraction prompt construction and parsing")
    try:
        data = [{"text": "The policy reduces emissions. Therefore the city should adopt it."}]
        prompts, idxs, splitted = build_claim_extraction_prompts(data, "Llama-2-7b-chat-hf")
        check("claim prompts count", len(prompts) == 1)
        check("claim prompt contains instruction", EXTRACTION_INSTRUCTION.split("\n")[0] in prompts[0])
        check("claim split nonempty", len(splitted[0]) == 2, f"got {splitted[0]}")
        parsed = parse_claim_extraction_responses(data, ["Conclusion: 2\nExplanation: final sentence"], idxs, splitted, FakeNegator())
        check("claim conclusion parsed", parsed[0]["conclusion"] == splitted[0][1])
        check("claim premise parsed", parsed[0]["premise"] == [splitted[0][0]])
        check("claim negations added", parsed[0]["neg_premise"] == ["not " + splitted[0][0]] and parsed[0]["neg_conclusion"] == "not " + splitted[0][1])
    except Exception as exc:
        skip_checks(6, f"claim extraction raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 3] Context sampling")
    try:
        ctx_data = [{
            "premise": ["A is true", "B follows"],
            "neg_premise": ["A is false", "B does not follow"],
            "neg_conclusion": "C is false",
        }]
        ctx_prompts, ctx_idxs = build_context_sampling_prompts(ctx_data, "Llama-2-7b-chat-hf")
        check("context prompt count", len(ctx_prompts) == 2)
        check("context idx shape", ctx_idxs == [[0, 0], [0, 1]])
        check("context prompt includes other premise", "B follows" in ctx_prompts[0])
        responses = ["Context 1: first world\nContext 2: second world", "1. third world\n2. fourth world"]
        sampled = parse_context_sampling_responses(ctx_data, responses, ctx_idxs)
        check("neg_context field added", "neg_context" in sampled[0])
        check("neg_context outer length", len(sampled[0]["neg_context"]) == 2)
        check("neg_context parsed values", sampled[0]["neg_context"][0] == ["first world", "second world"])
    except Exception as exc:
        skip_checks(6, f"context sampling raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 4] Revision under intervention")
    try:
        rev_data = [{
            "premise": ["A is true", "B follows"],
            "neg_context": [["context one", "context two"], ["context three"]],
        }]
        rev_prompts, rev_idxs = build_revision_prompts(rev_data, "Llama-2-7b-chat-hf")
        check("revision prompt count", len(rev_prompts) == 3)
        check("revision idxs", rev_idxs == [[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        check("revision prompt contains text", "Text: context one" in rev_prompts[0])
        revised = parse_revision_responses(rev_data, ["Response: Revised: A is true in context one", "Context: A is true in context two", "B follows in context three"], rev_idxs)
        check("contexts field added", "contexts" in revised[0])
        check("contexts nested sizes", len(revised[0]["contexts"]) == 2 and len(revised[0]["contexts"][0]) == 2)
        check("revision parsed content", revised[0]["contexts"][0][0] == "A is true in context one")
    except Exception as exc:
        skip_checks(6, f"revision raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 5] Probability estimation")
    try:
        prob_data = [{
            "conclusion": "The plan works",
            "contexts": [["context supports", "context also supports"], ["context contradicts"]],
        }]

        def score_fn(context, conclusion):
            if "contradicts" in context:
                return torch.tensor([0.8, 0.1, 0.1])
            return torch.tensor([0.1, 0.2, 0.7])

        estimated = estimate_probability_predictions(prob_data, score_fn)
        check("probability prediction added", estimated[0]["prediction"] == "insufficient")
        check("entailment labels added", estimated[0]["entailment"] == ["contradiction"])
        tie_data = [{"conclusion": "C", "contexts": [["a", "b"]]}]

        def tie_score_fn(context, conclusion):
            return torch.tensor([0.4, 0.1, 0.4]) if context == "a" else torch.tensor([0.2, 0.1, 0.6])

        tie_estimated = estimate_probability_predictions(tie_data, tie_score_fn)
        check("probability tie breaks by score sum", tie_estimated[0]["prediction"] == "sufficient")
        check("quoted context stripped", estimate_probability_predictions([{"conclusion": "C", "contexts": [['"quoted"']]}], lambda c, y: torch.tensor([0.1, 0.2, 0.7]))[0]["prediction"] == "sufficient")
        check("missing contexts skipped", estimate_probability_predictions([{"text": "x"}], score_fn)[0] == {"text": "x"})
    except Exception as exc:
        skip_checks(5, f"probability estimation raised {type(exc).__name__}: {exc}")
    print()

    print("-" * 60)
    print("[Group 6] Stop criteria")
    try:
        criteria = KeyWordsCriteria([[5, 6], [9]])
        ids_stop = torch.tensor([[1, 5, 6], [3, 9, 9]])
        ids_not_stop = torch.tensor([[1, 5, 6], [3, 4, 8]])
        check("stop criteria all stopped", criteria(ids_stop, torch.empty(0)) is True)
        check("stop criteria not all stopped", criteria(ids_not_stop, torch.empty(0)) is False)
        check("stop sequences preserved", criteria.stop_sequences == [[5, 6], [9]])
    except Exception as exc:
        skip_checks(3, f"stop criteria raised {type(exc).__name__}: {exc}")
    print()

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed == 0:
        print("All tests PASSED! The reasoning code is complete and correct.")
    else:
        print(f"{failed} check(s) FAILED - some target functions may be incomplete.")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
