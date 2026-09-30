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
    if task == "claim_extraction":
        llama_format = "### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:"
        tulu_format = "<|user|>\n### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n<|assistant|>"
    elif task == "context_sampling":
        llama_format = (
            "### Instruction:\n{instruction}\n\n### Input:\nPremise: {neg_premise}\n"
            "Conclusion: {neg_conclusion}\n\n### Response:"
        )
        tulu_format = (
            "<|user|>\n### Instruction:\n{instruction}\n\n### Input:\nPremise: {neg_premise}\n"
            "Conclusion: {neg_conclusion}\n\n### Response:\n<|assistant|>"
        )
    elif task == "revision":
        llama_format = (
            "### Instruction:\n{instruction}\n\n### Input:\nText: {cur_text}\n"
            "Statement: {premise}\n\n### Response:"
        )
        tulu_format = (
            "<|user|>\n### Instruction:\n{instruction}\n\n### Input:\nText: {cur_text}\n"
            "Statement: {premise}\n\n### Response:\n<|assistant|>"
        )
    else:
        raise ValueError(f"unknown task: {task}")

    if "Llama" in model_name:
        return llama_format
    elif "Tulu" in model_name:
        return tulu_format
    return llama_format


def segment_argument_text(argument_text):
    text = argument_text + "\nChoices:\n"
    cur_splitted = []
    red = [j for j in re.split("; |, |\\. |\\? |- ", argument_text) if len(j.strip()) >= 10]
    if len(red) == 1:
        for j in red:
            tmp = re.split(
                " because | so | if | and | but | otherwise | or |  Because | So | If | And | But | Otherwise | Or ",
                " " + j.strip() + " ",
            )
            for k in tmp:
                if len(k.strip()) < 10:
                    continue
                text += str(len(cur_splitted) + 1) + ". " + k.strip() + "\n"
                cur_splitted.append(k.strip())
    else:
        for j in red:
            text += str(len(cur_splitted) + 1) + ". " + j.strip() + "\n"
            cur_splitted.append(j.strip())
    return text.strip(), cur_splitted


def build_claim_extraction_prompts(data, model_name="Llama-2-7b-chat-hf"):
    extraction_prompt_format = select_prompt_format(model_name, "claim_extraction")
    prompts = []
    idxs = []
    splitted_text = []
    for idxi, i in enumerate(data):
        text, cur_splitted = segment_argument_text(i["text"])
        extraction_instance = extraction_prompt_format.format(
            instruction=EXTRACTION_INSTRUCTION, text=text.strip()
        )
        prompts.append(extraction_instance)
        idxs.append(idxi)
        splitted_text.append(cur_splitted)
    return prompts, idxs, splitted_text


def parse_claim_extraction_responses(data, responses, idxs, splitted_text, negator):
    for idxi, i in enumerate(responses):
        i = i.split("Response:")[-1].strip()
        cur_splitted = splitted_text[idxi]
        for j in i.split("\n"):
            if len(j) == 0:
                continue
            if "Conclusion:" in j or (j[0] >= "1" and j[0] <= "9"):
                if "Conclusion:" in j:
                    conclusion_num = j.split("Conclusion:")[1].strip()[0]
                else:
                    conclusion_num = j[0]
                try:
                    conclusion = cur_splitted[int(conclusion_num) - 1]
                except:
                    conclusion = cur_splitted[-1]
                premise = [k for k in cur_splitted if k != conclusion]
                data[idxs[idxi]].update({"premise": premise, "conclusion": conclusion})

                neg_premise = []
                for j in premise:
                    neg_premise.append(negator.negate_sentence(j))
                data[idxs[idxi]].update({"neg_premise": neg_premise})

                neg_conclusion = negator.negate_sentence(conclusion)
                data[idxs[idxi]].update({"neg_conclusion": neg_conclusion})

                break
    return data


def build_context_sampling_prompts(data, model_name="Llama-2-7b-chat-hf"):
    generate_prompt_format = select_prompt_format(model_name, "context_sampling")
    prompts = []
    idxs = []
    for idxi, i in enumerate(data):
        if not "neg_premise" in i:
            continue
        for j in range(len(i["premise"])):
            cur_neg_premise = i["neg_premise"][j].strip()
            cur_context = ". ".join([x for k, x in enumerate(i["premise"]) if k != j]).strip()
            if len(cur_context) > 0:
                generate_context_instance = generate_prompt_format.format(
                    instruction=GENERATE_INSTRUCTION_CONTEXT.format(cur_context=cur_context),
                    neg_premise=cur_neg_premise,
                    neg_conclusion=i["neg_conclusion"],
                )
            else:
                generate_context_instance = generate_prompt_format.format(
                    instruction=GENERATE_INSTRUCTION,
                    neg_premise=cur_neg_premise,
                    neg_conclusion=i["neg_conclusion"],
                )
            prompts.append(generate_context_instance)
            idxs.append([idxi, j])
    return prompts, idxs


def parse_context_sampling_responses(data, responses, idxs):
    for idxi, i in enumerate(responses):
        i = i.split("Response:")[-1].strip()
        response_reformat = []
        cur = ""
        for k in i.split("\n"):
            if len(k) == 0:
                continue
            if len(cur) >= 8 and cur[-1] == ":":
                cur += k
            else:
                response_reformat.append(cur.strip())
                cur = k
        response_reformat.append(cur.strip())

        cur_neg_contexts = []
        for k in response_reformat:
            if "Context" in k and ":" in k:
                cur_neg_contexts.append(k.split(":")[1].strip())
            elif len(k) > 2 and k[0] >= "1" and k[0] <= "9" and k[1] == ".":
                cur_neg_contexts.append(k[2:].strip())

        if not "neg_context" in data[idxs[idxi][0]]:
            data[idxs[idxi][0]]["neg_context"] = [cur_neg_contexts]
        else:
            data[idxs[idxi][0]]["neg_context"].append(cur_neg_contexts)
    return data


def build_revision_prompts(data, model_name="Llama-2-7b-chat-hf"):
    revise_prompt_format = select_prompt_format(model_name, "revision")
    revise_context_prompts = []
    idxs = []
    for idxi, i in enumerate(data):
        if not "neg_context" in i or len(i["neg_context"]) == 0 or len(i["neg_context"][0]) == 0 or len(i["neg_context"][0][0]) == 0:
            continue
        for j in range(len(i["premise"])):
            for idxk, k in enumerate(i["neg_context"][j]):
                cur_premise = i["premise"][j].strip()
                revise_context_instance = revise_prompt_format.format(
                    instruction=REVISE_CONTEXT_INSTRUCTION,
                    premise=cur_premise,
                    cur_text=k,
                )
                revise_context_prompts.append(revise_context_instance)
                idxs.append([idxi, j, idxk])
    return revise_context_prompts, idxs


def parse_revision_responses(data, responses, idxs):
    for idxi, i in enumerate(responses):
        i = i.split("Response:")[-1].strip()
        if not "contexts" in data[idxs[idxi][0]]:
            data[idxs[idxi][0]]["contexts"] = []
        while len(data[idxs[idxi][0]]["contexts"]) <= idxs[idxi][1]:
            data[idxs[idxi][0]]["contexts"].append([])
        while len(data[idxs[idxi][0]]["contexts"][idxs[idxi][1]]) <= idxs[idxi][2]:
            data[idxs[idxi][0]]["contexts"][idxs[idxi][1]].append("")

        for j in i.split("\n"):
            if len(j) > 10 and j.strip()[-1] != ":":
                data[idxs[idxi][0]]["contexts"][idxs[idxi][1]][idxs[idxi][2]] = j.split(": ")[-1].strip()
                break
    return data


def estimate_probability_predictions(data, score_fn, label_mapping=None):
    if label_mapping is None:
        label_mapping = LABEL_MAPPING
    for idxi, i in enumerate(data):
        if not "contexts" in i:
            continue
        pred = [0, 0]  # sufficient, insufficient
        for j in range(len(i["contexts"])):
            if len(i["contexts"][j]) == 0:
                continue

            scores = []
            score_max = []
            for k in range(len(i["contexts"][j])):
                cur_context = i["contexts"][j][k]
                if len(cur_context) > 0 and cur_context[0] == '"' and cur_context[-1] == '"':
                    cur_context = cur_context[1:-1]
                cur_score = score_fn(cur_context, i["conclusion"])
                if not torch.is_tensor(cur_score):
                    cur_score = torch.tensor(cur_score, dtype=torch.float32)
                scores.append(cur_score)
                score_max.append(int(cur_score.argmax(axis=0)))
            scores = torch.stack(scores, dim=0)

            i.update({"entailment": [label_mapping[x] for x in score_max]})
            votes = [0, 0, 0]  # 'contradiction', 'neutral', 'entailment'
            for k in score_max:
                votes[k] += 1
            if votes[0] > votes[2]:
                pred[1] += 1
            elif votes[0] < votes[2]:
                pred[0] += 1
            else:  # compare probability sum
                scores_sum = torch.sum(scores, axis=0)
                if scores_sum[0] > scores_sum[2]:
                    pred[1] += 1
                else:
                    pred[0] += 1

        if pred[1] > 0:
            i.update({"prediction": "insufficient"})
        else:
            i.update({"prediction": "sufficient"})
    return data


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
