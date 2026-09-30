"""
Self-contained symbolic-instruction-tuning benchmark answer key.

This repository does not define a custom neural architecture: training delegates
to HuggingFace seq2seq models. The project-owned core logic is the symbolic
instruction preprocessing stack that converts structured table inputs into
linearized, truncated, Jinja-rendered seq2seq training examples. This benchmark
therefore extracts those components and excludes training loops, remote model
loading, evaluation scripts, CLI wrappers, and baseline API callers.
"""

import abc
import logging
import random
from abc import ABC
from typing import Dict, List

from jinja2 import Template
from transformers import AutoTokenizer, BasicTokenizer


OPENAI_API_KEY = "<<YOUR_OPENAI_API_KEY>>"
MAX_LENGTH = 2048
MAX_TOKENS = 128
DEFAULT_TEMPLATE = "Read the following table and answer the question.\n{{table}}\nQuestion: {{question}}\nAnswer:"
DEL = "|"

logger = logging.getLogger(__name__)
random.seed(42)


# --- [Original file: preprocessor/default.py] ---

def preprocess_function_with_template(examples, tokenizer, template, lowercase, **kwargs):
    """
    The is_training FLAG is used to identify if we could use the supervision
    to truncate the table content if it is required.
    """
    assert "input_fields" in examples
    input_fields = examples["input_fields"][0]
    inputs = []

    for idx in range(len(examples["input_fields"])):
        render_dict = {field: examples[field][idx] for field in input_fields}
        jinja_template = Template(template)
        render_input = jinja_template.render(**render_dict)
        assert "{{ " not in render_input, "Template not rendered properly!"
        inputs.append(render_input)

    if lowercase:
        inputs = [example.lower() for example in inputs]

    model_inputs = tokenizer(
        text=inputs,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        text=examples["output"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def preprocess_function(examples, tokenizer, lowercase, **kwargs):
    """
    The is_training FLAG is used to identify if we could use the supervision
    to truncate the table content if it is required.
    """
    if lowercase:
        examples["input"] = [example.lower() for example in examples["input"]]

    model_inputs = tokenizer(
        text=examples["input"],
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    labels = tokenizer(
        text=examples["output"],
        max_length=256,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# --- [Original file: preprocessor/table_utils/table_linearzie.py] ---

class TableLinearize(abc.ABC):
    PROMPT_MESSAGE = """
        Please check that your table must follow the following format:
        {"header": ["col1", "col2", "col3"], "rows": [["row11", "row12", "row13"], ["row21", "row22", "row23"]]}
    """

    def process_table(self, table_content: Dict) -> str:
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        raise NotImplementedError

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        raise NotImplementedError

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        raise NotImplementedError


class IndexedRowTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + "\n"
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "header: " + " | ".join(headers)

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "row " + str(row_index) + " : " + row_str


class MarkdownTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + "\n"
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "| " + " | ".join(headers) + " |\n" + "".join(["|---" for _ in range(len(headers))]) + "|\n"

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += " | ".join(row_cell_values)
        return "| " + row_str + " |\n"


class NaturalTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        # process header
        _table_str = self.process_header(table_content["header"]) + " "
        # process rows
        _table_str += "The table includes {} rows. ".format(len(table_content["rows"]))
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + " "
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return "The table includes {} columns: {}.".format(len(headers), ", ".join(headers))

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value)
        row_str += ", ".join(row_cell_values)
        return "The {}th row is: {}.".format(row_index, row_str)


class CodexTableLinearize(TableLinearize):
    """
    FORMAT: col: col1 | col2 | col3 row 1 : val1 | val2 | val3 row 2 : ...
    """

    def process_table(self, table_content: Dict):
        """
        Given a table, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        assert "header" in table_content and "rows" in table_content, self.PROMPT_MESSAGE
        _table_str = self.process_header(table_content["header"]) + "\n"
        # process rows
        for i, row_example in enumerate(table_content["rows"]):
            # NOTE: the row should start from row 1 instead of 0
            _table_str += self.process_row(row_example, row_index=i + 1) + "\n"
        return _table_str.strip()

    def process_header(self, headers: List):
        """
        Given a list of headers, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        return " | ".join([header.replace("\\n", "-").replace("\n", "-") for header in headers])

    def process_row(self, row: List, row_index: int):
        """
        Given a row, TableLinearize aims at converting it into a flatten sequence with special symbols.
        """
        row_str = ""
        row_cell_values = []
        for cell_value in row:
            if isinstance(cell_value, int) or isinstance(cell_value, float):
                row_cell_values.append(str(cell_value))
            else:
                row_cell_values.append(cell_value.replace("\\n", " ").replace("\n", " "))
        row_str += " | ".join(row_cell_values)
        return row_str


# --- [Original file: preprocessor/table_utils/table_truncate.py] ---

class TableTruncate(ABC):

    def __init__(self, tokenizer: BasicTokenizer = None, max_input_length: int = 1024):
        """
        The class `TableTruncate` is used to compress a table to fit in memory.
        :param tokenizer: a huggingface transformer's tokenizer, to be used on BPE encoding to estimate expected tokens
        :param max_input_length: the maximum length of `question` and `table`, i.e., the max position id of a model
        """
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="facebook/bart-large")
        else:
            self.tokenizer = tokenizer
        self.max_length = max_input_length

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        """
        Given a table, return a truncated table with the same format.
        We enable optionally providing question and answer for precise truncating.
        :return: no return value, but may modify table_content and answer
        """
        raise NotImplementedError


class CellLimitTruncate(TableTruncate):
    """
    Limit the maximum length of cell values in a table to truncate the overall length
    """

    def __init__(self, max_cell_length: int = 15, **kwargs):
        super().__init__(**kwargs)
        self.max_cell_length = max_cell_length

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        cell_mapping = {}
        for row in table_content["rows"]:
            for i, cell in enumerate(row):
                truncate_cell = self.truncate_cell(cell)
                if truncate_cell is not None:
                    cell_mapping[cell] = truncate_cell
                    row[i] = truncate_cell

        # modify the answer list
        for i, case in enumerate(answer):
            if case in cell_mapping.keys():
                answer[i] = cell_mapping[case]

    def truncate_cell(self, cell_value):
        # do not process on these cases
        if isinstance(cell_value, int) or isinstance(cell_value, float):
            return cell_value
        if cell_value.strip() != "":
            try_tokens = self.tokenizer.tokenize(cell_value)
            if len(try_tokens) >= self.max_cell_length:
                retain_tokens = try_tokens[:self.max_cell_length]
                retain_cell_value = self.tokenizer.convert_tokens_to_string(retain_tokens)
                return retain_cell_value
            else:
                return None
        else:
            return cell_value


class RowDeleteTruncate(TableTruncate):
    """
    The row deleting principle is straightforward: randomly deleting rows to fit the table into memory,
    but do not make it too small (e.g., just lower than the limitation is ok).
    """

    def __init__(self, table_linearize: TableLinearize, **kwargs):
        super().__init__(**kwargs)
        self.table_linearize = table_linearize

    def truncate_table(self, table_content: Dict, question: str, answer: List):
        """
        :param table_content: {"header": xxx, "rows": xxx, "id" (Optionally): xxx}
        :param question: natural language sentence
        :param answer: if for training, is the supervision; otherwise will be empty
        """
        delete_ratio, remain_token_len = self.estimate_delete_ratio(table_content, question)
        # randomly delete unrelated rows
        self.delete_unrealted_rows(table_content, question, answer, delete_ratio)
        # guarantee the result < self.max_length
        maximum_keep_rows = 0
        for ind, row_example in enumerate(table_content["rows"]):
            value_string = self.table_linearize.process_row(row_example, ind + 1)
            value_token_len = len(self.tokenizer.tokenize(value_string))
            # over the size limit, and take action
            if value_token_len > remain_token_len:
                break
            remain_token_len -= value_token_len
            maximum_keep_rows += 1
        del table_content["rows"][maximum_keep_rows:]

    def estimate_delete_ratio(self, table_content: Dict, question: str):
        assert "header" in table_content and "rows" in table_content
        number_of_rows = len(table_content["rows"])
        # calculate the tokens of header, special tokens will only be pre-prepended into question
        question_tokens = self.tokenizer.tokenize(question, add_special_tokens=True)
        # calculate the tokens of header
        header_string = self.table_linearize.process_header(table_content["header"])
        header_tokens = self.tokenizer.tokenize(header_string, add_special_tokens=False)
        # split all cell values into tokens and see how many can be accommodated
        used_token_len = len(question_tokens) + len(header_tokens)
        # remaining token space for rows
        remain_token_len = self.max_length - used_token_len

        value_string = ""
        for _, row_example in enumerate(table_content["rows"]):
            # use a general index to roughly estimate the overall token len
            value_string += self.table_linearize.process_row(row_example, 100) + " "
        value_token_len = len(self.tokenizer.tokenize(value_string))

        if value_token_len < remain_token_len:
            # no row will be deleted
            return 0.0, remain_token_len
        else:
            # calc a roughly delete rate
            return 1.0 - remain_token_len / value_token_len, remain_token_len

    def delete_unrealted_rows(self, table_content: Dict, question: str, answer: List, delete_ratio: float):
        """
        The argument answer is used only during training.
        """
        truncated_unrelated_indices = []
        related_indices = []
        if len(answer) == 0:
            answer_set = set([])
        else:
            answer_set = set([ans_ex.lower() for ans_ex in answer])
        # add question key words into answer set
        if question is not None:
            answer_set.update(question.split())
        question_set = set(question.strip("?!.,").split(" "))
        row_max_len = len(table_content["rows"])
        for _row_idx, row in enumerate(table_content["rows"]):
            lower_row = set([str(cell).lower() for cell in row])
            if len(lower_row & answer_set) == 0 and len(lower_row & question_set) == 0:
                truncated_unrelated_indices.append(_row_idx)
            else:
                # add neighbours to preserve information aggressively
                related_indices.extend([_row_idx - 2, _row_idx - 1,
                                        _row_idx,
                                        _row_idx + 1, _row_idx + 2])

        # remove the neighbours
        truncated_unrelated_indices = [_row_idx for _row_idx in truncated_unrelated_indices
                                       if _row_idx not in related_indices]
        # select some cases to drop
        drop_items = min(len(truncated_unrelated_indices), int(len(table_content["rows"]) * delete_ratio))
        drop_row_indices = random.choices(truncated_unrelated_indices, k=drop_items)

        for _row_idx in reversed(range(row_max_len)):
            if _row_idx in drop_row_indices:
                del table_content["rows"][_row_idx]

        # only when the drop ratio is too large, logging for warning.
        if "id" in table_content and len(drop_row_indices) > 0:
            logger.warning("Delete {:.2f} rows in table {}".format(len(drop_row_indices), table_content["id"]))


# --- [Original file: preprocessor/table_utils/table_process.py] ---

class TableProcessor(object):

    def __init__(self, table_linearize_func: TableLinearize,
                 table_truncate_funcs: List[TableTruncate],
                 target_delimiter: str = DEL):
        self.table_linearize_func = table_linearize_func
        self.table_truncate_funcs = table_truncate_funcs
        self.target_delimiter = target_delimiter

    def process_input(self, table: Dict, question: str, template: str=None,
                      **kwargs) -> str:
        """
        Preprocess a sentence into the expected format for model translate.
        """
        if "{table}" in template:
            raise ValueError("You should not use {table} in template since you are using Jinja2 template rendering")

        if template is None:
            template = DEFAULT_TEMPLATE
            print("You do not specify the template, so we use the default template: {}".format(template))

        # modify a table internally
        for truncate_func in self.table_truncate_funcs:
            # use template to truncate table, especially for few-shot examples
            truncate_func.truncate_table(table, question + template, [])
        # linearize a table into a string
        linear_table = self.table_linearize_func.process_table(table)

        # use Jinja2 to render the template
        template = Template(template)
        joint_input = template.render(table=linear_table,
                                      question=question,
                                      **kwargs)
        return joint_input

    def process_output(self, answer: List[str]) -> str:
        """
        Flatten the output for translation
        """
        output = self.target_delimiter.join(answer)
        if output.strip() == "":
            return "@NULL@"
        else:
            return output


class _DummyTokenizer:
    def __init__(self):
        self.calls = []
        self.pad_token_id = 0

    def tokenize(self, text, add_special_tokens=False):
        tokens = str(text).replace("\n", " ").split()
        if add_special_tokens:
            return ["<s>"] + tokens + ["</s>"]
        return tokens

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    def __call__(self, text, max_length, padding, truncation, return_tensors):
        import torch

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)
        self.calls.append({
            "text": texts,
            "max_length": max_length,
            "padding": padding,
            "truncation": truncation,
            "return_tensors": return_tensors,
        })
        encoded = []
        for item in texts:
            ids = [min(len(token), 97) for token in self.tokenize(item)]
            if truncation:
                ids = ids[:max_length]
            if padding == "max_length":
                ids = ids + [self.pad_token_id] * max(0, max_length - len(ids))
            encoded.append(ids)
        return {"input_ids": torch.tensor(encoded, dtype=torch.long)}


class _RecorderTruncate:
    def __init__(self):
        self.calls = []

    def truncate_table(self, table_content, question, answer):
        self.calls.append((question, list(answer)))
        if table_content["rows"]:
            table_content["rows"] = table_content["rows"][:1]


# ============================================================
# __main__: Automated test suite for 7 ablated functions
# ============================================================

if __name__ == "__main__":
    import copy
    import torch

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

    def make_table():
        return {
            "header": ["Name", "Score"],
            "rows": [
                ["Alice", "10"],
                ["Bob", "20"],
                ["Carol", "30"],
                ["Delta", "40"],
                ["Echo", "50"],
                ["Foxtrot", "60"],
            ],
        }

    print("=" * 70)
    print("Symbolic instruction tuning benchmark: preprocessing core")
    print("=" * 70)

    # ==============================================================
    # Test 1/7: preprocess_function_with_template
    # ==============================================================
    print("-" * 60)
    print("[Test 1/7] preprocess_function_with_template")
    try:
        tokenizer = _DummyTokenizer()
        examples = {
            "input_fields": [["question", "table"], ["question", "table"]],
            "question": ["Who Won?", "Highest Score?"],
            "table": ["A | B", "C | D"],
            "output": ["Alice", "Carol"],
        }
        output = preprocess_function_with_template(
            examples,
            tokenizer,
            template="Q: {{ question }} T: {{ table }}",
            lowercase=True,
        )
        check("templated preprocess output not None", output is not None)
        if output is None:
            skip_checks(4, "templated preprocess output missing")
        else:
            check("templated preprocess has labels", "labels" in output)
            check("templated preprocess input shape", tuple(output["input_ids"].shape) == (2, MAX_LENGTH))
            check("templated preprocess label shape", tuple(output["labels"].shape) == (2, 256))
            check("templated preprocess lowercases rendered inputs", tokenizer.calls[0]["text"][0].startswith("q: who won?"))
    except Exception as exc:
        skip_checks(5, f"preprocess_function_with_template raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 2/7: preprocess_function
    # ==============================================================
    print("-" * 60)
    print("[Test 2/7] preprocess_function")
    try:
        tokenizer = _DummyTokenizer()
        examples = {
            "input": ["MIXED Case Input", "Second ITEM"],
            "output": ["Target A", "Target B"],
        }
        output = preprocess_function(examples, tokenizer, lowercase=True)
        check("plain preprocess output not None", output is not None)
        if output is None:
            skip_checks(4, "plain preprocess output missing")
        else:
            check("plain preprocess has labels", "labels" in output)
            check("plain preprocess input shape", tuple(output["input_ids"].shape) == (2, MAX_LENGTH))
            check("plain preprocess label shape", tuple(output["labels"].shape) == (2, 256))
            check("plain preprocess mutates lowercase inputs", examples["input"][0] == "mixed case input")
    except Exception as exc:
        skip_checks(5, f"preprocess_function raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 3/7: RowDeleteTruncate.estimate_delete_ratio
    # ==============================================================
    print("-" * 60)
    print("[Test 3/7] RowDeleteTruncate.estimate_delete_ratio")
    try:
        truncater = RowDeleteTruncate(
            table_linearize=IndexedRowTableLinearize(),
            tokenizer=_DummyTokenizer(),
            max_input_length=18,
        )
        ratio, remain = truncater.estimate_delete_ratio(make_table(), "Who has score 10?")
        check("estimate output not None", ratio is not None and remain is not None)
        check("estimate ratio finite", isinstance(ratio, float) and ratio == ratio)
        check("estimate remain exact", remain == 18 - len(truncater.tokenizer.tokenize("Who has score 10?", add_special_tokens=True)) - len(truncater.tokenizer.tokenize("header: Name | Score", add_special_tokens=False)))
        check("estimate requests deletion when over budget", ratio > 0.0)
    except Exception as exc:
        skip_checks(4, f"RowDeleteTruncate.estimate_delete_ratio raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 4/7: RowDeleteTruncate.delete_unrealted_rows
    # ==============================================================
    print("-" * 60)
    print("[Test 4/7] RowDeleteTruncate.delete_unrealted_rows")
    try:
        random.seed(42)
        table = make_table()
        truncater = RowDeleteTruncate(
            table_linearize=IndexedRowTableLinearize(),
            tokenizer=_DummyTokenizer(),
            max_input_length=128,
        )
        before_len = len(table["rows"])
        truncater.delete_unrealted_rows(table, "Alice", ["Alice"], delete_ratio=1.0)
        flattened = [cell for row in table["rows"] for cell in row]
        check("delete unrelated output not None", table is not None)
        check("delete unrelated reduces or keeps rows", len(table["rows"]) <= before_len)
        check("delete unrelated preserves answer row", "Alice" in flattened)
        check("delete unrelated preserves close neighbors", "Bob" in flattened and "Carol" in flattened)
    except Exception as exc:
        skip_checks(4, f"RowDeleteTruncate.delete_unrealted_rows raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 5/7: RowDeleteTruncate.truncate_table
    # ==============================================================
    print("-" * 60)
    print("[Test 5/7] RowDeleteTruncate.truncate_table")
    try:
        random.seed(42)
        table = make_table()
        truncater = RowDeleteTruncate(
            table_linearize=IndexedRowTableLinearize(),
            tokenizer=_DummyTokenizer(),
            max_input_length=20,
        )
        truncater.truncate_table(table, "Alice", ["Alice"])
        linearized = IndexedRowTableLinearize().process_table(table)
        total_tokens = len(truncater.tokenizer.tokenize("Alice", add_special_tokens=True)) + len(truncater.tokenizer.tokenize(linearized))
        check("truncate table output not None", table is not None)
        check("truncate table keeps schema", "header" in table and "rows" in table)
        check("truncate table reduces row count", len(table["rows"]) < 6)
        check("truncate table fits token budget", total_tokens <= 20)
    except Exception as exc:
        skip_checks(4, f"RowDeleteTruncate.truncate_table raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 6/7: TableProcessor.process_input
    # ==============================================================
    print("-" * 60)
    print("[Test 6/7] TableProcessor.process_input")
    try:
        table = make_table()
        recorder = _RecorderTruncate()
        processor = TableProcessor(
            table_linearize_func=IndexedRowTableLinearize(),
            table_truncate_funcs=[recorder],
        )
        rendered = processor.process_input(
            table,
            "Who scored 10?",
            template="Instruction: {{ question }}\nTable:\n{{ table }}\nExtra: {{ hint }}",
            hint="use rows",
        )
        check("process input output not None", rendered is not None)
        check("process input invoked truncater", len(recorder.calls) == 1)
        check("process input renders question", "Who scored 10?" in rendered)
        check("process input renders linearized table", "row 1 : Alice | 10" in rendered and "row 2" not in rendered)
        check("process input renders extra kwargs", "Extra: use rows" in rendered)
    except Exception as exc:
        skip_checks(5, f"TableProcessor.process_input raised {type(exc).__name__}: {exc}")

    # ==============================================================
    # Test 7/7: TableProcessor.process_output
    # ==============================================================
    print("-" * 60)
    print("[Test 7/7] TableProcessor.process_output")
    try:
        processor = TableProcessor(
            table_linearize_func=IndexedRowTableLinearize(),
            table_truncate_funcs=[],
            target_delimiter=" || ",
        )
        joined = processor.process_output(["Alice", "Bob"])
        null_value = processor.process_output([""])
        check("process output not None", joined is not None)
        check("process output joins answers", joined == "Alice || Bob")
        check("process output empty sentinel", null_value == "@NULL@")
    except Exception as exc:
        skip_checks(3, f"TableProcessor.process_output raised {type(exc).__name__}: {exc}")

    print("=" * 70)
    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    print("=" * 70)
    if failed != 0:
        raise SystemExit(1)
