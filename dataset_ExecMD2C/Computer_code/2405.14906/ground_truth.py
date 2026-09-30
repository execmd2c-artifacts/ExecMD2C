import os
import re
import logging
import subprocess
from typing import Dict, List, Tuple


CODE_INTERPRETER_SYSTEM_PROMPT = """You are an AI code interpreter.
Your goal is to help users do a variety of jobs by executing Python code.

You should:
1. Comprehend the user's requirements carefully & to the letter.
2. Give a brief description for what you plan to do & call the provided function to run code.
3. Provide results analysis based on the execution output.
4. If error occurred, try to fix it.
5. Response in the same language as the user."""


TOOLS_CODE = """
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os,sys
import re
from datetime import datetime
from sympy import symbols, Eq, solve
import torch 
import requests
from bs4 import BeautifulSoup
import json
import math
import yfinance
import time
"""


class BaseCodeInterpreter:
    def __init__(self):
        self.dialog = [
            {
                "role": "system",
                "content": CODE_INTERPRETER_SYSTEM_PROMPT,
            },
        ]

    @staticmethod
    def extract_code_blocks(text: str):
        pattern = r"```(?:python\n)?(.*?)```"  # Match optional 'python\n' but don't capture it
        code_blocks = re.findall(pattern, text, re.DOTALL)
        return [block.strip() for block in code_blocks]

    def execute_code_and_return_output(self, generated_code_block, matches_with_pip,
                                    image_name, container_name,
                                    dockerfile_name, sandbox_path) -> str:

        # Initialize the file/image/dockerfile information for each langauges
        code_blocks_output = {}
        # sandbox_path = "/home/uconn/BinLei/Data_engine/NLP_Task/multi_codes_demo/sandbox"
        file_map = {"python": "script.py",
                    "cpp": "script.cpp",
                    "fortran": "script.f90"}
        # image_map = {"python": "python-sandbox",
        #              "cpp": "cpp-sandbox",
        #              "fortran": "fortran-sandbox"}
        # dockerfile_map = {"python": "python/Dockerfile.python",
        #                   "cpp": "cpp/Dockerfile.cpp",
        #                   "fortran": "fortran/Dockerfile.fortran"}
        image_map = {"python": f"{image_name}",
                    "cpp": "cpp-sandbox",
                    "fortran": "fortran-sandbox"}
        dockerfile_map = {"python": f"{dockerfile_name}",
                        "cpp": "cpp/Dockerfile.cpp",
                        "fortran": "fortran/Dockerfile.fortran"}
        has_problem = True
        for lang, code in generated_code_block.items():
            if lang == "sh":
                lang = "python"
            # write the script into the corresponsing file
            file_name = file_map[lang]
            image_name = image_map[lang]
            # file_path = f"{sandbox_path}/{lang}/{file_name}"
            file_path = f"{sandbox_path}/{file_name}"
            with open(file_path, 'w') as file:
                file.write(code)

            # check and build the image
            image_name = image_map[lang]
            check_command = ["docker", "images", "-q", image_name]
            image_exists = subprocess.run(check_command, capture_output=True, text=True).stdout.strip()

            if not image_exists:
                dockerfile_path = os.path.join(sandbox_path, dockerfile_map[lang])
                # lang_sandbox_path = f"{sandbox_path}/{lang}/"
                lang_sandbox_path = f"{sandbox_path}"
                build_command = ["docker", "build", "-t", image_name, "-f", dockerfile_path, lang_sandbox_path]
                build_result = subprocess.run(build_command, capture_output=True, text=True)
                if build_result.returncode != 0:
                    print(f"Failed to build image {image_name}: {build_result.stderr}")
                    code_blocks_output[lang] = f"Failed to build image {image_name}: {build_result.stderr}"
                    continue

            # give docker the access to run the comile_run file
            # script_path = f"{sandbox_path}/{lang}/compile_run.sh"
            script_path = f"{sandbox_path}/compile_run.sh"
            chmod_command = ["chmod", "+x", script_path]
            chmod_result = subprocess.run(chmod_command, capture_output=True, text=True)
            # volume_mount = f"{sandbox_path}/{lang}:/app"
            volume_mount = f"{sandbox_path}:/app"

            # install external library for python if there are related commonds
            pip_command = None
            print("matches_with_pip", matches_with_pip)
            if lang == "python" and matches_with_pip:
                pip_commands = []
                for match in matches_with_pip:
                    pattern = r'^pip install.*'
                    matches_pip = re.findall(pattern, match, re.MULTILINE)
                    print("matches_pip", matches_pip)
                    for match_pip in matches_pip:
                        if match_pip:
                            pip_commands.append(match_pip.replace('\n', ' ').strip())
                print(f"pip_command:{pip_command}")
                pip_command = " && ".join(pip_commands)
                print(f"pip_command:{pip_command}")
                if pip_command:
                    # command = ["docker", "exec", "container_python", "sh", "-c", f"{pip_command}"]
                    command = ["docker", "exec", f"{container_name}", "sh", "-c", f"{pip_command}"]
                    # print(f"command:{command}")
                    try:
                        logging.info("Start to install related packages...")
                        pip_result = subprocess.run(command, check=True,
                                                    stdout=subprocess.PIPE,
                                                    stderr=subprocess.PIPE,
                                                    text=True)
                    except subprocess.CalledProcessError as e:
                        pip_result = e
                    # command = ["docker", "exec", "container_python", "python", "/app/script.py"]
                    command = ["docker", "exec", f"{container_name}", "python", "/app/script.py"]

            # if there is no external library for python, execute the code directly
            else:
                # command = ["docker", "exec", "container_python", "python", "/app/script.py"]
                command = ["docker", "exec", f"{container_name}", "python", "/app/script.py"]
            # result = subprocess.run(command, capture_output=True, text=True)
            try:
                logging.info("Start to run the code...")
                result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            except subprocess.TimeoutExpired:
                code_blocks_output[lang] = "Command execution timed out. This is probably because the function needs to run continuously."
                continue
            # record all the information into the code_blocks_output
            if result.stdout == "":
                result.stdout = "None"
            if result.stderr == "":
                has_problem = False
                result.stderr = "None"
            if pip_command:
                code_blocks_output[lang] = f"pip_result.stdout: \n{pip_result.stdout}\n pip_result.stderr: \n{pip_result.stderr}\n result.stdout:\n{result.stdout}\nresult.stderr:\n{result.stderr}"
            else:
                code_blocks_output[lang] = f"result.stdout:\n{result.stdout}\nresult.stderr:\n{result.stderr}"
        return has_problem, code_blocks_output


class AutoCoderInterpreter(BaseCodeInterpreter):
    def __init__(
        self,
        model_path: str,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        super().__init__()
        self.model_path = model_path
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.MAX_CODE_OUTPUT_LENGTH = 1000

    def dialog_to_prompt(self, dialog: List[Dict]) -> str:
        inputs = self.tokenizer.apply_chat_template(dialog, return_tensors="pt")

        return inputs

    def extract_code_blocks(self, prompt: str) -> Tuple[str, str, str]:
        has_code = False
        patterns = {
            "sh": re.escape("<API_RUN_START>```sh") + r"(.*?)" + re.escape("```<API_RUN_STOP>"),
            "python": re.escape("<API_RUN_START>```python") + r"(.*?)" + re.escape("```<API_RUN_STOP>"),
        }
        generated_code_block = {"sh": "", "python": ""}

        for lang, pattern in patterns.items():
            matches = re.findall(pattern, prompt, re.DOTALL)
            if matches:
                generated_code_block[lang] = matches[-1].strip()
                has_code = True

        return has_code, generated_code_block

    def clean_code_output(self, output: str) -> str:
        if self.MAX_CODE_OUTPUT_LENGTH < len(output):
            return (
                output[: self.MAX_CODE_OUTPUT_LENGTH // 5]
                + "\n...(truncated due to length)...\n"
                + output[-self.MAX_CODE_OUTPUT_LENGTH // 5:]
            )

        return output


class JupyterNotebook:
    def __init__(self):
        self.km = None
        self.kc = None

    def clean_output(self, outputs):
        outputs_only_str = list()
        for i in outputs:
            if type(i) == dict:
                if "text/plain" in list(i.keys()):
                    outputs_only_str.append(i["text/plain"])
            elif type(i) == str:
                outputs_only_str.append(i)
            elif type(i) == list:
                error_msg = "\n".join(i)
                error_msg = re.sub(r"\x1b\[.*?m", "", error_msg)
                outputs_only_str.append(error_msg)

        return "\n".join(outputs_only_str).strip()

    def add_and_run(self, code_string):
        raise NotImplementedError("Kernel execution is excluded from this benchmark extraction.")

    def close(self):
        """Shutdown the kernel."""
        if self.km is not None:
            self.km.shutdown_kernel()


if __name__ == "__main__":
    import builtins
    import tempfile
    from pathlib import Path
    from types import SimpleNamespace
    from unittest.mock import patch

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

    # Test group 1: AutoCoderInterpreter.extract_code_blocks
    try:
        interpreter = AutoCoderInterpreter(model_path="mock")
        prompt = (
            "Plan\n<API_RUN_START>```sh\npip install pandas\n```<API_RUN_STOP>"
            "\nText\n<API_RUN_START>```python\nprint('old')\n```<API_RUN_STOP>"
            "\nAgain\n<API_RUN_START>```python\nprint('new')\n```<API_RUN_STOP>"
        )
        has_code, blocks = interpreter.extract_code_blocks(prompt)
        check("AutoCoder extract output not None", blocks is not None)
        if blocks is None:
            skip_checks(4, "extract_code_blocks returned None")
        else:
            check("AutoCoder extract has_code", has_code is True)
            check("AutoCoder extract keys", set(blocks.keys()) == {"sh", "python"})
            check("AutoCoder extract latest python", blocks["python"] == "print('new')", blocks["python"])
            check("AutoCoder extract shell block", blocks["sh"] == "pip install pandas", blocks["sh"])
    except Exception as exc:
        skip_checks(5, f"AutoCoderInterpreter.extract_code_blocks raised {type(exc).__name__}: {exc}")

    # Test group 2: BaseCodeInterpreter.execute_code_and_return_output
    try:
        interpreter = BaseCodeInterpreter()
        calls = []
        written = {}

        def fake_run(command, *args, **kwargs):
            calls.append(command)
            if command[:3] == ["docker", "images", "-q"]:
                return SimpleNamespace(stdout="image-id\n", stderr="", returncode=0)
            if command[:2] == ["chmod", "+x"]:
                return SimpleNamespace(stdout="", stderr="", returncode=0)
            if command[:3] == ["docker", "exec", "container"] and command[3:5] == ["sh", "-c"]:
                return SimpleNamespace(stdout="installed", stderr="", returncode=0)
            if command[:3] == ["docker", "exec", "container"] and command[3:] == ["python", "/app/script.py"]:
                return SimpleNamespace(stdout="42\n", stderr="", returncode=0)
            return SimpleNamespace(stdout="", stderr="unexpected command", returncode=1)

        real_open = builtins.open

        def fake_open(path, mode="r", *args, **kwargs):
            if "w" in mode and str(path).endswith("script.py"):
                class Writer:
                    def __enter__(self_inner):
                        return self_inner

                    def __exit__(self_inner, exc_type, exc, tb):
                        return False

                    def write(self_inner, value):
                        written[str(path)] = value
                        return len(value)
                return Writer()
            return real_open(path, mode, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("subprocess.run", side_effect=fake_run), patch("builtins.open", side_effect=fake_open):
                has_problem, outputs = interpreter.execute_code_and_return_output(
                    {"python": "print(42)"},
                    ["pip install numpy\npython ignored"],
                    "image",
                    "container",
                    "Dockerfile.python",
                    tmpdir,
                )
        check("Base execute output not None", outputs is not None)
        if outputs is None:
            skip_checks(5, "execute_code_and_return_output returned None")
        else:
            check("Base execute success flag", has_problem is False)
            check("Base execute writes script", any(value == "print(42)" for value in written.values()))
            check("Base execute extracts pip install", any(call[:5] == ["docker", "exec", "container", "sh", "-c"] and call[-1] == "pip install numpy" for call in calls))
            check("Base execute runs python in container", any(call == ["docker", "exec", "container", "python", "/app/script.py"] for call in calls))
            check("Base execute output format", "pip_result.stdout" in outputs["python"] and "result.stdout:\n42" in outputs["python"])
    except Exception as exc:
        skip_checks(6, f"BaseCodeInterpreter.execute_code_and_return_output raised {type(exc).__name__}: {exc}")

    # Test group 3: AutoCoderInterpreter.clean_code_output
    try:
        interpreter = AutoCoderInterpreter(model_path="mock")
        interpreter.MAX_CODE_OUTPUT_LENGTH = 20
        long_output = "abcdefghijklmnopqrstuvwxyz"
        cleaned = interpreter.clean_code_output(long_output)
        short_output = interpreter.clean_code_output("short")
        check("AutoCoder clean output not None", cleaned is not None)
        if cleaned is None:
            skip_checks(3, "clean_code_output returned None")
        else:
            check("AutoCoder clean truncates long output", "(truncated due to length)" in cleaned)
            check("AutoCoder clean keeps boundaries", cleaned.startswith("abcd") and cleaned.endswith("wxyz"))
            check("AutoCoder clean leaves short output", short_output == "short")
    except Exception as exc:
        skip_checks(4, f"AutoCoderInterpreter.clean_code_output raised {type(exc).__name__}: {exc}")

    # Test group 4: JupyterNotebook.clean_output
    try:
        notebook = JupyterNotebook()
        raw_outputs = [
            {"text/plain": "array([1, 2])", "ignored": "x"},
            "printed line\n",
            ["\x1b[31mTraceback\x1b[0m", "ValueError: bad"],
            {"image/png": "..."},
        ]
        cleaned = notebook.clean_output(raw_outputs)
        check("Jupyter clean output not None", cleaned is not None)
        if cleaned is None:
            skip_checks(4, "clean_output returned None")
        else:
            check("Jupyter clean includes text/plain", "array([1, 2])" in cleaned)
            check("Jupyter clean includes stream", "printed line" in cleaned)
            check("Jupyter clean strips ansi", "\x1b[" not in cleaned and "Traceback" in cleaned)
            check("Jupyter clean ignores unsupported mime", "image/png" not in cleaned)
    except Exception as exc:
        skip_checks(5, f"JupyterNotebook.clean_output raised {type(exc).__name__}: {exc}")

    print(f"TEST RESULTS: {passed} / {passed + failed} checks passed")
    if failed != 0:
        raise SystemExit(1)
