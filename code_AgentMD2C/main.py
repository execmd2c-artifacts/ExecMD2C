import argparse
import ast
import base64
import json
import logging
import os
import re
import sys

from openai import OpenAI
from tqdm import tqdm


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)



def call_llm(args, messages):
    """Send one chat request using the configured model and temperature.

    Raise an exception for missing or empty response text so the calling agent
    can retry. Each agent retains its original three-attempt retry policy.
    """
    client = OpenAI(
        api_key=args.api_key,
        base_url=args.api_base
    )
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=args.temperature,
    )
    result = response.choices[0].message.content
    if not isinstance(result, str) or not result.strip():
        raise ValueError("The model returned no text content.")
    return result.strip()



def diagram_understanding(image_data=None, args=None):
    """Analyze a model architecture image.

    Args:
        image_data: Raw PNG image bytes.
        args: Parsed API and model configuration.

    Returns:
        The architecture analysis string, or None after three failed attempts.
    """

    system_prompt = """

    You are a Diagram Understanding Agent. Your task is to analyze the model architecture shown in the image and produce a faithful, structured description of the architecture.
    Requirements:
    1. Identify all visible major components, modules, and blocks in the diagram.
    2. Describe the connections and data flow between modules as accurately as possible.
    3. Do not invent modules, operations, or hyperparameters that are not supported by the image.
    4. If some parts of the diagram are unclear, explicitly mark them as uncertain instead of guessing.
    5. Organize the output into the following sections:
    - Architecture modules and components
    - Connections / Data Flow
    - Ambiguities / Unclear parts
    6. Output only the analysis result, without any additional explanations or comments outside the structured analysis.

    """
    user_prompt = """Please analyze the model architecture shown in the image and produce a faithful, structured description of the architecture."""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:

            analysis = call_llm(args, messages)
            logger.info(f"image_understanding succeeded on attempt {attempt + 1}")
            return analysis
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    logger.error(f"All {max_retries} attempts failed")
    return None



def text_extraction(content=None, args=None):
    """Extract implementation details from paper text.

    Args:
        content: The selected paper content passed to the original prompt.
        args: Parsed API and model configuration.

    Returns:
        The extracted description string, or None after three failed attempts.
    """

    system_prompt = """

    You are a Textual Information Extraction Agent. Your task is to extract implementation-relevant information from the provided text and produce a structured description of the model architecture.
    Requirements:
    1. Focus only on information useful for implementing the model in code.
    2. Identify all explicitly described major components, modules, and their internal implementation details.
    3. Describe the connections and data flow between modules as accurately as possible.
    4. Do not introduce information that is not supported by the text.
    5. Avoid long narrative descriptions. Keep the output concise and structured.
    6. Organize the output into the following sections:
    - Architecture modules and components
    - Connections / Data Flow
    - Ambiguities / Unclear parts
    7. Output only the structured result, without any extra explanations or comments.
    """
    user_prompt = f"""

                Extract implementation-relevant information from the following  text.

                Text: {content}

                """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ]
        }
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            text_mining = call_llm(args, messages)
            logger.info(f"text_mining succeeded on attempt {attempt + 1}")
            return text_mining
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    logger.error(f"All {max_retries} attempts failed")
    return None



def semantic_fusion (analysis=None, text_mining=None, image_data=None, args=None):
    """Reconcile image analysis with information extracted from paper text.

    Args:
        analysis: Output from image_understanding.
        text_mining: Output from the text mining agent.
        image_data: Reserved for interface compatibility; currently unused.
        args: Parsed API and model configuration.

    Returns:
        The architecture specification string, or None on exhausted retries.
    """

    
    system_prompt = """
    You are a Semantic Fusion Agent. Your task is to integrate and reconcile model architecture information from:
    1. The diagram analysis result
    2. The textual extraction result
    The diagram analysis serves as the primary description of the model architecture, while the textual extraction result provides complementary information when relevant. The goal is to produce a consistent, complete, and implementation-ready model architecture specification.
    Requirements:
    1. Use the diagram analysis result as the primary architectural basis, including the identified modules, connections, data flow, and structural organization.
    2. Carefully examine the textual extraction result for additional information that can refine or clarify the diagram analysis, such as:
    - module functionality,
    - internal operations,
    - input/output relationships,
    - execution order,
    - mathematical operations,
    - architectural parameters or hyperparameters.
    3. If the textual extraction result provides more detailed or explicit information about an element already identified in the diagram, incorporate this information to enrich and refine the corresponding architectural description.
    4. If the textual extraction result does not provide additional useful architectural information, retain the corresponding diagram analysis result without forcing
    textual information into the specification or weakening information already obtained from the diagram.
    5. Merge duplicated or overlapping information from the two sources instead of simply concatenating their analysis results.
    6. When the two sources provide complementary information, combine them into a single coherent description while preserving all mutually consistent and implementation-relevant details.
    7. When conflicts occur:
    - Prefer information that is explicitly and unambiguously specified.
    - For implementation details that are not visually specified or remain ambiguous in the diagram, use explicit textual descriptions to resolve or supplement them.
    - Do not override clear structural information from the diagram with vague, indirect, or generic textual descriptions.
    - If a conflict cannot be reliably resolved from the provided information, preserve the reliable information and record the unresolved issue under "Ambiguities / Unclear parts".
    - Output only the resolved specification; do not explain the reasoning or resolution process.
    8. Do not introduce any module, operation, connection, relationship, architectural parameter, or hyperparameter that is not supported by either the diagram analysis result or the textual extraction result.
    9. Do not infer missing implementation details solely from general knowledge, common architectural conventions, or assumptions about how the model is usually implemented.
    10. Keep the output concise, structured, and implementation-oriented. Preserve details that are useful for reconstructing the model in code, while removing redundant descriptive or background information.
    11. Organize the output into the following sections:
    - Architecture Modules and Components
    - Connections / Data Flow
    - Ambiguities / Unclear Parts
    12. Output only the fused architecture specification, without any additional explanations, comments, or discussion of the fusion process.
    """

    user_prompt = f"""

        Diagram Analysis :

        {analysis}



        Text Mining:

        {text_mining}



        Please merge them into a final canonical architecture specification.

        """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ]
        }
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            architecture_specification = call_llm(args, messages)
            logger.info(f"semantic_disambiguation succeeded on attempt {attempt + 1}")
            return architecture_specification
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    logger.error(f"All {max_retries} attempts failed")
    return None



def planning_implementation(architecture_analysis=None, generated_code=None, validation_feedback=None, image_data=None, args=None):
    """Produce an implementation plan from the image and architecture analysis.

    Args:
        architecture_analysis: The architecture description used for planning.
        generated_code: Reserved for interface compatibility; currently unused.
        validation_feedback: Reserved for interface compatibility; currently unused.
        image_data: Raw PNG image bytes.
        args: Parsed API and model configuration.

    Returns:
        The implementation plan string, or None after three failed attempts.
    """


    system_prompt = """

    You are an Implementation Planning Agent. Your task is to generate an implementation plan that guides the conversion of a model architecture diagram into Python code using the following inputs:
    1. the original model diagram image
    2. the architecture analysis result
    The goal is to produce an implementation-ready plan that can guide code generation.
    Requirements:
    1. Do not introduce modules or operations that are not supported by the image or the architecture specification.
    2. Do not generate any Python code.
    3. Only output a structured implementation plan.

    """

    user_prompt = f"""

    Architecture Analysis Result:

    {architecture_analysis}



    Generate the implementation plan.

    """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            planning = call_llm(args, messages)
            logger.info(f"planning_generation succeeded on attempt {attempt + 1}")
            return planning
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    logger.error(f"All {max_retries} attempts failed")
    return None



def code_generation(image_data=None, template=None, planning=None, args=None):
    """Complete the supplied template using the image and implementation plan.

    Args:
        image_data: Raw PNG image bytes.
        template: Original Python template text.
        planning: Output from planning_generation.
        args: Parsed API and model configuration.

    Returns:
        The raw generated response string, or None on exhausted retries.
        The caller extracts the Python code before saving it.
    """


    system_prompt = """

    You are a Code Generation Agent. Your task is to fill in missing implementations in the given Python template, using the following references:
    1. the model diagram image
    2. the implementation plan
    CRITICAL RULES (HARD CONSTRAINTS):
    1. You must preserve the entire template structure exactly.
    - Do not delete any class, function, import, or executable code line.
    - You must remove all comments from the template (including inline comments, standalone comments, and docstrings).
    - Do not add new classes or functions.
    - Do not rename anything.
    2. You are only allowed to replace placeholder regions such as:
    - pass
    - TODO
    - missing implementation parts
    3. The number of classes and functions in the output must be identical to the template.
    4. Every function signature must remain unchanged.
    5. You must use imports exactly as defined in the template.
    - Extract and follow the exact import statements from the template.
    - You must use the exact imported names and aliases as they appear.
    - If the template contains: import torch
    you must use `torch`, and must not use `th`.
    - If the template contains: import torch as th
    you must use `th`, and must not use `torch`.
    - If the template contains: from torch import nn
    you must use `nn`, and must not use `torch.nn`.
    - DO not assume or infer alternative aliases.
    - DO not introduce any new module names.
    - Every external reference must match an existing import exactly.
    6. You must not call any method, function, attribute, class, or module that is not explicitly defined in the template or legally available from existing imports.
    - Do not invent helper methods.
    - Do not assume hidden methods exist.
    - Before calling `self.xxx(...)`, ensure that `xxx` is defined in the same class or inherited from a known superclass.
    7. If you cannot implement a part safely, leave it as `pass` — Do not delete it and do not call undefined helpers.
    8. Do not include any explanations, comments, or markdown outside the code block, and ensure that the output code contains no comments at all.
    9. Output only the completed Python template inside ``` ```.
    Violation of any rule is not allowed.

    """
    user_prompt = f"""

        Implementation Plan:

        {planning}



        Complete  the following Python template:

        Output the fully completed Python template only. Enclose the entire code within triple backticks ``` ```.



        Python Template:

        ```

        {template}

        ```

        """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            generated_code = call_llm(args, messages)
            logger.info(f"code_generation succeeded on attempt {attempt + 1}")
            return generated_code
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    logger.error(f"All {max_retries} attempts failed")
    return None



def code_repair(image_data=None, original_code=None, execution_result=None, args=None):
    """Repair generated code using a paper-specific execution log.

    Args:
        image_data: Raw PNG image bytes.
        original_code: The original code generation response.
        execution_result: The complete execution log text.
        args: Parsed API and model configuration.

    Returns:
        The raw repair response string, or None after three failed attempts.
        The caller retains the original code if repair does not succeed.
    """

    system_prompt = """

        You are a Code Repair Agent. You are provided with:
        1. The original model diagram image
        2. The original implementation code of the model diagram
        3. The execution result of the code
        Your task is to analyze the model architecture image and repair the original implementation according to the execution result. Finally, output the complete repaired code. Do not add any explanations or comments unrelated to the code.
        CRITICAL RULES (HARD CONSTRAINTS):
        1. You must repair the error directly within the original code.
        - Make only the changes necessary to fix the current error.
        - All unrelated code must remain unchanged.
        2. You must preserve the overall structure of the original code.
        - Do not add any new standalone classes or module-level functions.
        - You may modify the contents of existing classes as necessary to fix the error.
        3. The repaired code must preserve the original:
        - Import statements
        - Class names
        - Module-level function names
        4. You must use imports exactly as defined in the original code.
        - Follow the exact import statements in the original code.
        - Use the exact imported names and aliases as they appear.
        - Do not add, remove, rename, or replace any imports.
        5. You must not call any method, function, attribute, class, or module that is not explicitly defined in the original code or legally available from the existing imports.
        - Do not invent helper methods.
        - Do not assume that hidden methods exist.
        - Before calling `self.xxx(...)`, ensure that `xxx` is defined in the same class or inherited from a known superclass.
        6. The output code must contain no comments or docstrings.
        - Do not include any explanations or text outside the code block.
        7. Output only the complete repaired Python code inside triple backticks. Violation of any rule is not allowed.

        """
    user_prompt = f"""

        Repair the original implementation code according to the attached model architecture image and the execution result.

        Execution Result:

        {execution_result}

        Original Implementation Code:

        ```

        {original_code}

        ```

        Make only the changes necessary to fix the reported error and preserve the original code structure.

        Output only the complete repaired Python code inside triple backticks ``` ```.

        """

    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64.b64encode(image_data).decode()}"
                    }
                }
            ]
        }
    ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            generated_code = call_llm(args, messages)
            logger.info(f"code_repair succeeded on attempt {attempt + 1}")
            return generated_code
        except Exception as e:
            logger.error(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < max_retries - 1:
                logger.info("Retrying...")
                continue
    logger.error(f"All {max_retries} attempts failed")
    return None



def load_dataset(args):
    """Collect (image_path, template_path, content_path) tuples across domains.

    Preserve the original selection rule: paper directory names begin with a
    digit. Content files are read only when textual input is enabled, so their
    absence does not exclude papers from image-only experiments.
    """
    dataset_path = args.dataset_path
    if not os.path.isdir(dataset_path):
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_path}")

    dataset = []
    for domain in os.listdir(dataset_path):
        domain_path = os.path.join(dataset_path, domain)
        if not os.path.isdir(domain_path):
            continue

        for paper in os.listdir(domain_path):
            paper_path = os.path.join(domain_path, paper)
            if not paper[:1].isdigit() or not os.path.isdir(paper_path):
                continue
            image_path = os.path.join(paper_path, "model.png")
            template_path = os.path.join(paper_path, args.template_type)
            content_path = os.path.join(domain_path, "content.json")
            if not os.path.isfile(image_path) or not os.path.isfile(template_path):
                logger.warning("Missing image or template: %s; %s", image_path, template_path)
                continue
            dataset.append((image_path, template_path, content_path))

    logger.info("Found %d valid image-template pairs.", len(dataset))
    return dataset



def parse_args():
    """Parse the original command-line options and the optional repair switch."""
    parser = argparse.ArgumentParser(description='Model Generation and Evaluation')
    parser.add_argument('--dataset_path', type=str, default='dataset',
                      help='Root directory of the dataset')
    # Output paths and dataset selection.
    parser.add_argument('--result_count_path', type=str, default="result/gpt54nano20260317/Abstract/result_count_without_main.json",
                      help='Path to the JSON list of completed paper keys')
    parser.add_argument('--generate_path', type=str, default="llm_generate.py",
                      help='Generated code filename or relative path within each paper output directory')
    parser.add_argument('--template_type', type=str, default="template.py",
                      help='Template filename within each paper directory')
    parser.add_argument('--agent_out_path', type=str, default='code/result/gpt54nano20260317/Abstract/agent_output.json',
                      help='Path to the JSON mapping of agent outputs by paper key')
    parser.add_argument('--feedback_path', type=str, default='result/feedback_path/gpt54nano20260317/Abstract',
                      help='Root directory containing <domain>/<domain>__<paper_id>.log')
    parser.add_argument('--content_type', type=str, default='Abstract',
                      help='Paper content field to use, such as Abstract')
    # Control whether paper text is used as an input.
    parser.add_argument('--is_content', type=int, choices=[0, 1], default=1,
                      help='Use textual paper content: 1 to enable, 0 to disable')
    # API configuration.
    parser.add_argument('--api_key', type=str, 
                      default='...',
                      help='API key')
    parser.add_argument('--api_base', type=str,
                      default='...',
                      help='API base URL')
    parser.add_argument('--model', type=str, default='gpt-5.4-nano-2026-03-17',
                      help='Model to use for generation')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for generation')
    parser.add_argument(
        '--is_repair', type=int, choices=[0, 1], default=0,
        help='Load execution logs and enable code repair: 1 to enable, 0 to disable',
    )
    args = parser.parse_args()
    return args



def extract_code_from_response(generated_code: str) -> str:
    """Extract the complete Python code, preferring a syntactically valid block."""
    if not generated_code:
        return ""

    text = generated_code.strip()
    lines = text.splitlines()

    opening_fence_pattern = re.compile(
        r"^\s*```(?:python)?\s*$",
        re.IGNORECASE
    )

    start_index = next(
        (
            index
            for index, line in enumerate(lines)
            if opening_fence_pattern.fullmatch(line)
        ),
        None
    )

    if start_index is None:
        return text

    closing_indices = [
        index
        for index in range(start_index + 1, len(lines))
        if re.fullmatch(r"\s*```\s*", lines[index])
    ]

    if not closing_indices:
        return "\n".join(lines[start_index + 1:]).strip()

    # Try the widest code block first to avoid truncating embedded fences.
    # The outer closing fence is usually the last standalone triple backtick.
    for end_index in reversed(closing_indices):
        candidate = "\n".join(
            lines[start_index + 1:end_index]
        ).strip()

        try:
            ast.parse(candidate)
            return candidate
        except SyntaxError:
            continue

    # If no candidate parses, preserve the widest block for later execution
    # feedback rather than silently discarding the generated implementation.
    return "\n".join(
        lines[start_index + 1:closing_indices[-1]]
    ).strip()



def main():
    
    args = parse_args()
    try:
        dataset = load_dataset(args)
        # A bare filename has no dirname; use the working directory in that case.
        for path in (args.result_count_path, args.agent_out_path):
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        if os.path.exists(args.result_count_path):
            with open(args.result_count_path, "r", encoding="utf-8") as file:
                result_count = json.load(file)
            if not isinstance(result_count, list) or not all(
                isinstance(key, str) for key in result_count
            ):
                raise ValueError("The completion file must contain a JSON list of paper keys.")
        else:
            result_count = []
            with open(args.result_count_path, "w", encoding="utf-8") as file:
                json.dump(result_count, file, indent=2, ensure_ascii=False)
            logger.info("Created completion file: %s", args.result_count_path)

        if os.path.exists(args.agent_out_path):
            with open(args.agent_out_path, "r", encoding="utf-8") as file:
                agent_output_all = json.load(file)
            if not isinstance(agent_output_all, dict):
                raise ValueError("The agent output file must contain a JSON object.")
        else:
            agent_output_all = {}
    except (OSError, ValueError) as error:
        # Do not overwrite invalid existing checkpoints with an empty state.
        logger.error("Cannot initialize the experiment: %s", error)
        return 1

    completed = set(result_count)
    count = 0
    fail_count = 0
    for image_path, template_path, content_path in tqdm(
        dataset, desc="Processing papers", unit="paper"
    ):
        domain = os.path.basename(os.path.dirname(os.path.dirname(template_path)))
        paper_id = os.path.basename(os.path.dirname(template_path))
        paper_key = f"{domain}/{paper_id}"
        # Check identities rather than list lengths: a checkpoint may contain
        # duplicate keys or entries from a different dataset selection.
        if paper_key in completed:
            logger.info("Paper %s is already complete; skipping.", paper_key)
            continue

        agent_output_one = {}
        try:
            with open(image_path, "rb") as file:
                image_data = file.read()
            with open(template_path, "r", encoding="utf-8") as file:
                template_code = file.read()

            # Image-only runs must not depend on content files or content keys.
            if args.is_content == 1:
                with open(content_path, "r", encoding="utf-8") as file:
                    content_domain = json.load(file)
                content = content_domain[paper_id][args.content_type]
                if content is None or (isinstance(content, str) and not content.strip()):
                    raise ValueError(f"Paper content is empty for {paper_key}.")

            # 1. Analyze the model architecture image.
            analysis = image_understanding(image_data, args)
            agent_output_one["analysis"] = analysis
            if not analysis:
                raise RuntimeError("Image understanding failed.")

            if args.is_content == 1:
                # 2. Extract implementation details from the selected paper text.
                mining = text_extraction(content, args)
                agent_output_one["mining"] = mining
                if not mining:
                    raise RuntimeError("Text mining failed.")

                # 3. Reconcile textual and visual architecture information.
                architecture_specification = semantic_fusion(
                    analysis=analysis, text_mining=mining, args=args,
                )
                agent_output_one["architecture_specification"] = architecture_specification
                if not architecture_specification:
                    raise RuntimeError("Semantic disambiguation failed.")
            else:
                architecture_specification = analysis

            # 4. Plan the implementation with the original agent interface.
            planning = planning_generation(
                architecture_analysis=architecture_specification,
                generated_code=None,
                validation_feedback=None,
                image_data=image_data,
                args=args,
            )
            agent_output_one["planning"] = planning
            if not planning:
                raise RuntimeError("Implementation planning failed.")

            # 5. Generate code before considering optional execution feedback.
            generated_code = code_generation(
                image_data=image_data, template=template_code, planning=planning, args=args,
            )
            agent_output_one["generated_code"] = generated_code
            if not isinstance(generated_code, str) or not extract_code_from_response(generated_code):
                fail_count += 1
                logger.warning("Code generation failed for %s; skipping.", paper_key)
                if fail_count >= 50:
                    logger.error("Stopping after %d accumulated generation failures.", fail_count)
                    break
                continue

            # 6. Optional repair: no log lookup or read occurs when disabled.
            if args.is_repair == 1:
                feedback_path = os.path.join(
                    args.feedback_path, domain, f"{domain}__{paper_id}.log",
                )
                try:
                    if os.path.isfile(feedback_path):
                        with open(feedback_path, "r", encoding="utf-8", errors="replace") as file:
                            execution_result = file.read()
                        if execution_result.strip():
                            repaired_code = code_repair(
                                image_data=image_data,
                                original_code=generated_code,
                                execution_result=execution_result,
                                args=args,
                            )
                            candidate = (
                                extract_code_from_response(repaired_code)
                                if isinstance(repaired_code, str) else ""
                            )
                            if not candidate:
                                raise ValueError("Repair returned no code.")
                            # Reject unusable repair output without replacing the
                            # original response. Compilation does not execute code.
                            compile(candidate, "<repaired_code>", "exec")
                            generated_code = repaired_code
                            agent_output_one["code_repair"] = repaired_code
                        else:
                            logger.info("Execution log is empty for %s; skipping repair.", paper_key)
                    else:
                        logger.info("No execution log for %s: %s", paper_key, feedback_path)
                except Exception as error:
                    logger.warning(
                        "Code repair failed for %s; retaining original generated code: %s",
                        paper_key, error,
                    )

            # Preserve extraction behavior for initial generations so their
            # execution failures can still be measured by the experiment.
            generated_code = extract_code_from_response(generated_code)
            output_path = os.path.join(
                os.path.dirname(args.result_count_path), domain, paper_id, args.generate_path,
            )
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(generated_code)

            # Persist artifacts before recording completion, permitting retries
            # when saving the generated code or agent outputs fails.
            agent_output_all[paper_key] = agent_output_one
            with open(args.agent_out_path, "w", encoding="utf-8") as file:
                json.dump(agent_output_all, file, indent=2)
            updated_result_count = result_count + [paper_key]
            with open(args.result_count_path, "w", encoding="utf-8") as file:
                json.dump(updated_result_count, file, indent=2, ensure_ascii=False)
            result_count = updated_result_count
            completed.add(paper_key)
            count += 1
            logger.info("Paper %s processed successfully.", paper_key)
        except Exception as error:
            logger.error("Failed to process paper %s; skipping: %s", paper_key, error)
            continue

    logger.info("Successfully processed %d papers in this run.", count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
