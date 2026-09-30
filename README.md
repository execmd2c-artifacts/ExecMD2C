# ExecMD2C

Official code and dataset for **"ExecMD2C: An Execution-Grounded Benchmark and Multi-Agent Framework for Model Diagram-to-Code Generation."**

ExecMD2C studies the generation of executable model implementations from architecture diagrams and accompanying paper text. This repository provides the AgentMD2C implementation and the ExecMD2C benchmark used in our experiments.

## Repository Structure

```text
ExecMD2C/
├── code_AgentMD2C/
│   ├── main.py          # Multi-agent generation pipeline
│   ├── log/             # Execution-feedback logs, when provided
│   └── result/          # Generated code and pipeline checkpoints
└── dataset_ExecMD2C/
    ├── Adversarial/
    ├── Audio/
    ├── Computer_code/
    ├── Computer_Vision/
    ├── Graphs/
    ├── Knowledge_Base/
    ├── Medical/
    ├── Methodology/
    ├── Miscellaneous/
    ├── Music/
    ├── NLP/
    ├── Playing_Games/
    ├── Reasoning/
    ├── Robots/
    ├── Speech/
    └── Time_Series/
```

## Dataset

The ExecMD2C dataset contains **128 model diagram-to-code tasks** from 16 research domains. Each task directory is identified by its paper ID and generally contains:

```text
<paper_id>/
├── model.png            # Model architecture diagram
├── template.py          # Incomplete implementation template
├── ground_truth.py      # Reference implementation
├── environment.yml      # Task-specific execution environment
├── README.md            # Task and source information
└── pretrained_models/   # Optional pretrained-model information
```

Each domain also includes a `content.json` file containing the paper text used by the text-based agent.

## AgentMD2C

`code_AgentMD2C/main.py` implements the multi-agent pipeline for:

1. understanding the model architecture diagram;
2. extracting implementation details from paper text;
3. fusing the visual and textual information;
4. planning and generating the implementation; and
5. optionally repairing generated code using execution feedback.

## Quick Start

Install the dependencies required by the generation pipeline:

```bash
pip install openai tqdm
```

Run AgentMD2C with an OpenAI-compatible multimodal API. The selected model must accept image inputs:

```bash
python code_AgentMD2C/main.py \
  --dataset_path dataset_ExecMD2C \
  --result_count_path code_AgentMD2C/result/run/completed.json \
  --agent_out_path code_AgentMD2C/result/run/agent_output.json \
  --generate_path generated.py \
  --api_key YOUR_API_KEY \
  --api_base YOUR_API_BASE \
  --model YOUR_MODEL_NAME \
  --content_type Abstract \
  --is_content 1 \
  --is_repair 0
```

The command stores generated implementations under:

```text
code_AgentMD2C/result/run/<domain>/<paper_id>/generated.py
```

It also saves the completed-task list in `completed.json` and the intermediate agent outputs in `agent_output.json`, allowing interrupted runs to resume.

Set `--is_content 0` for diagram-only generation. Set `--is_repair 1` to enable repair from existing execution-feedback logs and provide their root directory with `--feedback_path`. The expected log path is `<feedback_path>/<domain>/<domain>__<paper_id>.log`. The current pipeline does not execute generated implementations itself.

The task-specific dependencies needed to execute individual implementations are recorded in each task's `environment.yml` file.

## Notes

- API credentials are not included in this repository.
- Generated outputs and execution logs may vary with the selected model and API configuration.
- Some tasks require pretrained weights or other external resources; see the corresponding task directory for details.
