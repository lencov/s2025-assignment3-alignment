#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

# Assuming datasets library is installed, but we'll load JSONL manually
# from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Define prompt templates (adjust paths if necessary)
ZERO_SHOT_SYSTEM_PROMPT_FILE = Path("cs336_alignment/prompts/zero_shot_system_prompt.prompt")

# Function to format the prompt using the system prompt
def format_sst_prompt(system_prompt: str, instruction: str) -> str:
    """Formats the full prompt for a given SST instruction using the system prompt."""
    # Embed the instruction within the general system prompt structure
    if "{instruction}" not in system_prompt:
        print("Warning: '{instruction}' placeholder not found in system prompt. Appending instruction.")
        return f"{system_prompt}\n\n# Query:\n```{instruction}```\n\n# Answer:\n```"
    else:
        return system_prompt.replace("{instruction}", instruction)

def load_jsonl(file_path):
    """Loads a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {line.strip()} - Error: {e}")
    return data

def main(args):
    # --- 1. Load System Prompt ---
    print(f"Loading system prompt from: {args.system_prompt_file}")
    try:
        with open(args.system_prompt_file, 'r') as f:
            system_prompt = f.read().strip()
        print("System prompt loaded successfully.")
    except FileNotFoundError:
        print(f"Error: System prompt file not found at {args.system_prompt_file}")
        return
    except Exception as e:
        print(f"Error reading system prompt file: {e}")
        return

    # --- 2. Load SimpleSafetyTests Dataset ---
    print(f"Loading SimpleSafetyTests dataset from: {args.dataset_path}")
    try:
        # Assuming the dataset is JSONL and has a 'prompt' key for the instruction
        sst_dataset = load_jsonl(args.dataset_path)
        # Check if dataset is loaded and is a list
        if not isinstance(sst_dataset, list) or not sst_dataset:
             print(f"Error: Failed to load or empty dataset from {args.dataset_path}")
             return
         # Verify expected key exists in the first item (assuming consistent structure)
        if args.instruction_key not in sst_dataset[0]:
             print(f"Error: Instruction key '{args.instruction_key}' not found in the first item of the dataset.")
             print(f"Available keys: {list(sst_dataset[0].keys())}")
             return
        print(f"Loaded {len(sst_dataset)} examples from SimpleSafetyTests.")
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {args.dataset_path}")
        return
    except Exception as e:
        print(f"Error loading SimpleSafetyTests dataset: {e}")
        return

    # Limit examples if specified
    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} examples.")
        sst_dataset = sst_dataset[:args.limit]

    # --- 3. Format all prompts ---
    print("Formatting prompts...")
    instructions = [example[args.instruction_key] for example in sst_dataset]
    prompts = [format_sst_prompt(system_prompt, instruction) for instruction in tqdm(instructions)]
    print(f"Formatted {len(prompts)} prompts.")

    # --- 4. Initialize vLLM ---
    print(f"Initializing vLLM with model: {args.model_path}")
    try:
        llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
        print("vLLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return

    # --- 5. Configure Sampling Parameters ---
    # Greedy decoding
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens, # Adjust as needed for safety responses
        stop=["# Query:"] # Stop token based on the general system prompt format
    )
    print(f"Sampling parameters set: temp={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}, stop={sampling_params.stop}")

    # --- 6. Generate Outputs ---
    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation completed in {generation_time:.2f} seconds.")

    # --- 7. Serialize Results in JSONL format ---
    print(f"Saving results to {args.output_file}...")
    try:
        # Ensure parent directory exists
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for i, output in enumerate(tqdm(outputs)):
                instruction = instructions[i]
                raw_output = output.outputs[0].text.strip()
                # Remove potential trailing ```
                if raw_output.endswith("```"):
                    raw_output = raw_output[:-3].strip()

                # Create JSON object for the line
                result_line = {
                    "prompts_final": instruction, # Use original instruction as required
                    "output": raw_output
                    # Add other fields from original dataset if needed, e.g.:
                    # **sst_dataset[i]
                }
                # Write the JSON object as a line
                f_out.write(json.dumps(result_line) + '\n')
        print("Results saved successfully in JSONL format.")
    except Exception as e:
        print(f"Error saving results to JSONL: {e}")

    # --- Optional: Print throughput (Part b) ---
    throughput = len(outputs) / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SimpleSafetyTests Zero-Shot Outputs")
    parser.add_argument("--model_path", type=str, default="/content/s2025-assignment3-alignment/Qwen/Qwen2.5-0.5B",
                        help="Path to the HuggingFace model directory.")
    parser.add_argument("--dataset_path", type=str, default="data/simple_safety_tests.jsonl",
                        help="Path to the SimpleSafetyTests JSONL file.")
    parser.add_argument("--instruction_key", type=str, default="prompt",
                        help="Key in the JSONL file that contains the instruction/prompt.")
    parser.add_argument("--system_prompt_file", type=Path, default=ZERO_SHOT_SYSTEM_PROMPT_FILE,
                        help="Path to the file containing the zero-shot system prompt.")
    parser.add_argument("--output_file", type=str, default="sst_baseline_results.jsonl",
                        help="Path to save the generated outputs in JSONL format.")
    parser.add_argument("--max_tokens", type=int, default=512, # Allow reasonable length for safety answers/refusals
                        help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of examples for testing (-1 for no limit).")

    args = parser.parse_args()
    main(args) 