#!/usr/bin/env python3
import argparse
import csv
import json
import os
import time
from pathlib import Path

from tqdm import tqdm
from vllm import LLM, SamplingParams

# SFT Chat Template (Ensure this matches the template used during training)
SFT_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# Function to format the SST prompt using the SFT chat template
def format_sst_sft_prompt(instruction: str) -> str:
    """Formats the full prompt for a given SST instruction using the SFT template."""
    return SFT_CHAT_TEMPLATE.format(instruction=instruction)

def load_csv_as_dict(file_path):
    """Loads a CSV file into a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
    except Exception as e:
        print(f"Error reading CSV file {file_path}: {e}")
        return None
    return data

def main(args):
    # --- 1. Load SimpleSafetyTests Dataset --- (System prompt loading removed)
    print(f"Loading SimpleSafetyTests dataset from CSV: {args.dataset_path}")
    try:
        sst_dataset = load_csv_as_dict(args.dataset_path)
        if sst_dataset is None:
             print(f"Error: Failed to load dataset from {args.dataset_path}")
             return
        if not sst_dataset:
            print(f"Error: Dataset appears empty: {args.dataset_path}")
            return

         # Verify expected key (column name) exists in the first item
        if args.instruction_key not in sst_dataset[0]:
             print(f"Error: Instruction key/column '{args.instruction_key}' not found in the CSV header.")
             print(f"Available columns: {list(sst_dataset[0].keys())}")
             return
        print(f"Loaded {len(sst_dataset)} examples from SimpleSafetyTests CSV.")
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

    # --- 2. Format all prompts --- (Using SFT format)
    print("Formatting prompts using SFT template...")
    instructions = [example[args.instruction_key] for example in sst_dataset]
    prompts = [format_sst_sft_prompt(instruction) for instruction in tqdm(instructions)] # Use SFT formatter
    print(f"Formatted {len(prompts)} prompts.")

    # --- 3. Initialize vLLM ---
    print(f"Initializing vLLM with SFT model: {args.model_path}") # Updated log message
    try:
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype='half' # Use float16 for T4 compatibility
        )
        print("vLLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return

    # --- 4. Configure Sampling Parameters ---
    # Greedy decoding, use SFT stop token
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens, # Allow reasonable length
        stop=["<|im_end|>"] # Use chat end token as stop sequence
    )
    print(f"Sampling parameters set: temp={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}, stop={sampling_params.stop}")

    # --- 5. Generate Outputs ---
    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation completed in {generation_time:.2f} seconds.")

    # --- 6. Serialize Results in JSONL format ---
    print(f"Saving results to {args.output_file}...")
    try:
        # Ensure parent directory exists
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w', encoding='utf-8') as f_out:
            for i, output in enumerate(tqdm(outputs)):
                instruction = instructions[i]
                raw_output = output.outputs[0].text.strip()
                # No need to remove trailing ```

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

    # --- Optional: Print throughput ---
    throughput = len(outputs) / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate SimpleSafetyTests Outputs using SFT Model") # Updated description
    parser.add_argument("--model_path", type=str, default="./output/qwen-0.5b-sft-short", # Updated default
                        help="Path to the fine-tuned SFT model directory.")
    parser.add_argument("--dataset_path", type=str, default="data/simple_safety_tests.csv",
                        help="Path to the SimpleSafetyTests CSV file.")
    parser.add_argument("--instruction_key", type=str, default="prompts_final", # Changed default from "prompt"
                        help="Column name in the CSV file that contains the instruction/prompt.")
    # Removed system_prompt_file argument
    parser.add_argument("--output_file", type=str, default="sst_sft_results.jsonl", # Updated default
                        help="Path to save the generated outputs in JSONL format.")
    parser.add_argument("--max_tokens", type=int, default=512, # Keep same as baseline
                        help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of examples for testing (-1 for no limit).")

    args = parser.parse_args()
    main(args) 