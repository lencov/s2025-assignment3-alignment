#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Define prompt templates (adjust paths if necessary)
ZERO_SHOT_SYSTEM_PROMPT_FILE = Path("cs336_alignment/prompts/zero_shot_system_prompt.prompt")

# Function to format the AlpacaEval prompt using the system prompt
def format_alpaca_prompt(system_prompt: str, instruction: str) -> str:
    """Formats the full prompt for a given AlpacaEval instruction using the system prompt."""
    # Embed the instruction within the general system prompt structure
    # Assumes the system prompt has a placeholder like "{instruction}"
    if "{instruction}" not in system_prompt:
        # Fallback or error if placeholder is missing
        # For simplicity, appending instruction to the system prompt if placeholder is missing
        # A better approach might be to ensure the system prompt file is correct.
        print("Warning: '{instruction}' placeholder not found in system prompt. Appending instruction.")
        return f"{system_prompt}\n\n# Query:\n```{instruction}```\n\n# Answer:\n```"
    else:
        # Use the placeholder
        # Ensure the instruction is wrapped correctly if the system prompt expects it (e.g., in ```)
        # The provided system prompt expects ```{instruction}```
        return system_prompt.replace("{instruction}", instruction)

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

    # --- 2. Load AlpacaEval Dataset ---
    print(f"Loading AlpacaEval dataset '{args.dataset_name}' split '{args.dataset_split}'...")
    try:
        # Load the evaluation set
        alpaca_eval_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        print(f"Loaded {len(alpaca_eval_dataset)} examples from AlpacaEval {args.dataset_split} split.")
    except Exception as e:
        print(f"Error loading AlpacaEval dataset: {e}")
        print("Please ensure the 'datasets' library is installed and the dataset name is correct.")
        return

    # Limit examples if specified
    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} examples.")
        alpaca_eval_dataset = alpaca_eval_dataset.select(range(args.limit))

    # --- 3. Format all prompts ---
    print("Formatting prompts...")
    instructions = [example['instruction'] for example in alpaca_eval_dataset]
    prompts = [format_alpaca_prompt(system_prompt, instruction) for instruction in tqdm(instructions)]
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
        max_tokens=args.max_tokens, # Allow for potentially long answers
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

    # --- 7. Prepare results for AlpacaEval format ---
    print("Formatting results for AlpacaEval...")
    alpaca_formatted_results = []
    # Get the base model name for the 'generator' field
    generator_name = args.generator_name or Path(args.model_path).name

    for i, output in enumerate(tqdm(outputs)):
        example = alpaca_eval_dataset[i]
        raw_output = output.outputs[0].text.strip()
        # Remove potential trailing ``` if the model adds it based on the prompt format
        if raw_output.endswith("```"):
            raw_output = raw_output[:-3].strip()

        alpaca_formatted_results.append({
            "instruction": example.get('instruction', 'N/A'),
            "output": raw_output,
            "generator": generator_name,
            "dataset": example.get('dataset', 'alpaca_eval') # Use dataset field from example if available
        })

    # --- 8. Serialize Results ---
    print(f"Saving results to {args.output_file}...")
    try:
        # Ensure parent directory exists
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            # Save as a JSON array (list of dicts)
            json.dump(alpaca_formatted_results, f, indent=2)
        print("Results saved successfully in AlpacaEval format.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # --- Optional: Print throughput ---
    throughput = len(outputs) / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AlpacaEval Zero-Shot Outputs")
    parser.add_argument("--model_path", type=str, default="/content/s2025-assignment3-alignment/Qwen/Qwen2.5-0.5B",
                        help="Path to the HuggingFace model directory.")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca_eval",
                        help="Name of the AlpacaEval dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_split", type=str, default="eval",
                        help="Split of the AlpacaEval dataset to use (usually 'eval').")
    parser.add_argument("--system_prompt_file", type=Path, default=ZERO_SHOT_SYSTEM_PROMPT_FILE,
                        help="Path to the file containing the zero-shot system prompt.")
    parser.add_argument("--output_file", type=str, default="alpaca_eval_baseline_results.json",
                        help="Path to save the generated outputs in AlpacaEval JSON format.")
    parser.add_argument("--generator_name", type=str, default="Qwen2.5-0.5B-baseline",
                        help="Identifier for the model used for generation.")
    parser.add_argument("--max_tokens", type=int, default=1024, # Allow longer outputs for diverse instructions
                        help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of examples for testing (-1 for no limit).")

    args = parser.parse_args()
    main(args) 