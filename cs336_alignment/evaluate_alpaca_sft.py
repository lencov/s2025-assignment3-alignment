#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# SFT Chat Template (Ensure this matches the template used during training)
SFT_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

# Function to format the AlpacaEval prompt using the SFT chat template
def format_alpaca_sft_prompt(instruction: str) -> str:
    """Formats the full prompt for a given AlpacaEval instruction using the SFT template."""
    return SFT_CHAT_TEMPLATE.format(instruction=instruction)

def main(args):
    # --- 1. Load AlpacaEval Dataset --- (System prompt loading removed)
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

    # --- 2. Format all prompts --- (Using SFT format)
    print("Formatting prompts using SFT template...")
    instructions = [example['instruction'] for example in alpaca_eval_dataset]
    prompts = [format_alpaca_sft_prompt(instruction) for instruction in tqdm(instructions)] # Use SFT formatter
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
    # Greedy decoding for AlpacaEval, use SFT stop token
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens, # Allow for potentially long answers
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

    # --- 6. Prepare results for AlpacaEval format ---
    print("Formatting results for AlpacaEval...")
    alpaca_formatted_results = []
    # Get the SFT model name for the 'generator' field
    generator_name = args.generator_name # Use the provided default or argument

    for i, output in enumerate(tqdm(outputs)):
        example = alpaca_eval_dataset[i]
        raw_output = output.outputs[0].text.strip()
        # No need to remove trailing ``` as the SFT format doesn't use it

        alpaca_formatted_results.append({
            "instruction": example.get('instruction', 'N/A'),
            "output": raw_output,
            "generator": generator_name,
            "dataset": example.get('dataset', 'alpaca_eval') # Use dataset field from example if available
        })

    # --- 7. Serialize Results ---
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
    parser = argparse.ArgumentParser(description="Generate AlpacaEval Outputs using SFT Model") # Updated description
    parser.add_argument("--model_path", type=str, default="./output/qwen-0.5b-sft-short", # Updated default
                        help="Path to the fine-tuned SFT model directory.")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca_eval",
                        help="Name of the AlpacaEval dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_split", type=str, default="eval",
                        help="Split of the AlpacaEval dataset to use (usually 'eval').")
    # Removed system_prompt_file argument
    parser.add_argument("--output_file", type=str, default="alpaca_eval_sft_results.json", # Updated default
                        help="Path to save the generated outputs in AlpacaEval JSON format.")
    parser.add_argument("--generator_name", type=str, default="qwen-0.5b-sft-short", # Updated default
                        help="Identifier for the SFT model used for generation.")
    parser.add_argument("--max_tokens", type=int, default=1024, # Keep same as baseline
                        help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of examples for testing (-1 for no limit).")

    args = parser.parse_args()
    main(args) 