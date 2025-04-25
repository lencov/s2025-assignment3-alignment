#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

SFT_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def format_alpaca_sft_prompt(instruction: str) -> str:
    return SFT_CHAT_TEMPLATE.format(instruction=instruction)

def main(args):
    print(f"Loading AlpacaEval dataset '{args.dataset_name}' split '{args.dataset_split}'...")
    try:
        alpaca_eval_dataset = load_dataset(args.dataset_name, split=args.dataset_split)
        print(f"Loaded {len(alpaca_eval_dataset)} examples from AlpacaEval {args.dataset_split} split.")
    except Exception as e:
        print(f"Error loading AlpacaEval dataset: {e}")
        print("Please ensure the 'datasets' library is installed and the dataset name is correct.")
        return

    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} examples.")
        alpaca_eval_dataset = alpaca_eval_dataset.select(range(args.limit))

    print("Formatting prompts using SFT template...")
    instructions = [example['instruction'] for example in alpaca_eval_dataset]
    prompts = [format_alpaca_sft_prompt(instruction) for instruction in tqdm(instructions)]
    print(f"Formatted {len(prompts)} prompts.")

    print(f"Initializing vLLM with SFT model: {args.model_path}")
    try:
        llm = LLM(
            model=args.model_path,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype='half'
        )
        print("vLLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        return

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"]
    )
    print(f"Sampling parameters set: temp={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}, stop={sampling_params.stop}")

    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation completed in {generation_time:.2f} seconds.")

    print("Formatting results for AlpacaEval...")
    alpaca_formatted_results = []

    generator_name = args.generator_name

    for i, output in enumerate(tqdm(outputs)):
        example = alpaca_eval_dataset[i]
        raw_output = output.outputs[0].text.strip()

        alpaca_formatted_results.append({
            "instruction": example.get('instruction', 'N/A'),
            "output": raw_output,
            "generator": generator_name,
            "dataset": example.get('dataset', 'alpaca_eval')
        })

    print(f"Saving results to {args.output_file}...")
    try:
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(alpaca_formatted_results, f, indent=2)
        print("Results saved successfully in AlpacaEval format.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    throughput = len(outputs) / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AlpacaEval Outputs using SFT Model")
    parser.add_argument("--model_path", type=str, default="./output/qwen-0.5b-sft-short",
                        help="Path to the fine-tuned SFT model directory.")
    parser.add_argument("--dataset_name", type=str, default="tatsu-lab/alpaca_eval",
                        help="Name of the AlpacaEval dataset on Hugging Face Hub.")
    parser.add_argument("--dataset_split", type=str, default="eval",
                        help="Split of the AlpacaEval dataset to use (usually 'eval').")

    parser.add_argument("--output_file", type=str, default="alpaca_eval_sft_results.json",
                        help="Path to save the generated outputs in AlpacaEval JSON format.")
    parser.add_argument("--generator_name", type=str, default="qwen-0.5b-sft-short",
                        help="Identifier for the SFT model used for generation.")
    parser.add_argument("--max_tokens", type=int, default=1024,
                        help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of examples for testing (-1 for no limit).")

    args = parser.parse_args()
    main(args) 