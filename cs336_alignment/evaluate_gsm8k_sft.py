#!/usr/bin/env python3
import argparse
import json
import os
import re
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

# Assuming parsing_utils.py is in the same directory or accessible in python path
from cs336_alignment.parsing_utils import parse_gsm8k_response

# GSM8K prompt template - Question Only
GSM8K_QUESTION_TEMPLATE = "{question}"

# SFT Chat Template (Ensure this matches the template used during training)
SFT_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def extract_gsm8k_answer(solution: str) -> str | None:
    """Extracts the final numeric answer from the GSM8K solution string."""
    # The final answer is usually after "####"
    match = re.search(r"####\s*([\d,.]+)", solution)
    if match:
        # Remove commas and return
        return match.group(1).replace(",", "")
    return None # Or handle cases where the format might differ

def format_gsm8k_sft_prompt(example: dict) -> str:
    """Formats the prompt for a GSM8K example using the SFT template."""
    # We only need the question for the instruction part in the SFT template
    instruction = GSM8K_QUESTION_TEMPLATE.format(question=example['question'])
    full_prompt = SFT_CHAT_TEMPLATE.format(instruction=instruction)
    return full_prompt

def main(args):
    # --- 1. Load GSM8K Dataset ---
    print(f"Loading GSM8K dataset '{args.gsm8k_dataset_name}', split '{args.gsm8k_split}'...")
    try:
        # Using 'datasets' library. Need 'main' config for gsm8k
        gsm8k_dataset = load_dataset(args.gsm8k_dataset_name, 'main', split=args.gsm8k_split)
        print(f"Loaded {len(gsm8k_dataset)} examples from GSM8K {args.gsm8k_split} split.")
    except Exception as e:
        print(f"Error loading GSM8K dataset: {e}")
        print("Please ensure the 'datasets' library is installed and the dataset name/config is correct.")
        return

    # Limit examples if specified
    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} examples.")
        gsm8k_dataset = gsm8k_dataset.select(range(args.limit))

    # --- 2. Format all prompts --- (Using SFT format)
    print("Formatting prompts using SFT template...")
    prompts = [format_gsm8k_sft_prompt(example) for example in tqdm(gsm8k_dataset)] # Use SFT formatter
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
    # Use sampling parameters suitable for GSM8K (higher temp/top_p for reasoning)
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=args.max_tokens, # Adjust max tokens as needed for GSM8K reasoning + answer
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

    # --- 6. Parse Outputs & Calculate Metrics ---
    print("Parsing outputs and calculating metrics...")
    results = []
    correct_predictions = 0
    parsing_failures = 0

    for i, output in enumerate(tqdm(outputs)):
        example = gsm8k_dataset[i]
        raw_output = output.outputs[0].text.strip()
        parsed_prediction = parse_gsm8k_response(raw_output)

        # Extract ground truth answer from the 'answer' field
        ground_truth_answer = extract_gsm8k_answer(example.get('answer', ''))

        is_correct = None
        if parsed_prediction is None:
            parsing_failures += 1
        elif ground_truth_answer is not None:
            # Compare as strings after canonicalization (removing commas etc.)
            is_correct = (parsed_prediction == ground_truth_answer)
            if is_correct:
                correct_predictions += 1
        else:
             print(f"Warning: Could not extract ground truth answer for example {i}. Skipping.")

        results.append({
            "index": i,
            "question": example.get('question', 'N/A'),
            "ground_truth_solution": example.get('answer', 'N/A'),
            "ground_truth_answer": ground_truth_answer,
            "full_prompt": prompts[i],
            "raw_output": raw_output,
            "parsed_prediction": parsed_prediction,
            "is_correct": is_correct,
        })

    total_examples = len(outputs)
    parsable_examples = total_examples - parsing_failures
    # Accuracy is based on examples where both prediction and ground truth could be parsed
    valid_comparisons = sum(1 for r in results if r['parsed_prediction'] is not None and r['ground_truth_answer'] is not None)
    accuracy = correct_predictions / valid_comparisons if valid_comparisons > 0 else 0.0
    parsing_failure_rate = parsing_failures / total_examples if total_examples > 0 else 0.0

    print("\n--- Evaluation Results ---")
    print(f"Total examples: {total_examples}")
    print(f"Parsing failures (prediction): {parsing_failures} ({parsing_failure_rate:.2%})")
    print(f"Examples with valid ground truth: {sum(1 for r in results if r['ground_truth_answer'] is not None)}")
    print(f"Examples with valid prediction and ground truth: {valid_comparisons}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy (Exact Match on final number): {accuracy:.4f} ({accuracy:.2%})")

    # --- 7. Serialize Results ---
    print(f"Saving results to {args.output_file}...")
    try:
        # Ensure parent directory exists
        Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # --- Optional: Print throughput ---
    throughput = total_examples / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFT Model GSM8K Performance") # Updated description
    parser.add_argument("--model_path", type=str, required=True, # Made required
                        help="Path to the fine-tuned SFT model directory.")
    parser.add_argument("--gsm8k_dataset_name", type=str, default="gsm8k",
                        help="Name of the GSM8K dataset on Hugging Face Hub.")
    parser.add_argument("--gsm8k_split", type=str, default="test",
                        help="Split of the GSM8K dataset to use (e.g., 'test', 'train').")
    parser.add_argument("--output_file", type=str, default="gsm8k_sft_results.json", # Updated default
                        help="Path to save the detailed evaluation results (JSON).")
    parser.add_argument("--max_tokens", type=int, default=350, # Increased default for SFT reasoning
                        help="Maximum number of tokens to generate.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of examples for testing (-1 for no limit).")

    args = parser.parse_args()
    main(args) 