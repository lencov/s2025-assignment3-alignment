#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
from vllm import LLM, SamplingParams

from cs336_alignment.parsing_utils import parse_mmlu_response

# Removed Zero-Shot system prompt file loading

# MMLU specific prompt - adjust if structure differs slightly
MMLU_PROMPT_TEMPLATE = \
"""Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer:"""

SFT_CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"

def format_mmlu_sft_prompt(example: dict) -> str:
    """Formats the full prompt for a given MMLU example using the SFT template."""
    mmlu_specific_prompt = MMLU_PROMPT_TEMPLATE.format(
        subject=example['subject'],
        question=example['question'],
        options=example['choices'] # Assuming 'choices' key holds the list
    )
    # Embed the MMLU prompt within the SFT chat template
    full_prompt = SFT_CHAT_TEMPLATE.format(instruction=mmlu_specific_prompt)
    return full_prompt

def main(args):
    # --- 1. Load System Prompt --- (Removed)
    # print(f"Loading system prompt from: {args.system_prompt_file}")
    # try:
    #     with open(args.system_prompt_file, 'r') as f:
    #         system_prompt = f.read().strip()
    #     print("System prompt loaded successfully.")
    # except FileNotFoundError:
    #     print(f"Error: System prompt file not found at {args.system_prompt_file}")
    #     return
    # except Exception as e:
    #     print(f"Error reading system prompt file: {e}")
    #     return

    # --- 2. Load MMLU Dataset ---
    print(f"Loading MMLU dataset '{args.mmlu_dataset_name}', split '{args.mmlu_split}'...")
    try:
        # Using 'datasets' library. Assumes internet connection or cached data.
        # If data is local in data/*, adjust loading logic (e.g., manual JSON/CSV loading)
        mmlu_dataset = load_dataset(args.mmlu_dataset_name, 'all', split=args.mmlu_split)
        print(f"Loaded {len(mmlu_dataset)} examples from MMLU {args.mmlu_split} split.")
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        print("Please ensure the 'datasets' library is installed and the dataset name is correct.")
        print("If using local data, modify the script to load from the 'data/' directory.")
        return

    # Limit examples if specified (for debugging/testing)
    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} examples.")
        mmlu_dataset = mmlu_dataset.select(range(args.limit))

    # --- 3. Format all prompts ---
    print("Formatting prompts using SFT template...")
    prompts = [format_mmlu_sft_prompt(example) for example in tqdm(mmlu_dataset)] # Use SFT formatter
    print(f"Formatted {len(prompts)} prompts.")

    # --- 4. Initialize vLLM ---
    print(f"Initializing vLLM with SFT model: {args.model_path}") # Updated log message
    try:
        # Adjust tensor_parallel_size based on available GPUs
        llm = LLM(model=args.model_path, trust_remote_code=True, tensor_parallel_size=args.tensor_parallel_size)
        print("vLLM initialized successfully.")
    except Exception as e:
        print(f"Error initializing vLLM: {e}")
        print("Ensure the model path is correct and vLLM is installed properly.")
        return

    # --- 5. Configure Sampling Parameters ---
    # Greedy decoding as specified
    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens, # Max tokens for the generated answer part
        stop=["<|im_end|>"] # Use chat end token as stop sequence
    )
    print(f"Sampling parameters set: temp={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}, stop={sampling_params.stop}")

    # --- 6. Generate Outputs ---
    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation completed in {generation_time:.2f} seconds.")

    # --- 7. Parse Outputs & Calculate Metrics ---
    print("Parsing outputs and calculating metrics...")
    results = []
    correct_predictions = 0
    parsing_failures = 0
    letter_map = ['A', 'B', 'C', 'D'] # To map MMLU answer index to letter

    for i, output in enumerate(tqdm(outputs)):
        example = mmlu_dataset[i]
        raw_output = output.outputs[0].text.strip()
        parsed_prediction = parse_mmlu_response(raw_output)

        # Assuming MMLU 'answer' field is an integer index 0-3
        # If it's already a letter ('A'-'D'), remove the letter_map lookup
        try:
            # Check if 'answer' key exists and is valid index
            if 'answer' in example and isinstance(example['answer'], int) and 0 <= example['answer'] < len(letter_map):
                 ground_truth_letter = letter_map[example['answer']]
            elif 'answer' in example and isinstance(example['answer'], str) and example['answer'] in letter_map:
                 # Handle case where answer is already a letter
                 ground_truth_letter = example['answer']
            else:
                 print(f"Warning: Invalid or missing 'answer' key in example {i}. Skipping metric calculation for this example.")
                 ground_truth_letter = None # Cannot determine correctness

        except Exception as e:
            print(f"Warning: Error processing answer key for example {i}: {e}. Skipping metric calculation.")
            ground_truth_letter = None


        is_correct = None
        if parsed_prediction is None:
            parsing_failures += 1
        elif ground_truth_letter is not None:
            is_correct = (parsed_prediction == ground_truth_letter)
            if is_correct:
                correct_predictions += 1

        results.append({
            "index": i,
            "subject": example.get('subject', 'N/A'),
            "question": example.get('question', 'N/A'),
            "options": example.get('choices', []),
            "ground_truth_answer": ground_truth_letter, # Store the letter
            "full_prompt": prompts[i],
            "raw_output": raw_output,
            "parsed_prediction": parsed_prediction,
            "is_correct": is_correct,
        })

    total_examples = len(outputs)
    parsable_examples = total_examples - parsing_failures
    accuracy = correct_predictions / parsable_examples if parsable_examples > 0 else 0.0
    parsing_failure_rate = parsing_failures / total_examples if total_examples > 0 else 0.0

    print("\n--- Evaluation Results ---")
    print(f"Total examples: {total_examples}")
    print(f"Parsing failures: {parsing_failures} ({parsing_failure_rate:.2%})")
    print(f"Examples used for accuracy: {parsable_examples}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy:.2%})")

    # --- 8. Serialize Results ---
    print(f"Saving results to {args.output_file}...")
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    # --- Optional: Print throughput ---
    throughput = total_examples / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SFT Model MMLU Performance") # Updated description
    parser.add_argument("--model_path", type=str, required=True, # Made required, removed default
                        help="Path to the fine-tuned SFT model directory.")
    parser.add_argument("--mmlu_dataset_name", type=str, default="cais/mmlu",
                        help="Name of the MMLU dataset on Hugging Face Hub or path to local data.")
    parser.add_argument("--mmlu_split", type=str, default="test",
                        help="Split of the MMLU dataset to use (e.g., 'test', 'validation').")
    # parser.add_argument("--system_prompt_file", type=Path, default=ZERO_SHOT_SYSTEM_PROMPT_FILE,
    #                     help="Path to the file containing the zero-shot system prompt.") # Removed system prompt arg
    parser.add_argument("--output_file", type=str, default="mmlu_sft_results.json", # Updated default output file
                        help="Path to save the detailed evaluation results (JSON).")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate for the answer.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism with vLLM.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit the number of examples to evaluate (e.g., for debugging). -1 means no limit.")

    args = parser.parse_args()
    main(args) 