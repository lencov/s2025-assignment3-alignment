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

ZERO_SHOT_SYSTEM_PROMPT_FILE = Path("cs336_alignment/prompts/zero_shot_system_prompt.prompt")

MMLU_PROMPT_TEMPLATE = \
"""Answer the following multiple choice question about {subject}. Respond with a single sentence of the form "The correct answer is _", filling the blank with the letter corresponding to the correct answer (i.e., A, B, C or D).

Question: {question}
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}
Answer:"""

def format_mmlu_prompt(system_prompt: str, example: dict) -> str:
    mmlu_specific_prompt = MMLU_PROMPT_TEMPLATE.format(
        subject=example['subject'],
        question=example['question'],
        options=example['choices']
    )
    full_prompt = system_prompt.replace("{instruction}", mmlu_specific_prompt)
    return full_prompt

def main(args):
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

    print(f"Loading MMLU dataset '{args.mmlu_dataset_name}', split '{args.mmlu_split}'...")
    try:
        mmlu_dataset = load_dataset(args.mmlu_dataset_name, 'all', split=args.mmlu_split)
        print(f"Loaded {len(mmlu_dataset)} examples from MMLU {args.mmlu_split} split.")
    except Exception as e:
        print(f"Error loading MMLU dataset: {e}")
        print("Please ensure the 'datasets' library is installed and the dataset name is correct.")
        print("If using local data, modify the script to load from the 'data/' directory.")
        return

    if args.limit > 0:
        print(f"Limiting evaluation to {args.limit} examples.")
        mmlu_dataset = mmlu_dataset.select(range(args.limit))

    print("Formatting prompts...")
    prompts = [format_mmlu_prompt(system_prompt, example) for example in tqdm(mmlu_dataset)]
    print(f"Formatted {len(prompts)} prompts.")

    print(f"Initializing vLLM with model: {args.model_path}")
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
        print("Ensure the model path is correct and vLLM is installed properly.")
        return

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["# Query:"]
    )
    print(f"Sampling parameters set: temp={sampling_params.temperature}, top_p={sampling_params.top_p}, max_tokens={sampling_params.max_tokens}, stop={sampling_params.stop}")

    print(f"Generating responses for {len(prompts)} prompts...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    generation_time = end_time - start_time
    print(f"Generation completed in {generation_time:.2f} seconds.")

    print("Parsing outputs and calculating metrics...")
    results = []
    correct_predictions = 0
    parsing_failures = 0
    letter_map = ['A', 'B', 'C', 'D']

    for i, output in enumerate(tqdm(outputs)):
        example = mmlu_dataset[i]
        raw_output = output.outputs[0].text.strip()
        parsed_prediction = parse_mmlu_response(raw_output)

        try:
            if 'answer' in example and isinstance(example['answer'], int) and 0 <= example['answer'] < len(letter_map):
                 ground_truth_letter = letter_map[example['answer']]
            elif 'answer' in example and isinstance(example['answer'], str) and example['answer'] in letter_map:
                 ground_truth_letter = example['answer']
            else:
                 print(f"Warning: Invalid or missing 'answer' key in example {i}. Skipping metric calculation for this example.")
                 ground_truth_letter = None

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
            "ground_truth_answer": ground_truth_letter,
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

    print(f"Saving results to {args.output_file}...")
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print("Results saved successfully.")
    except Exception as e:
        print(f"Error saving results to JSON: {e}")

    throughput = total_examples / generation_time if generation_time > 0 else 0
    print(f"Throughput: {throughput:.2f} examples/second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Zero-Shot MMLU Performance")
    parser.add_argument("--model_path", type=str, default="../../../Qwen/Qwen2.5-0.5B",
                        help="Path to the HuggingFace model directory (downloaded locally).")
    parser.add_argument("--mmlu_dataset_name", type=str, default="cais/mmlu",
                        help="Name of the MMLU dataset on Hugging Face Hub or path to local data.")
    parser.add_argument("--mmlu_split", type=str, default="test",
                        help="Split of the MMLU dataset to use (e.g., 'test', 'validation').")
    parser.add_argument("--system_prompt_file", type=Path, default=ZERO_SHOT_SYSTEM_PROMPT_FILE,
                        help="Path to the file containing the zero-shot system prompt.")
    parser.add_argument("--output_file", type=str, default="mmlu_baseline_results.json",
                        help="Path to save the detailed evaluation results (JSON).")
    parser.add_argument("--max_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate for the answer.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Number of GPUs to use for tensor parallelism with vLLM.")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit the number of examples to evaluate (e.g., for debugging). -1 means no limit.")

    args = parser.parse_args()
    main(args) 