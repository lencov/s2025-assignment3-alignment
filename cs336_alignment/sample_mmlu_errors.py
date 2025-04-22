#!/usr/bin/env python3
import argparse
import json
import random

def main(args):
    print(f"Loading results from: {args.results_file}")
    try:
        with open(args.results_file, 'r') as f:
            results_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {args.results_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {args.results_file}")
        return
    except Exception as e:
        print(f"Error reading results file: {e}")
        return

    incorrect_predictions = [
        result for result in results_data
        if result.get("is_correct") is False # Explicitly check for False
    ]

    num_incorrect = len(incorrect_predictions)
    num_samples = min(num_incorrect, args.num_samples)

    print(f"\nFound {num_incorrect} incorrect predictions.")
    if num_incorrect == 0:
        print("No incorrect predictions to sample.")
        return

    print(f"Sampling {num_samples} incorrect predictions for analysis...")

    # Sample without replacement
    sampled_errors = random.sample(incorrect_predictions, num_samples)

    print("\n--- Sampled Incorrect Predictions ---")
    for i, error in enumerate(sampled_errors):
        print(f"\nSample {i+1} (Index: {error.get('index', 'N/A')})")
        print(f"Subject: {error.get('subject', 'N/A')}")
        print(f"Question: {error.get('question', 'N/A')}")
        options_str = "\n".join([f"  {chr(ord('A') + j)}. {opt}" for j, opt in enumerate(error.get('options', []))])
        print(f"Options:\n{options_str}")
        print(f"Ground Truth Answer: {error.get('ground_truth_answer', 'N/A')}")
        print(f"Parsed Prediction: {error.get('parsed_prediction', 'N/A')}")
        # Optionally print raw output too
        # print(f"Raw Output: {error.get('raw_output', 'N/A')}")
        print("---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample incorrect MMLU predictions for error analysis.")
    parser.add_argument("results_file", type=str,
                        help="Path to the JSON file containing MMLU evaluation results.")
    parser.add_argument("-n", "--num_samples", type=int, default=10,
                        help="Number of incorrect predictions to sample.")
    args = parser.parse_args()
    main(args) 