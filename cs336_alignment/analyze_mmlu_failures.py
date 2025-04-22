#!/usr/bin/env python3
import argparse
import json

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

    print("\n--- Parsing Failures ---")
    failure_count = 0
    for i, result in enumerate(results_data):
        if result.get("parsed_prediction") is None:
            failure_count += 1
            print(f"\nExample Index: {result.get('index', i)}")
            print(f"Raw Output:\n---\n{result.get('raw_output', 'N/A')}\n---")

    if failure_count == 0:
        print("No parsing failures found.")
    else:
        print(f"\nFound {failure_count} parsing failures.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MMLU evaluation results for parsing failures.")
    parser.add_argument("results_file", type=str,
                        help="Path to the JSON file containing MMLU evaluation results.")
    args = parser.parse_args()
    main(args) 