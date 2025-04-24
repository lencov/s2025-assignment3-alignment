import json
import random
import os

# --- Configuration ---
TARGET_DIR = "data/sft"
JSONL_FILENAME = "train.jsonl"
JSONL_FILEPATH = os.path.join(TARGET_DIR, JSONL_FILENAME)
NUM_SAMPLES_TO_SHOW = 10

# --- Main Analysis Logic ---
def analyze_data(filepath, num_samples):
    """Reads a JSONL file, samples random lines, and prints them."""
    print(f"Analyzing data from: {filepath}")
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}. Please run the download script first.")
        return

    all_examples = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                try:
                    example = json.loads(line.strip())
                    all_examples.append(example)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {i+1} due to JSON parsing error: {e}")
                except Exception as e:
                    print(f"Warning: Skipping line {i+1} due to unexpected error: {e}")

        print(f"Successfully read {len(all_examples)} examples.")

        if len(all_examples) < num_samples:
            print(f"Warning: Requested {num_samples} samples, but only found {len(all_examples)}. Showing all.")
            samples_to_show = all_examples
        else:
            samples_to_show = random.sample(all_examples, num_samples)

        print(f"\n--- Random {len(samples_to_show)} Samples ---")
        for i, sample in enumerate(samples_to_show):
            print(f"\n--- Sample {i+1} ---")
            # --- Attempt to identify standard keys --- 
            # Common keys are 'prompt'/'instruction'/'input' and 'response'/'output'/'completion'
            prompt_keys = ['prompt', 'instruction', 'input', 'query']
            response_keys = ['response', 'output', 'completion', 'answer']
            
            prompt_text = "<Prompt key not found>"
            for key in prompt_keys:
                if key in sample:
                    prompt_text = sample[key]
                    break
            
            response_text = "<Response key not found>"
            for key in response_keys:
                if key in sample:
                    response_text = sample[key]
                    break
            
            print(f"Prompt: {prompt_text}\n")
            print(f"Response: {response_text}\n")
            # print(f"Full Sample: {sample}") # Uncomment to see all keys
            print("-" * 20)

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please run the download script first.")
    except Exception as e:
        print(f"An unexpected error occurred while reading or processing the file: {e}")

if __name__ == "__main__":
    analyze_data(JSONL_FILEPATH, NUM_SAMPLES_TO_SHOW) 