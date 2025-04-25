#!/usr/bin/env python3
import gzip
import json
import os
import requests # Added for downloading
from typing import List, Dict, Any
import random # Added for sampling in analysis

# --- Constants ---
HUGGINGFACE_HUB_URL = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/"
HH_FILES = [
    "harmless-base.jsonl.gz",
    "helpful-base.jsonl.gz",
    "helpful-online.jsonl.gz",
    "helpful-rejection-sampled.jsonl.gz",
]

def download_file(url: str, destination: str):
    """Downloads a file from a URL to a destination, handling potential errors."""
    print(f"  Downloading {os.path.basename(destination)} from {url}...")
    try:
        response = requests.get(url, stream=True, timeout=30) # Added timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Ensure destination directory exists
        os.makedirs(os.path.dirname(destination), exist_ok=True)

        with open(destination, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"  Successfully downloaded to {destination}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"  Error downloading {url}: {e}")
        # Attempt to remove partially downloaded file if it exists
        if os.path.exists(destination):
            try:
                os.remove(destination)
                print(f"  Removed partially downloaded file: {destination}")
            except OSError as remove_e:
                print(f"  Error removing partially downloaded file {destination}: {remove_e}")
        return False
    except Exception as e:
        print(f"  An unexpected error occurred during download: {e}")
        # Attempt removal here too
        if os.path.exists(destination):
             try:
                 os.remove(destination)
                 print(f"  Removed partially downloaded file: {destination}")
             except OSError as remove_e:
                 print(f"  Error removing partially downloaded file {destination}: {remove_e}")
        return False

def load_hh_dataset(data_dir: str, max_examples_per_file: int = -1) -> List[Dict[str, Any]]:
    """
    Loads and processes the Anthropic HH-RLHF dataset from specified files.
    Downloads files from Hugging Face Hub if they are not found locally.

    Args:
        data_dir: The directory containing the .jsonl.gz files
                  (e.g., '/home/shared/hh' or './data/hh').
        max_examples_per_file: Maximum number of examples to load per file
                               (-1 for unlimited). Useful for quick testing.

    Returns:
        A list of dictionaries, where each dictionary represents a single-turn
        preference example with keys: 'instruction', 'chosen', 'rejected',
        and 'source_file'.
    """
    dataset = []

    print(f"Loading HH dataset from {data_dir}...")
    # Ensure the target data directory exists or can be created
    try:
        os.makedirs(data_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create data directory {data_dir}: {e}")
        return [] # Cannot proceed without data directory

    for filename in HH_FILES:
        filepath = os.path.join(data_dir, filename)

        # Check if file exists, download if not
        if not os.path.exists(filepath):
            print(f"File not found locally: {filepath}")
            download_url = HUGGINGFACE_HUB_URL + filename
            if not download_file(download_url, filepath):
                print(f"Skipping {filename} due to download failure.")
                continue # Skip this file if download failed
        # Proceed if file exists locally or was just downloaded successfully
        print(f"Processing {filename}...")
        source_file_short = filename.replace(".jsonl.gz", "")
        loaded_count = 0

        try:
            with gzip.open(filepath, "rt", encoding="utf-8") as f:
                for line in f:
                    if max_examples_per_file > 0 and loaded_count >= max_examples_per_file:
                        print(f"  Reached max_examples_per_file ({max_examples_per_file}) for {filename}.")
                        break # Stop reading this file

                    try:
                        data = json.loads(line)
                        chosen_conversation = data.get("chosen")
                        rejected_conversation = data.get("rejected")

                        if not chosen_conversation or not rejected_conversation:
                            continue # Skip if essential data is missing

                        # Simple check for single-turn: Human speaks once, Assistant responds once.
                        # Counting turns more robustly might involve parsing the '\n\nHuman:' structure.
                        # For simplicity, we check if the 'chosen' conversation contains '\n\nHuman:' more than once.
                        # And also check if the first turn is indeed Human.
                        # We also need to ensure Assistant responds only once in the chosen/rejected text.
                        is_single_turn_chosen = (
                            chosen_conversation.strip().startswith("Human:") and
                            chosen_conversation.count("\n\nHuman:") == 1 and
                            chosen_conversation.count("\n\nAssistant:") == 1
                        )
                        is_single_turn_rejected = (
                            rejected_conversation.strip().startswith("Human:") and
                            rejected_conversation.count("\n\nHuman:") == 1 and
                            rejected_conversation.count("\n\nAssistant:") == 1
                        )

                        if is_single_turn_chosen and is_single_turn_rejected:

                            # Extract first human message (instruction)
                            instruction_end_index = chosen_conversation.find("\n\nAssistant:")
                            if instruction_end_index == -1:
                                continue # Should not happen if count check passed, but safety first

                            instruction = chosen_conversation[len("Human:"):instruction_end_index].strip()

                            # Extract first assistant response (chosen)
                            chosen_response_start_index = instruction_end_index + len("\n\nAssistant:")
                            # Find the end of the first assistant turn (either EOF or next Human turn)
                            chosen_response_end_index = chosen_conversation.find("\n\nHuman:", chosen_response_start_index)
                            chosen_response = chosen_conversation[chosen_response_start_index : chosen_response_end_index if chosen_response_end_index != -1 else None].strip()


                            # Extract first assistant response (rejected) - instruction should be identical
                            rejected_instruction_end_index = rejected_conversation.find("\n\nAssistant:")
                            if rejected_instruction_end_index == -1:
                                 continue # Sanity check
                            rejected_response_start_index = rejected_instruction_end_index + len("\n\nAssistant:")
                            rejected_response_end_index = rejected_conversation.find("\n\nHuman:", rejected_response_start_index)
                            rejected_response = rejected_conversation[rejected_response_start_index : rejected_response_end_index if rejected_response_end_index != -1 else None].strip()

                            # Ensure responses are not empty
                            if instruction and chosen_response and rejected_response:
                                 dataset.append({
                                    "instruction": instruction,
                                    "chosen": chosen_response,
                                    "rejected": rejected_response,
                                    "source_file": source_file_short,
                                })
                                 loaded_count += 1

                    except json.JSONDecodeError:
                        # print(f"Warning: Skipping malformed JSON line in {filename}")
                        pass # Reduce verbosity
                    except Exception as e:
                        print(f"Warning: Error processing line in {filename}: {e}")
        except gzip.BadGzipFile:
             print(f"Error: Bad gzip file at {filepath}. It might be corrupted or not a valid gzip file.")
        except Exception as e:
            print(f"Error opening or reading {filepath}: {e}")


    print(f"Finished loading. Total single-turn examples processed: {len(dataset)}")
    return dataset

def analyze_hh_samples(dataset: List[Dict[str, Any]], num_samples: int = 3):
    """Analyzes random samples from the loaded HH dataset."""
    if not dataset:
        print("Dataset is empty. Cannot analyze samples.")
        return

    helpful_sources = ["helpful-base", "helpful-online", "helpful-rejection-sampled"]
    harmless_sources = ["harmless-base"]

    helpful_samples = [d for d in dataset if d['source_file'] in helpful_sources]
    harmless_samples = [d for d in dataset if d['source_file'] in harmless_sources]

    print(f"\n--- Analyzing {num_samples} Random 'Helpful' Samples ---")
    if len(helpful_samples) >= num_samples:
        for i, sample in enumerate(random.sample(helpful_samples, num_samples)):
            print(f"\nHelpful Sample {i+1} (from {sample['source_file']}):")
            print(f"  Instruction: {sample['instruction'][:200]}...") # Truncate for readability
            print(f"  Chosen:      {sample['chosen'][:200]}...")
            print(f"  Rejected:    {sample['rejected'][:200]}...")
            # Add analysis comments here based on full text if needed
    else:
        print(f"  Not enough helpful samples found (found {len(helpful_samples)}).")

    print(f"\n--- Analyzing {num_samples} Random 'Harmless' Samples ---")
    if len(harmless_samples) >= num_samples:
        for i, sample in enumerate(random.sample(harmless_samples, num_samples)):
            print(f"\nHarmless Sample {i+1} (from {sample['source_file']}):")
            print(f"  Instruction: {sample['instruction'][:200]}...")
            print(f"  Chosen:      {sample['chosen'][:200]}...")
            print(f"  Rejected:    {sample['rejected'][:200]}...")
            # Add analysis comments here based on full text if needed
    else:
         print(f"  Not enough harmless samples found (found {len(harmless_samples)}).")


if __name__ == "__main__":
    # Example usage: Assumes data is in a subdirectory './data/hh'
    # Adjust this path as needed.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_data_dir = os.path.join(os.path.dirname(current_dir), 'data', 'hh') # Assumes script is in scripts/, data in ../data/

    # --- Configuration ---
    hh_data_directory = os.environ.get("HH_DATA_DIR", default_data_dir) # Use environment variable or default
    examples_per_file_limit = int(os.environ.get("HH_MAX_EXAMPLES", 100)) # Load only 100 examples per file for faster testing, set -1 for all

    print(f"Using data directory: {hh_data_directory}")
    print(f"Max examples per file: {examples_per_file_limit if examples_per_file_limit > 0 else 'Unlimited'}")

    # --- Load Data ---
    loaded_dataset = load_hh_dataset(hh_data_directory, max_examples_per_file=examples_per_file_limit)

    # --- Analyze Samples ---
    if loaded_dataset:
        analyze_hh_samples(loaded_dataset, num_samples=3)
        print(f"\nTotal examples loaded: {len(loaded_dataset)}")
        source_counts = {}
        for item in loaded_dataset:
            source = item['source_file']
            source_counts[source] = source_counts.get(source, 0) + 1
        print("Examples per source file:")
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
    else:
        print("\nDataset loading failed or resulted in an empty dataset.")
        print(f"Please check the path ({hh_data_directory}) and ensure network connectivity if downloads are needed.") 