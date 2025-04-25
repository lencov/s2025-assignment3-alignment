import gzip
import json
import os
import logging
from tqdm import tqdm
import random

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'hh') # Use absolute path for robustness

DOWNLOADED_FILES_MAP = {
    "train.jsonl": "harmless-base",
    "train1.jsonl": "helpful-base",
    "train2.jsonl": "helpful-online",
    "train3.jsonl": "helpful-rejection-sampled",
}

def parse_conversation(conversation_text):
    """Parses the conversation string into a list of {'speaker': speaker, 'text': text} dicts."""
    turns = []
    segments = conversation_text.strip().split('\n\n')
    for segment in segments:
        if segment.startswith("Human:"):
            speaker = "Human"
            text = segment[len("Human:"):].strip()
        elif segment.startswith("Assistant:"):
            speaker = "Assistant"
            text = segment[len("Assistant:"):].strip()
        else:
            logging.debug(f"Skipping unrecognized segment: {segment[:50]}...")
            continue
        if text: 
            turns.append({"speaker": speaker, "text": text})
    return turns


def load_processed_hh_dataset(data_dir=DEFAULT_DATA_DIR, file_map=DOWNLOADED_FILES_MAP, max_examples_per_file=None):
    """
    Loads the pre-downloaded Anthropic HH dataset files.
    Filters for single-turn conversations and extracts instruction, chosen, and rejected responses.

    Args:
        data_dir (str): The directory where the data files are located.
        file_map (dict): A dictionary mapping actual filenames to conceptual source names.
        max_examples_per_file (int, optional): Maximum examples to load per file. Defaults to None (load all).

    Returns:
        list: A list of dictionaries, each containing:
              'instruction' (str): The initial human prompt.
              'chosen' (str): The chosen assistant response.
              'rejected' (str): The rejected assistant response.
              'source_file' (str): The conceptual source file name (e.g., "harmless-base").
    """
    logging.info(f"Using data directory: {data_dir}")
    if max_examples_per_file:
        logging.info(f"Max examples per file: {max_examples_per_file}")

    all_examples = []

    logging.info(f"Loading HH dataset from pre-downloaded files in {data_dir}...")

    for filename, source_name in file_map.items():
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}. Skipping.")
            continue
        else:
            logging.info(f"Processing file: {file_path} (Source: {source_name})")

        processed_count = 0
        skipped_multi_turn = 0
        skipped_parsing_error = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {filename}"):
                    if max_examples_per_file and processed_count >= max_examples_per_file:
                        break
                    try:
                        data = json.loads(line)

                        chosen_conversation = parse_conversation(data['chosen'])
                        rejected_conversation = parse_conversation(data['rejected'])

                        is_single_turn = (
                            len(chosen_conversation) == 2 and
                            chosen_conversation[0]['speaker'] == 'Human' and
                            chosen_conversation[1]['speaker'] == 'Assistant' and
                            len(rejected_conversation) == 2 and
                            rejected_conversation[0]['speaker'] == 'Human' and
                            rejected_conversation[1]['speaker'] == 'Assistant' and
                            chosen_conversation[0]['text'] == rejected_conversation[0]['text'] # Ensure prompts match
                        )

                        if is_single_turn:
                            instruction = chosen_conversation[0]['text']
                            chosen_response = chosen_conversation[1]['text']
                            rejected_response = rejected_conversation[1]['text']

                            all_examples.append({
                                "instruction": instruction,
                                "chosen": chosen_response,
                                "rejected": rejected_response,
                                "source_file": source_name
                            })
                            processed_count += 1
                        else:
                            skipped_multi_turn += 1

                    except json.JSONDecodeError:
                        skipped_parsing_error += 1
                        logging.debug(f"Skipping line due to JSON decode error in {filename}")
                    except KeyError as e:
                         skipped_parsing_error += 1
                         logging.debug(f"Skipping line due to missing key {e} in {filename}")
                    except Exception as e_inner:
                         skipped_parsing_error += 1
                         logging.warning(f"Skipping line due to unexpected error: {e_inner} in {filename}")


            logging.info(f"Finished processing {filename}. Added {processed_count} examples. "
                         f"Skipped {skipped_multi_turn} (multi-turn/mismatched), Skipped {skipped_parsing_error} (parse error).")

        except FileNotFoundError:
             logging.error(f"File disappeared during processing: {file_path}. Skipping.")
        except Exception as e_outer:
             logging.error(f"An error occurred processing file {file_path}: {e_outer}. Skipping.")


    logging.info(f"Finished loading all files. Total single-turn examples processed: {len(all_examples)}")
    return all_examples


if __name__ == "__main__":
    loaded_data = load_processed_hh_dataset(max_examples_per_file=None) # Load all

    if not loaded_data:
        print("\nDataset loading failed or resulted in an empty dataset.")
        print(f"Please check the path ({DEFAULT_DATA_DIR}) and ensure the files {list(DOWNLOADED_FILES_MAP.keys())} exist.")
    else:
        print(f"\nSuccessfully loaded {len(loaded_data)} single-turn examples.")

        print("\nAnalysis for Problem 6.2.2:")

        helpful_examples = [ex for ex in loaded_data if 'helpful' in ex['source_file']]
        harmless_examples = [ex for ex in loaded_data if 'harmless' in ex['source_file']]

        print(f"\nFound {len(helpful_examples)} 'helpful' source examples and {len(harmless_examples)} 'harmless' source examples.")

        num_to_sample = 3

        if len(helpful_examples) >= num_to_sample:
            print("\n--- Random 'Helpful' Source Examples ---")
            random.seed(42) # for reproducibility
            # Sample 3 random examples from the 'helpful' sources
            sampled_helpful = random.sample(helpful_examples, num_to_sample)
            for i, example in enumerate(sampled_helpful):
                 print(f"\nHelpful Example {i+1} (from {example['source_file']}):")
                 print(f"Instruction:\n{example['instruction']}")
                 print("-" * 20)
                 print(f"Chosen Response:\n{example['chosen']}")
                 print("-" * 20)
                 print(f"Rejected Response:\n{example['rejected']}")
                 print("-" * 20)
        else:
            print(f"\nNot enough 'helpful' examples to sample {num_to_sample}.")


        if len(harmless_examples) >= num_to_sample:
            print("\n--- Random 'Harmless' Source Examples ---")
            random.seed(42) 
            sampled_harmless = random.sample(harmless_examples, num_to_sample)
            for i, example in enumerate(sampled_harmless):
                 print(f"\nHarmless Example {i+1} (from {example['source_file']}):")
                 print(f"Instruction:\n{example['instruction']}")
                 print("-" * 20)
                 print(f"Chosen Response:\n{example['chosen']}")
                 print("-" * 20)
                 print(f"Rejected Response:\n{example['rejected']}")
                 print("-" * 20)
        else:
             print(f"\nNot enough 'harmless' examples to sample {num_to_sample}.")