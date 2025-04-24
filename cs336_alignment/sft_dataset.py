import json
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Alpaca template
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)

class PackedSFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning (SFT) that packs multiple examples
    into fixed-length sequences.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_path: str, seq_length: int = 2048, shuffle: bool = True):
        """
        Args:
            tokenizer: The tokenizer to use for encoding text.
            dataset_path: Path to the JSONL dataset file.
            seq_length: The desired fixed sequence length for packed examples.
            shuffle: Whether to shuffle the documents before concatenation.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.eos_token_id = tokenizer.eos_token_id

        if self.eos_token_id is None:
            # Use pad token if eos token is not defined, common in some models
            # Warning: This might not be ideal for all models.
            self.eos_token_id = tokenizer.pad_token_id
            logging.warning(f"Tokenizer does not have an EOS token defined. Using PAD token ID {self.eos_token_id} as EOS.")
            if self.eos_token_id is None:
                 raise ValueError("Tokenizer must have either eos_token_id or pad_token_id defined.")

        logging.info(f"Loading and processing dataset from: {dataset_path}")
        tokenized_chunks = self._load_and_process_data(dataset_path)
        self.tokenized_chunks = tokenized_chunks
        logging.info(f"Dataset initialized with {len(self.tokenized_chunks)} sequences of length {self.seq_length}.")

    def _load_and_process_data(self, dataset_path: str) -> list[list[int]]:
        """Loads data, formats, tokenizes, concatenates, and chunks it."""
        all_examples = []
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(tqdm(f, desc="Reading dataset file")):
                    try:
                        example = json.loads(line.strip())
                        # Assuming keys are 'prompt' and 'response' based on dataset description
                        if 'prompt' in example and 'response' in example:
                             all_examples.append(example)
                        else:
                             logging.warning(f"Skipping line {i+1}: Missing 'prompt' or 'response' key.")
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping line {i+1} due to JSON parsing error.")
        except FileNotFoundError:
            logging.error(f"Dataset file not found at {dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Failed to read or parse dataset file: {e}")
            raise

        logging.info(f"Read {len(all_examples)} valid examples from {dataset_path}")

        if self.shuffle:
            logging.info("Shuffling examples before concatenation.")
            random.shuffle(all_examples)

        all_token_ids = []
        logging.info("Formatting, tokenizing, and concatenating examples...")
        for example in tqdm(all_examples, desc="Processing examples"):
            formatted_text = ALPACA_TEMPLATE.format(
                prompt=example['prompt'],
                response=example['response']
            )
            # Add EOS token to the end of each formatted example
            # Handle cases where tokenizer might automatically add EOS/BOS if not desired
            # We explicitly add EOS here assuming the tokenizer does not add it automatically with add_special_tokens=False
            formatted_text += self.tokenizer.eos_token
            token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=False)
            all_token_ids.extend(token_ids)

        logging.info(f"Total number of tokens after concatenation: {len(all_token_ids)}")

        # Chunk the concatenated tokens
        num_tokens = len(all_token_ids)
        num_sequences = num_tokens // self.seq_length
        logging.info(f"Chunking into {num_sequences} sequences of length {self.seq_length}.")

        tokenized_chunks = []
        for i in range(num_sequences):
            start_idx = i * self.seq_length
            end_idx = start_idx + self.seq_length
            chunk = all_token_ids[start_idx:end_idx]
            tokenized_chunks.append(chunk)

        # Report discarded tokens
        discarded_tokens = num_tokens % self.seq_length
        if discarded_tokens > 0:
            logging.info(f"Discarded {discarded_tokens} tokens from the end of the dataset.")

        return tokenized_chunks

    def __len__(self) -> int:
        """Returns the number of fixed-length sequences in the dataset."""
        return len(self.tokenized_chunks)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """
        Returns the i-th sequence chunk as input_ids and labels.
        For standard language modeling, input_ids and labels are the same.
        """
        if i >= len(self.tokenized_chunks):
            raise IndexError(f"Index {i} out of bounds for dataset with length {len(self)}")

        token_ids = self.tokenized_chunks[i]
        input_ids_tensor = torch.tensor(token_ids, dtype=torch.long)
        # Labels are the same as input_ids for next-token prediction
        labels_tensor = input_ids_tensor.clone()

        return {
            "input_ids": input_ids_tensor,
            "labels": labels_tensor,
        }

def get_batch_iterator(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    """
    Creates a DataLoader for batching examples from the dataset.

    Args:
        dataset: The Dataset instance to batch from.
        batch_size: The number of sequences per batch.
        shuffle: Whether to shuffle the sequences each epoch.

    Returns:
        A PyTorch DataLoader instance.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, # Simple case, adjust if I/O becomes a bottleneck
        pin_memory=False # Adjust if using GPU and it helps
    )

# Example usage (optional, for basic testing)
if __name__ == '__main__':
    from transformers import AutoTokenizer

    # --- Configuration for Example Usage ---
    # Make sure you have downloaded the data using download_sft_data.py
    EXAMPLE_DATASET_PATH = "../data/sft/train.jsonl" # Adjusted relative path
    EXAMPLE_TOKENIZER_PATH = "../Qwen/Qwen2.5-0.5B" # Adjusted relative path
    EXAMPLE_SEQ_LENGTH = 512 # Use a smaller length for quick testing
    EXAMPLE_BATCH_SIZE = 4

    # Check if paths exist relative to the script location
    script_dir = os.path.dirname(__file__)
    abs_dataset_path = os.path.join(script_dir, EXAMPLE_DATASET_PATH)
    abs_tokenizer_path = os.path.join(script_dir, EXAMPLE_TOKENIZER_PATH)

    if not os.path.exists(abs_dataset_path):
         print(f"Example Usage ERROR: Dataset file not found at {abs_dataset_path}")
         print("Please run download_sft_data.py first or adjust EXAMPLE_DATASET_PATH.")
    elif not os.path.exists(abs_tokenizer_path):
         print(f"Example Usage ERROR: Tokenizer not found at {abs_tokenizer_path}")
         print("Please download the Qwen model first or adjust EXAMPLE_TOKENIZER_PATH.")
    else:
        print("\n--- Example Dataset Usage ---")
        tokenizer = AutoTokenizer.from_pretrained(abs_tokenizer_path, trust_remote_code=True)
        # Make sure pad_token is set if eos_token is missing (common for some base models)
        if tokenizer.eos_token is None and tokenizer.pad_token is not None:
             tokenizer.eos_token = tokenizer.pad_token
             print(f"Set tokenizer.eos_token to tokenizer.pad_token ({tokenizer.eos_token})")

        # Create the dataset (don't shuffle for deterministic example)
        sft_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=abs_dataset_path, # Use absolute path
            seq_length=EXAMPLE_SEQ_LENGTH,
            shuffle=False # Keep order for this simple example
        )
        print(f"Dataset length: {len(sft_dataset)}")

        # Get the first item
        if len(sft_dataset) > 0:
            first_item = sft_dataset[0]
            print("\nFirst item shapes:")
            print(f"  input_ids: {first_item['input_ids'].shape}")
            print(f"  labels: {first_item['labels'].shape}")
            # Optionally decode to see the text
            # print("\nDecoded first item (input_ids):")
            # print(tokenizer.decode(first_item['input_ids']))

            # Create a DataLoader
            print("\n--- Example DataLoader Usage ---")
            batch_iterator = get_batch_iterator(
                sft_dataset,
                batch_size=EXAMPLE_BATCH_SIZE,
                shuffle=True # Shuffle batches each epoch
            )

            # Iterate through a few batches
            print(f"Iterating through first 3 batches (Batch Size: {EXAMPLE_BATCH_SIZE}):")
            for i, batch in enumerate(batch_iterator):
                print(f"\nBatch {i+1}:")
                print(f"  input_ids shape: {batch['input_ids'].shape}") # Should be [batch_size, seq_length]
                print(f"  labels shape: {batch['labels'].shape}")     # Should be [batch_size, seq_length]
                if i >= 2: # Show first 3 batches
                    break
            print("\nExample Usage Complete.")
        else:
             print("Dataset created but is empty. Cannot show first item or iterate batches.") 