import json
import random
import os
import torch
from torch.utils.data import Dataset
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
        # Ensure EOS token ID is available
        self.eos_token_id = tokenizer.eos_token_id
        if self.eos_token_id is None:
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
                # Read all lines first to allow shuffling before processing
                lines = f.readlines()
        except FileNotFoundError:
            logging.error(f"Dataset file not found at {dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Failed to read dataset file: {e}")
            raise

        # Parse JSON lines
        logging.info(f"Parsing {len(lines)} lines from {dataset_path}")
        for i, line in enumerate(tqdm(lines, desc="Parsing JSON lines")):
             try:
                 example = json.loads(line.strip())
                 if 'prompt' in example and 'response' in example:
                     all_examples.append(example)
                 else:
                     logging.warning(f"Skipping line {i+1}: Missing 'prompt' or 'response' key.")
             except json.JSONDecodeError:
                 logging.warning(f"Skipping line {i+1} due to JSON parsing error.")

        logging.info(f"Successfully parsed {len(all_examples)} valid examples.")

        if self.shuffle:
            logging.info("Shuffling examples before concatenation.")
            random.shuffle(all_examples)

        all_token_ids = []
        logging.info("Formatting, tokenizing, and concatenating examples...")
        # Important: Use the EOS *token string* from the tokenizer for concatenation
        eos_token_str = self.tokenizer.eos_token
        if eos_token_str is None and self.tokenizer.pad_token is not None:
            eos_token_str = self.tokenizer.pad_token # Fallback if EOS not defined
            logging.warning(f"Using PAD token string '{eos_token_str}' as EOS delimiter.")
        elif eos_token_str is None:
             raise ValueError("Tokenizer must have either eos_token or pad_token defined to use as a delimiter.")

        for example in tqdm(all_examples, desc="Processing examples"):
            formatted_text = ALPACA_TEMPLATE.format(
                prompt=example['prompt'],
                response=example['response']
            )
            # Concatenate with EOS token *string* as delimiter between examples
            full_doc = formatted_text + eos_token_str
            token_ids = self.tokenizer.encode(full_doc, add_special_tokens=False)
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

# Renamed function and implemented manual batching
def iterate_batches(dataset: PackedSFTDataset, batch_size: int, shuffle: bool = True):
    """
    Yields batches of data from the dataset.

    Args:
        dataset: The PackedSFTDataset instance.
        batch_size: The number of sequences per batch.
        shuffle: Whether to shuffle the sequences each epoch.

    Yields:
        A dictionary representing a batch, with keys like 'input_ids' and 'labels',
        where values are stacked tensors.
    """
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]
        # Skip last batch if it's smaller than batch_size (optional, could also pad)
        if len(batch_indices) < batch_size and i > 0: # Ensure we don't skip if total size < batch_size
             logging.info(f"Skipping last batch with size {len(batch_indices)} (smaller than batch_size {batch_size}).")
             continue
        elif len(batch_indices) == 0:
             continue # Should not happen with range logic, but safety check

        # Collect data for the batch
        batch_data = [dataset[idx] for idx in batch_indices]

        # Collate the batch: Stack tensors for each key
        collated_batch = {}
        if not batch_data: continue # Should not happen if checks above are correct

        keys = batch_data[0].keys()
        for key in keys:
            # Ensure all items in batch_data have the key (should be true)
            if all(key in item for item in batch_data):
                 # Stack tensors along a new batch dimension (dim=0)
                 collated_batch[key] = torch.stack([item[key] for item in batch_data], dim=0)
            else:
                 logging.warning(f"Key '{key}' missing in some items of the batch, skipping collation for this key.")

        if collated_batch: # Only yield if we successfully collated something
            yield collated_batch

# --- Example Usage Update --- 
# (Keep the example usage part if it was present, or add it back if needed for testing)
# Ensure the example usage calls iterate_batches correctly.
if __name__ == '__main__':
    from transformers import AutoTokenizer

    EXAMPLE_DATASET_PATH = "../data/sft/train.jsonl"
    EXAMPLE_TOKENIZER_PATH = "../Qwen/Qwen2.5-0.5B"
    EXAMPLE_SEQ_LENGTH = 512
    EXAMPLE_BATCH_SIZE = 4

    script_dir = os.path.dirname(__file__)
    abs_dataset_path = os.path.join(script_dir, EXAMPLE_DATASET_PATH)
    abs_tokenizer_path = os.path.join(script_dir, EXAMPLE_TOKENIZER_PATH)

    if not os.path.exists(abs_dataset_path):
         print(f"Example Usage ERROR: Dataset file not found at {abs_dataset_path}")
    elif not os.path.exists(abs_tokenizer_path):
         print(f"Example Usage ERROR: Tokenizer not found at {abs_tokenizer_path}")
    else:
        print("\n--- Example Dataset Usage ---")
        tokenizer = AutoTokenizer.from_pretrained(abs_tokenizer_path, trust_remote_code=True)
        if tokenizer.eos_token is None and tokenizer.pad_token is not None:
             tokenizer.eos_token = tokenizer.pad_token
             print(f"Set tokenizer.eos_token to tokenizer.pad_token ({tokenizer.eos_token})")

        sft_dataset = PackedSFTDataset(
            tokenizer=tokenizer,
            dataset_path=abs_dataset_path,
            seq_length=EXAMPLE_SEQ_LENGTH,
            shuffle=False
        )
        print(f"Dataset length: {len(sft_dataset)}")

        if len(sft_dataset) > 0:
            first_item = sft_dataset[0]
            print("\nFirst item shapes:")
            print(f"  input_ids: {first_item['input_ids'].shape}")
            print(f"  labels: {first_item['labels'].shape}")

            print("\n--- Example iterate_batches Usage ---")
            batch_iterator = iterate_batches( # Changed function call
                sft_dataset,
                batch_size=EXAMPLE_BATCH_SIZE,
                shuffle=True
            )

            print(f"Iterating through first 3 batches (Batch Size: {EXAMPLE_BATCH_SIZE}):")
            for i, batch in enumerate(batch_iterator):
                print(f"\nBatch {i+1}:")
                print(f"  input_ids shape: {batch['input_ids'].shape}")
                print(f"  labels shape: {batch['labels'].shape}")
                if i >= 2:
                    break
            print("\nExample Usage Complete.")
        else:
             print("Dataset created but is empty.") 