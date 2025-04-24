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

# --- Remove batch iterator and example usage for now --- 