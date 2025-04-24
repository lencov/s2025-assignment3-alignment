# cs336_alignment/sft_dataset.py
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

# Alpaca template as specified in the handout
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)

class PackedSFTDataset(Dataset):
    """
    Dataset for supervised fine-tuning (SFT) that packs multiple examples
    into fixed-length sequences, correctly handling labels across chunk boundaries.

    Conforms to the interface specified in CS336 Assignment 5, Section 4.2.1.
    """
    def __init__(self, tokenizer: PreTrainedTokenizer, dataset_path: str, seq_length: int = 2048, shuffle: bool = True):
        """
        Constructs the PackedSFTDataset.

        Args:
            tokenizer: The Hugging Face tokenizer.
            dataset_path: Path to the JSONL dataset file (each line is a JSON
                          object with 'prompt' and 'response' keys).
            seq_length: The desired fixed sequence length for packed examples.
            shuffle: Whether to shuffle the documents before concatenation.
        """
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.shuffle = shuffle
        self.eos_token_id = tokenizer.eos_token_id

        if self.eos_token_id is None:
            # Attempt to use pad_token as a fallback if eos_token is not defined
            self.eos_token_id = tokenizer.pad_token_id
            if self.eos_token_id is not None:
                logging.warning(f"Tokenizer does not have an EOS token defined. Using PAD token ID {self.eos_token_id} as EOS delimiter.")
            else:
                 raise ValueError("Tokenizer must have either eos_token_id or pad_token_id defined to use as a delimiter.")

        logging.info(f"Loading and processing dataset from: {dataset_path}")
        # Load, process, and chunk the data during initialization
        self.input_chunks, self.label_chunks = self._load_and_process_data(dataset_path)
        logging.info(f"Dataset initialized with {len(self.input_chunks)} sequences of length {self.seq_length}.")

    def _load_and_process_data(self, dataset_path: str) -> tuple[list[list[int]], list[list[int]]]:
        """
        Internal method to load, format, tokenize, concatenate, create labels,
        and chunk the data.
        """
        all_examples = []
        # --- 1. Load Data ---
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logging.error(f"Dataset file not found at {dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Failed to read dataset file: {e}")
            raise

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

        # --- 2. Shuffle (Optional) ---
        if self.shuffle:
            logging.info("Shuffling examples before concatenation.")
            random.shuffle(all_examples)

        # --- 3. Format, Tokenize, Concatenate ---
        all_token_ids = []
        logging.info("Formatting, tokenizing, and concatenating examples...")
        for example in tqdm(all_examples, desc="Processing examples"):
            formatted_text = ALPACA_TEMPLATE.format(
                prompt=example['prompt'],
                response=example['response']
            )
            # Tokenize *without* adding special tokens automatically
            token_ids = self.tokenizer.encode(formatted_text, add_special_tokens=True)
            # Manually append the EOS token ID as a delimiter between documents
            token_ids.append(self.eos_token_id) # self.eos_token_id should be 128001 for the test
            all_token_ids.extend(token_ids)

        logging.info(f"Total number of tokens after concatenation: {len(all_token_ids)}")

        # --- 4. Create Shifted Input/Label Sequences ---
        # Ensure we have enough tokens for at least one sequence + the subsequent token for the label
        if len(all_token_ids) <= self.seq_length:
             logging.warning(f"Dataset has only {len(all_token_ids)} tokens, which is less than or equal to seq_length ({self.seq_length}). Cannot form any sequences.")
             return [], []

        # Input sequence: t1, t2, ..., t(N-1)
        # Label sequence: t2, t3, ..., tN
        # Where N is the total number of tokens in all_token_ids
        input_sequence = all_token_ids[:-1]
        label_sequence = all_token_ids[1:]
        assert len(input_sequence) == len(label_sequence)

        # --- 5. Chunk Sequences ---
        num_tokens_for_chunking = len(input_sequence) # Length is N-1
        # We drop the last partial chunk
        num_sequences = num_tokens_for_chunking // self.seq_length
        logging.info(f"Chunking into {num_sequences} sequences of length {self.seq_length}.")

        input_chunks = []
        label_chunks = []
        for i in range(num_sequences):
            start_idx = i * self.seq_length
            end_idx = start_idx + self.seq_length
            input_chunks.append(input_sequence[start_idx:end_idx])
            label_chunks.append(label_sequence[start_idx:end_idx])

        # Report discarded tokens (from the end of the N-1 sequence)
        discarded_tokens = num_tokens_for_chunking % self.seq_length
        if discarded_tokens > 0:
            logging.info(f"Discarded {discarded_tokens} tokens from the end of the input/label sequences during chunking.")

        # Sanity checks
        if num_sequences > 0:
            assert all(len(chunk) == self.seq_length for chunk in input_chunks), "Input chunk length mismatch after processing."
            assert all(len(chunk) == self.seq_length for chunk in label_chunks), "Label chunk length mismatch after processing."
            assert len(input_chunks) == len(label_chunks), "Input/Label chunk count mismatch after processing."

        return input_chunks, label_chunks

    def __len__(self) -> int:
        """
        Returns the total number of fixed-length sequences in the dataset.
        """
        return len(self.input_chunks)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        """
        Returns the i-th element of the Dataset.

        Args:
            i: The index of the element to retrieve.

        Returns:
            A dictionary with keys "input_ids" and "labels", containing
            corresponding PyTorch tensors of shape (seq_length,).
        """
        if i >= len(self.input_chunks):
            raise IndexError(f"Index {i} out of bounds for dataset with length {len(self)}")

        # Retrieve the pre-computed chunks
        input_ids = self.input_chunks[i]
        labels = self.label_chunks[i]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

# Implementation of the batch iterator function (outside the class)
def iterate_batches(dataset: PackedSFTDataset, batch_size: int, shuffle: bool = True):
    """
    Yields batches of data from the dataset.

    Args:
        dataset: The PackedSFTDataset instance.
        batch_size: The number of sequences per batch.
        shuffle: Whether to shuffle the sequences within each epoch.

    Yields:
        A dictionary representing a batch, with keys like 'input_ids' and 'labels',
        where values are stacked PyTorch tensors of shape (current_batch_size, seq_length).
        Note: The last batch might have a size smaller than batch_size.
    """
    indices = list(range(len(dataset)))
    if shuffle:
        random.shuffle(indices)

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i : i + batch_size]

        # Safety check for empty batch list (shouldn't happen with range logic)
        if len(batch_indices) == 0:
             continue

        # Collect data for the batch
        batch_data = [dataset[idx] for idx in batch_indices]

        # Collate the batch: Stack tensors for each key
        collated_batch = {}
        if not batch_data: continue # Should not happen

        keys = batch_data[0].keys() # Get keys from the first item ('input_ids', 'labels')
        for key in keys:
            # Ensure all items in batch_data have the key
            if all(key in item for item in batch_data):
                 # Stack tensors along a new batch dimension (dim=0)
                 collated_batch[key] = torch.stack([item[key] for item in batch_data], dim=0)
            else:
                 logging.warning(f"Key '{key}' missing in some items of the batch, skipping collation for this key.")

        # Only yield if we successfully collated something
        if collated_batch:
            yield collated_batch

# --- Example Usage ---
# This block allows testing the script directly
if __name__ == '__main__':
    from transformers import AutoTokenizer

    # --- Configuration ---
    # Define paths relative to the project root (assuming this script is in cs336_alignment)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) # Go up one level

    # Make sure these paths are correct for your setup
    # Using a small dummy dataset for quick testing might be useful
    EXAMPLE_DATASET_PATH = os.path.join(project_root, "data/sft/train.jsonl") # Adjust if needed
    # Use a valid tokenizer path accessible from your environment
    EXAMPLE_TOKENIZER_PATH = os.path.join(project_root, "Qwen/Qwen2.5-0.5B") # Adjust if needed, e.g., "gpt2"

    EXAMPLE_SEQ_LENGTH = 128 # Use a smaller seq_length for faster testing
    EXAMPLE_BATCH_SIZE = 4
    # --- End Configuration ---

    print(f"Attempting to load dataset from: {EXAMPLE_DATASET_PATH}")
    print(f"Attempting to load tokenizer from: {EXAMPLE_TOKENIZER_PATH}")

    if not os.path.exists(EXAMPLE_DATASET_PATH):
         print(f"\nExample Usage ERROR: Dataset file not found at {EXAMPLE_DATASET_PATH}")
         print("Please ensure the path is correct and the data has been downloaded/created.")
    elif not os.path.exists(EXAMPLE_TOKENIZER_PATH):
         print(f"\nExample Usage ERROR: Tokenizer files not found at {EXAMPLE_TOKENIZER_PATH}")
         print("Please ensure the path points to a valid Hugging Face tokenizer directory.")
    else:
        print("\n--- Example Dataset Usage ---")
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(EXAMPLE_TOKENIZER_PATH, trust_remote_code=True)

            # Handle potential missing EOS/PAD tokens
            if tokenizer.eos_token is None:
                if tokenizer.pad_token is not None:
                    tokenizer.eos_token = tokenizer.pad_token
                    print(f"Set tokenizer.eos_token to tokenizer.pad_token ('{tokenizer.eos_token}')")
                else:
                    # Add a default EOS token if none exists. This might require resizing model embeddings if training.
                    new_eos = '<|endoftext|>'
                    print(f"Warning: Tokenizer lacks EOS token. Adding '{new_eos}' as EOS.")
                    tokenizer.add_special_tokens({'eos_token': new_eos})

            if tokenizer.pad_token is None:
                 tokenizer.pad_token = tokenizer.eos_token
                 print(f"Set tokenizer.pad_token to tokenizer.eos_token ('{tokenizer.pad_token}')")

            # Initialize dataset (disable shuffle for predictable first item)
            sft_dataset = PackedSFTDataset(
                tokenizer=tokenizer,
                dataset_path=EXAMPLE_DATASET_PATH,
                seq_length=EXAMPLE_SEQ_LENGTH,
                shuffle=False # Turn shuffle off for predictable output in example
            )
            print(f"\nDataset length: {len(sft_dataset)}")

            # Test dataset item access and batch iteration
            if len(sft_dataset) > 0:
                first_item = sft_dataset[0]
                print("\nFirst item shapes:")
                print(f"  input_ids: {first_item['input_ids'].shape}")
                print(f"  labels:    {first_item['labels'].shape}")

                # Print first few tokens to visually check the shift
                print("\nFirst 10 tokens of first item:")
                print(f"  input_ids: {first_item['input_ids'][:10].tolist()}")
                print(f"  labels:    {first_item['labels'][:10].tolist()}")
                print("  (Labels should be input_ids shifted left by one)")

                print("\n--- Example iterate_batches Usage ---")
                # Use shuffle=True for batch iteration example
                batch_iterator = iterate_batches(
                    sft_dataset,
                    batch_size=EXAMPLE_BATCH_SIZE,
                    shuffle=True
                )

                print(f"\nIterating through first 3 batches (Batch Size: {EXAMPLE_BATCH_SIZE}):")
                batch_count = 0
                for i, batch in enumerate(batch_iterator):
                    batch_count += 1
                    print(f"\nBatch {i+1}:")
                    print(f"  input_ids shape: {batch['input_ids'].shape}")
                    print(f"  labels shape:    {batch['labels'].shape}")
                    if i >= 2: # Stop after showing 3 batches
                        break
                if batch_count == 0:
                    print("\nNo batches were generated (Dataset might be smaller than batch size).")

                print("\nExample Usage Complete.")
            else:
                 print("\nDataset created but contains no sequences (check data and seq_length).")

        except Exception as e:
             print(f"\nAn error occurred during example usage: {e}")
             import traceback
             traceback.print_exc()