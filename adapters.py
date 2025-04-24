from cs336_alignment.sft_dataset import PackedSFTDataset, get_batch_iterator
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

# --- Problem 4.2.1: SFT Data Loading Adapters ---

def get_packed_sft_dataset(
    tokenizer: PreTrainedTokenizer,
    dataset_path: str,
    seq_length: int,
    shuffle: bool
) -> Dataset:
    """Adapter function for test_packed_sft_dataset."""
    return PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=dataset_path,
        seq_length=seq_length,
        shuffle=shuffle
    )

def run_iterate_batches(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool
) -> list[dict[str, list]]:
    """Adapter function for test_iterate_batches.

    Iterates through the dataloader once and returns all batches.
    Note: Converts tensors to lists for easier comparison in tests.
    """
    dataloader = get_batch_iterator(dataset, batch_size, shuffle)
    batches = []
    for batch in dataloader:
        # Convert tensors to lists for easier comparison if needed by tests
        batch_list = {
            key: value.tolist() for key, value in batch.items()
        }
        batches.append(batch_list)
    return batches

# --- [Existing adapter functions if any] ---
# Add the new functions above any existing adapters in the file. 