import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Tuple, List, Dict
import os
import json
import random
import logging
from tqdm import tqdm

# Setup basic logging if not already done
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Alpaca template parts
PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n"
)

def format_alpaca(prompt: str, response: str) -> str:
    """Formats prompt and response using the Alpaca template."""
    return PROMPT_TEMPLATE.format(prompt=prompt) + response

def get_sequence_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor, # Shape (batch_size, seq_len)
    labels: torch.Tensor,    # Shape (batch_size, seq_len)
    attention_mask: torch.Tensor # Shape (batch_size, seq_len)
) -> torch.Tensor: # Shape (batch_size,)
    """
    Computes the log probability of each sequence in the batch.
    This is log P(sequence) = sum_{i} log P(token_i | token_0, ..., token_{i-1})
    """
    with torch.no_grad(): # Ensure no gradients are computed during this calculation
        # Get logits from the model
        # Logits shape: (batch_size, seq_len, vocab_size)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Shift logits and labels for next token prediction
        # We want to predict token i using tokens 0 to i-1
        # Logits shape: (batch_size, seq_len - 1, vocab_size)
        # Labels shape: (batch_size, seq_len - 1)
        shifted_logits = logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()

        # Calculate the log probabilities using log_softmax
        # Log probs shape: (batch_size, seq_len - 1, vocab_size)
        log_probs = F.log_softmax(shifted_logits, dim=-1)

        # Create mask for valid (non -100) labels BEFORE gathering
        mask = (shifted_labels != -100)

        # Clone shifted_labels for indexing, clamp invalid indices to 0 (doesn't matter due to mask)
        # This prevents the gather operation from failing on -100
        gather_labels = shifted_labels.clone()
        gather_labels[~mask] = 0 # Set indices for masked positions to 0 (valid index)

        # Gather the log probabilities using the clamped labels
        per_token_log_probs = torch.gather(log_probs, -1, gather_labels.unsqueeze(-1)).squeeze(-1)

        # Apply the original mask AFTER gathering to zero out log probs for padded tokens
        masked_log_probs = per_token_log_probs * mask

        # Sum the log probabilities for each sequence
        sequence_log_probs = masked_log_probs.sum(dim=-1)

    return sequence_log_probs

def compute_per_instance_dpo_loss(
    lm: PreTrainedModel,
    lm_ref: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    response_chosen: str,
    response_rejected: str,
) -> torch.Tensor:
    """
    Computes the per-instance DPO loss as described in Problem 6.3.

    Args:
        lm: The language model being trained.
        lm_ref: The reference language model.
        tokenizer: The tokenizer for both models.
        beta: The DPO beta hyperparameter.
        prompt: The instruction/prompt string.
        response_chosen: The preferred response string.
        response_rejected: The rejected response string.

    Returns:
        A scalar tensor containing the DPO loss for this instance.
    """

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Format sequences
    chosen_sequence_str = format_alpaca(prompt, response_chosen) + tokenizer.eos_token
    rejected_sequence_str = format_alpaca(prompt, response_rejected) + tokenizer.eos_token

    # Tokenize prompt separately to get its length - REMOVED based on hint
    # prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
    # prompt_length = prompt_tokens.input_ids.shape[1]

    # 2. Tokenize the full sequences
    tokenized_sequences = tokenizer(
        [chosen_sequence_str, rejected_sequence_str],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=tokenizer.model_max_length
    )

    input_ids = tokenized_sequences.input_ids
    attention_mask = tokenized_sequences.attention_mask

    # 3. Create labels, masking out ONLY padding tokens (following the hint)
    labels = input_ids.clone()
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100
    # REMOVED prompt masking: labels[:, :prompt_length] = -100

    # 4. Ensure tensors are on the correct devices
    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    input_ids_lm = input_ids.to(lm_device)
    attention_mask_lm = attention_mask.to(lm_device)
    labels_lm = labels.to(lm_device) # Use labels without prompt masking

    input_ids_ref = input_ids.to(ref_device)
    attention_mask_ref = attention_mask.to(ref_device)
    labels_ref = labels.to(ref_device) # Use labels without prompt masking

    # 5. Compute sequence log probabilities for the FULL sequence
    log_probs_lm = get_sequence_log_probs(lm, input_ids_lm, labels_lm, attention_mask_lm)
    log_probs_ref = get_sequence_log_probs(lm_ref, input_ids_ref, labels_ref, attention_mask_ref)

    # Separate chosen and rejected log probabilities
    log_probs_lm_chosen = log_probs_lm[0]
    log_probs_lm_rejected = log_probs_lm[1]
    log_probs_ref_chosen = log_probs_ref[0].to(lm_device)
    log_probs_ref_rejected = log_probs_ref[1].to(lm_device)

    # 6. Calculate log-prob differences (prompt part cancels out)
    pi_log_prob_diff = log_probs_lm_chosen - log_probs_lm_rejected
    ref_log_prob_diff = log_probs_ref_chosen - log_probs_ref_rejected

    # 7. Compute the DPO loss (Equation 3)
    logits = beta * (pi_log_prob_diff - ref_log_prob_diff)
    loss = -F.logsigmoid(logits)

    return loss 

PROMPT_FORMAT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)

def get_sequence_logprob(
    policy: PreTrainedModel,
    input_ids: torch.Tensor, # Shape (1, seq_len)
    labels: torch.Tensor     # Shape (1, seq_len-1)
) -> torch.Tensor: # Shape (1,)
    """Helper function to compute the log probability of a sequence given a policy model (Simplified)."""
    # Ensure model is on the same device as input_ids
    input_ids = input_ids.to(policy.device)
    labels = labels.to(policy.device)

    # Get logits, shape (1, seq_len, vocab_size)
    logits = policy(input_ids=input_ids).logits # Pass input_ids directly

    # Shift logits and labels for next token prediction
    # Logits shape: (1, seq_len - 1, vocab_size)
    # Labels shape: (1, seq_len - 1)
    shifted_logits = logits[:, :-1, :].contiguous()
    # Labels are already shifted (passed in as labels[:, 1:])

    # Calculate the log probabilities using log_softmax
    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)

    # Gather the log probabilities of the actual next tokens
    # Log probs shape: (1, seq_len - 1)
    # No need for -100 handling as padding is not introduced by tokenizer.encode
    per_token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

    # Sum the log probabilities for the sequence
    # sequence_log_probs shape: (1,)
    sequence_log_prob = per_token_log_probs.sum(-1)
    return sequence_log_prob

def dpo_loss(
    policy_model: PreTrainedModel,
    reference_model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    beta: float,
    prompt: str,
    chosen_response: str,
    rejected_response: str
) -> torch.Tensor:
    """Computes the DPO loss for a single instance, matching the provided example structure."""

    # Set pad token if not defined (GPT-2 needs this for eos_token_id usage)
    if tokenizer.pad_token_id is None:
         if tokenizer.eos_token_id is not None:
             # Common practice for models like GPT-2
             tokenizer.pad_token_id = tokenizer.eos_token_id
         else:
             # Fallback if EOS is also missing (shouldn't happen with typical models)
             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             logging.warning("Tokenizer missing pad and eos token. Added '[PAD]' as pad token.")

    # Ensure EOS token ID is available
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        # Attempt to add EOS if missing, though this is less common
        tokenizer.add_special_tokens({'eos_token': '</s>'})
        eos_token_id = tokenizer.eos_token_id
        logging.warning("Tokenizer missing eos token. Added '</s>' as eos token.")

    # 1. Format and tokenize sequences using tokenizer.encode()
    chosen_sequence_str = PROMPT_FORMAT.format(prompt=prompt, response=chosen_response)
    rejected_sequence_str = PROMPT_FORMAT.format(prompt=prompt, response=rejected_response)

    chosen_input_ids = torch.tensor(tokenizer.encode(chosen_sequence_str) + [eos_token_id], dtype=torch.long).unsqueeze(0)
    rejected_input_ids = torch.tensor(tokenizer.encode(rejected_sequence_str) + [eos_token_id], dtype=torch.long).unsqueeze(0)

    # Create labels by shifting
    chosen_labels = chosen_input_ids[:, 1:].clone()
    rejected_labels = rejected_input_ids[:, 1:].clone()

    # 2. Compute log probabilities
    with torch.no_grad():
        # Reference model calculations (frozen, no gradients)
        ref_chosen_logprob = get_sequence_logprob(reference_model, chosen_input_ids, chosen_labels)
        ref_rejected_logprob = get_sequence_logprob(reference_model, rejected_input_ids, rejected_labels)

    # Policy model calculations (gradients enabled)
    policy_chosen_logprob = get_sequence_logprob(policy_model, chosen_input_ids, chosen_labels)
    policy_rejected_logprob = get_sequence_logprob(policy_model, rejected_input_ids, rejected_labels)

    # Ensure calculations are on the policy model's device
    policy_device = policy_model.device
    ref_chosen_logprob = ref_chosen_logprob.to(policy_device)
    ref_rejected_logprob = ref_rejected_logprob.to(policy_device)

    # 3. Calculate DPO loss terms
    pi_logratios = policy_chosen_logprob - policy_rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob

    logits = pi_logratios - ref_logratios
    # Note: The example had `pi_chosen - pi_rejected + pi_ref_rejected - pi_ref_chosen`
    # which is equivalent to `(pi_chosen - pi_rejected) - (pi_ref_chosen - pi_ref_rejected)`
    # i.e., `pi_logratios - ref_logratios`
    loss = -torch.nn.functional.logsigmoid(beta * logits).mean() # Use mean() for consistency, result is scalar here

    return loss

def get_dpo_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase, # Tokenizer needed for parsing prompt
    val_set_size: int = 200,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """
    Loads and processes the HH dataset from specified files for DPO training.
    Uses the user's specific file names and assumes they are unzipped.
    (Implementation from previous step, assuming it's correct for your data)
    """
    # Map the actual downloaded filenames to their conceptual source
    file_map = {
        "train.jsonl": "harmless-base",
        "train1.jsonl": "helpful-base",
        "train2.jsonl": "helpful-online",
        "train3.jsonl": "helpful-rejection-sampled",
    }
    filenames = list(file_map.keys())

    all_examples = []
    logging.info(f"Loading DPO dataset from: {data_dir}")

    for filename in filenames:
        source_name = file_map[filename]
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}. Skipping.")
            continue

        logging.info(f"Processing file: {file_path} (Source: {source_name})")
        processed_count = 0
        skipped_parsing_error = 0

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {filename}"):
                    try:
                        data = json.loads(line)
                        prompt = data["chosen"].split("\n\nAssistant:", maxsplit=1)[0]
                        if prompt.startswith("\n\nHuman: "):
                            prompt = prompt[len("\n\nHuman: "):].strip()
                        else:
                             logging.debug(f"Prompt format issue in {filename}: {prompt[:50]}...")
                             continue

                        chosen_parts = data["chosen"].split("\n\nAssistant: ", maxsplit=1)
                        if len(chosen_parts) < 2: skipped_parsing_error += 1; continue
                        chosen_response = chosen_parts[1].split("\n\nHuman: ", maxsplit=1)[0].strip()

                        rejected_parts = data["rejected"].split("\n\nAssistant: ", maxsplit=1)
                        if len(rejected_parts) < 2: skipped_parsing_error += 1; continue
                        rejected_response = rejected_parts[1].split("\n\nHuman: ", maxsplit=1)[0].strip()

                        if not prompt or not chosen_response or not rejected_response:
                             skipped_parsing_error += 1; continue

                        all_examples.append({
                            "prompt": prompt,
                            "chosen": chosen_response,
                            "rejected": rejected_response,
                            "source_file": source_name
                        })
                        processed_count += 1
                    except json.JSONDecodeError:
                        skipped_parsing_error += 1
                    except Exception as e_inner:
                        skipped_parsing_error += 1
                        logging.debug(f"Skipping line error: {e_inner} in {filename}")
            logging.info(f"Finished {filename}. Added {processed_count}. Skipped {skipped_parsing_error}.")
        except Exception as e_outer:
             logging.error(f"Error processing file {file_path}: {e_outer}")

    logging.info(f"Finished loading. Total examples: {len(all_examples)}")
    random.seed(seed)
    random.shuffle(all_examples)

    if val_set_size >= len(all_examples) or val_set_size < 0:
        logging.warning(f"Invalid val_set_size ({val_set_size}). Using 0 validation examples.")
        val_set_size = 0

    val_dataset = all_examples[:val_set_size]
    train_dataset = all_examples[val_set_size:]
    logging.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation examples.")
    return train_dataset, val_dataset 