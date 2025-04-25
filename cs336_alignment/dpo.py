import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Tuple

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

    # Set pad token if not defined (common for GPT-2 style models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Format sequences using Alpaca template and add EOS token
    chosen_sequence_str = format_alpaca(prompt, response_chosen) + tokenizer.eos_token
    rejected_sequence_str = format_alpaca(prompt, response_rejected) + tokenizer.eos_token

    # 2. Tokenize the sequences
    # We tokenize them together for potential padding efficiency, though for single instances it doesn't matter much.
    # Ensure padding is done correctly if batching in the future.
    # For single instance, we often don't need explicit padding if handled by the model/loss function.
    # Using return_tensors='pt' gives PyTorch tensors.
    # Set padding=True and truncation=True for robust handling, though DPO often uses unpacked sequences.
    # max_length might need adjustment based on model capabilities.
    tokenized_sequences = tokenizer(
        [chosen_sequence_str, rejected_sequence_str],
        return_tensors="pt",
        padding=True, # Pad to the longest sequence in the pair
        truncation=True,
        max_length=tokenizer.model_max_length # Use model's max length
    )

    input_ids = tokenized_sequences.input_ids
    attention_mask = tokenized_sequences.attention_mask

    # Create labels - in standard Causal LM, labels are the same as input_ids
    # Loss function inside the model usually handles the shifting
    # For manual calculation, we might need to handle ignored indices (e.g., padding)
    labels = input_ids.clone()
    # Ensure padding tokens are ignored in loss calculation if necessary (often -100)
    if tokenizer.pad_token_id is not None:
        labels[labels == tokenizer.pad_token_id] = -100

    # Ensure tensors are on the correct devices
    lm_device = next(lm.parameters()).device
    ref_device = next(lm_ref.parameters()).device

    input_ids_lm = input_ids.to(lm_device)
    attention_mask_lm = attention_mask.to(lm_device)
    labels_lm = labels.to(lm_device)

    input_ids_ref = input_ids.to(ref_device)
    attention_mask_ref = attention_mask.to(ref_device)
    labels_ref = labels.to(ref_device) # Labels might also be needed if ref model computes loss internally

    # 3. Compute log probabilities using the helper function
    # Log probs shape: (batch_size,) where batch_size is 2 here (chosen, rejected)
    log_probs_lm = get_sequence_log_probs(lm, input_ids_lm, labels_lm, attention_mask_lm)
    log_probs_ref = get_sequence_log_probs(lm_ref, input_ids_ref, labels_ref, attention_mask_ref)

    # Separate chosen and rejected log probabilities
    # Ensure ref log probs are moved to the main model's device for comparison
    log_probs_lm_chosen = log_probs_lm[0]
    log_probs_lm_rejected = log_probs_lm[1]
    log_probs_ref_chosen = log_probs_ref[0].to(lm_device)
    log_probs_ref_rejected = log_probs_ref[1].to(lm_device)

    # 4. Calculate log-prob differences
    # pi_log_prob_diff = log P_pi(chosen | prompt) - log P_pi(rejected | prompt)
    # ref_log_prob_diff = log P_ref(chosen | prompt) - log P_ref(rejected | prompt)
    # Leveraging the hint: log P(y|x) = log P(x,y) - log P(x)
    # So, log P(yw|x) - log P(yl|x) = log P(x,yw) - log P(x,yl)
    pi_log_prob_diff = log_probs_lm_chosen - log_probs_lm_rejected
    ref_log_prob_diff = log_probs_ref_chosen - log_probs_ref_rejected

    # 5. Compute the DPO loss (Equation 3)
    # loss = -log_sigmoid(beta * (pi_log_prob_diff - ref_log_prob_diff))
    logits = beta * (pi_log_prob_diff - ref_log_prob_diff)
    loss = -F.logsigmoid(logits)

    return loss 