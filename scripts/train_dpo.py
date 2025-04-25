import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase, BitsAndBytesConfig
from tqdm import tqdm
import time
import os
import argparse
import logging
import random
import wandb

# Assuming these utility functions exist in your project structure
from cs336_alignment.dpo import compute_per_instance_dpo_loss, get_sequence_log_probs, format_alpaca
from scripts.load_hh_data import load_processed_hh_dataset # Use the HH loading script

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- DPO Training Setup ---

def get_optimizer(model: PreTrainedModel, learning_rate: float, weight_decay: float) -> optim.Optimizer:
    """Sets up the RMSprop optimizer as recommended."""
    # Note: RMSprop doesn't typically use weight decay in the same way AdamW does.
    # We apply it here as requested, but effectiveness might vary.
    # Consider if specific parameter groups need different handling (e.g., no decay for biases/norms)
    # For simplicity, applying to all parameters for now.
    return optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99, weight_decay=weight_decay)

@torch.no_grad()
def evaluate_dpo(model: PreTrainedModel, model_ref: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
                 val_dataset: Dataset, beta: float, eval_batch_size: int, device: torch.device) -> Tuple[float, float]:
    """Evaluates the model on the validation set for DPO loss and implicit reward accuracy."""
    model.eval()
    model_ref.eval() # Ref model should already be in eval mode, but good practice

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    # Simple batching for validation
    val_loader = DataLoader(val_dataset, batch_size=eval_batch_size)

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        batch_losses = []
        batch_pi_log_probs_chosen = []
        batch_pi_log_probs_rejected = []

        # Process each item individually in the batch (as DPO loss is per-instance)
        # This is inefficient but matches the single-instance loss function structure
        # A batched implementation of compute_per_instance_dpo_loss would be faster
        for i in range(len(batch["instruction"])):
            prompt = batch["instruction"][i]
            chosen = batch["chosen"][i]
            rejected = batch["rejected"][i]

            loss = compute_per_instance_dpo_loss(
                lm=model, # Both models are on the same device now
                lm_ref=model_ref,
                tokenizer=tokenizer,
                beta=beta,
                prompt=prompt,
                response_chosen=chosen,
                response_rejected=rejected,
            )
            batch_losses.append(loss.item())

            # --- Calculate implicit reward accuracy ---
            # We need the log_probs for the trained model (lm)
            # Reuse formatting and tokenization logic from the loss function
            chosen_sequence_str = format_alpaca(prompt, chosen) + tokenizer.eos_token
            rejected_sequence_str = format_alpaca(prompt, rejected) + tokenizer.eos_token
            tokenized_sequences = tokenizer(
                [chosen_sequence_str, rejected_sequence_str],
                return_tensors="pt", padding=True, truncation=True,
                max_length=tokenizer.model_max_length
            ).to(device)

            labels = tokenized_sequences.input_ids.clone()
            if tokenizer.pad_token_id is not None:
                labels[labels == tokenizer.pad_token_id] = -100

            # Reverted: Explicitly calculate prompt length only once outside the loop if needed
            # or ensure the label masking logic here matches compute_per_instance_dpo_loss if it's necessary
            # For now, assume get_sequence_log_probs handles labels correctly (which it should with the mask)
            # IMPORTANT: Mask prompt tokens for accurate response-only logprob calculation
            # Re-tokenize prompt here for length calculation (inefficient but direct)
            prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            prompt_length = prompt_tokens.input_ids.shape[1]
            labels[:, :prompt_length] = -100 # Apply prompt mask

            # Use imported get_sequence_log_probs
            log_probs_lm = get_sequence_log_probs(model, tokenized_sequences.input_ids, labels, tokenized_sequences.attention_mask)
            log_probs_lm_chosen = log_probs_lm[0]
            log_probs_lm_rejected = log_probs_lm[1]

            batch_pi_log_probs_chosen.append(log_probs_lm_chosen.item())
            batch_pi_log_probs_rejected.append(log_probs_lm_rejected.item())

        avg_batch_loss = sum(batch_losses) / len(batch_losses)
        total_loss += avg_batch_loss * len(batch_losses) # Accumulate total loss correctly

        # Count correct predictions for accuracy
        for pi_chosen, pi_rejected in zip(batch_pi_log_probs_chosen, batch_pi_log_probs_rejected):
            if pi_chosen > pi_rejected:
                correct_predictions += 1
        total_predictions += len(batch_losses)

        pbar.set_postfix({"avg_loss": avg_batch_loss})

    avg_val_loss = total_loss / total_predictions
    val_accuracy = correct_predictions / total_predictions

    model.train() # Set model back to train mode
    return avg_val_loss, val_accuracy

# --- Main Training Function ---

def train_dpo(
    sft_model_path: str,
    output_dir: str,
    hh_dataset_dir: str,
    # Hyperparameters
    beta: float = 0.1,
    learning_rate: float = 1e-6,
    weight_decay: float = 0.0,
    gradient_accumulation_steps: int = 2, # Adjust based on memory and desired effective batch size
    train_batch_size: int = 1, # Actual batch size per forward/backward pass
    eval_batch_size: int = 4, # Batch size for validation
    max_train_time_minutes: int = 30,
    val_set_size: int = 200,
    eval_interval_seconds: int = 120, # Evaluate every 2 minutes
    save_interval_seconds: int = 300, # Consider saving checkpoint periodically
    seed: int = 42,
    # Technical details
    device_str: str = "cuda",
    dtype_str: str = "bfloat16", # Use bfloat16 for A100
    use_flash_attn: bool = True,
    # Logging
    wandb_project: str | None = "cs336_alignment_dpo",
    wandb_run_name: str | None = None
):
    """Main DPO training loop."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Setup Device and Dtype ---
    if not torch.cuda.is_available() and device_str == "cuda":
        logging.warning("CUDA not available, falling back to CPU. Training will be very slow.")
        device_str = "cpu"
    device = torch.device(device_str)

    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_str]
    logging.info(f"Using device: {device}, dtype: {pt_dtype}")

    # --- Load Model and Tokenizer ---
    logging.info(f"Loading SFT model and tokenizer from: {sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(sft_model_path)
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": pt_dtype,
        "use_cache": False, # Important for training
    }
    if use_flash_attn and dtype_str != "float32":
        # FlashAttention is usually fastest with bf16/fp16
        logging.info("Using Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif use_flash_attn:
         logging.warning("Flash Attention requested but dtype is float32. Using default attention.")

    # Load the main model (will be trained)
    model = AutoModelForCausalLM.from_pretrained(sft_model_path, **model_kwargs).to(device)

    # Load the reference model (frozen)
    # Use a separate instance to avoid modifying weights during training
    model_ref = AutoModelForCausalLM.from_pretrained(sft_model_path, **model_kwargs).to(device)
    model_ref.eval() # Set to evaluation mode
    for param in model_ref.parameters():
        param.requires_grad = False # Freeze parameters
    logging.info("Loaded main model and frozen reference model.")

    # --- Load Dataset ---
    logging.info(f"Loading HH dataset from: {hh_dataset_dir}")
    # Use the function from the script we prepared
    # Make sure the file_map inside load_hh_data corresponds to your file names
    full_dataset = load_processed_hh_dataset(data_dir=hh_dataset_dir)
    if not full_dataset:
        raise ValueError(f"Failed to load dataset from {hh_dataset_dir}. Check paths and file names.")

    # --- Split Dataset ---
    # Shuffle indices before splitting
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    val_indices = indices[:val_set_size]
    train_indices = indices[val_set_size:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    logging.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation examples.")

    # --- Setup Optimizer ---
    optimizer = get_optimizer(model, learning_rate, weight_decay)

    # --- Initialize Logging (Optional: WandB) ---
    if wandb_project:
        try:
            wandb.init(project=wandb_project, name=wandb_run_name, config=locals())
            logging.info("Weights & Biases initialized.")
            use_wandb = True
        except ImportError:
            logging.warning("wandb not installed, skipping Weights & Biases logging.")
            use_wandb = False
    else:
        use_wandb = False

    # --- Training Loop ---
    model.train()
    start_time = time.time()
    last_eval_time = start_time
    last_save_time = start_time
    total_steps = 0
    processed_examples = 0
    best_val_accuracy = -1.0
    os.makedirs(output_dir, exist_ok=True)

    # Use DataLoader for shuffling and potentially parallel loading later
    # Note: Custom collate might be needed if examples vary greatly in length and packing is desired
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

    training_complete = False
    optimizer.zero_grad() # Initialize gradients

    effective_batch_size = train_batch_size * gradient_accumulation_steps
    logging.info(f"Starting DPO training. Effective batch size: {effective_batch_size}")
    logging.info(f"Training for a maximum of {max_train_time_minutes} minutes.")

    while not training_complete:
        epoch_iterator = tqdm(train_loader, desc=f"Training Step {total_steps}")
        for step, batch in enumerate(epoch_iterator):
            # Check training time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > max_train_time_minutes * 60:
                logging.info(f"Training time limit ({max_train_time_minutes} minutes) reached.")
                training_complete = True
                break

            # Process each item individually (matches single-instance loss function)
            step_loss = 0.0
            actual_batch_size = 0 # Handle potential last smaller batch
            for i in range(len(batch["instruction"])):
                prompt = batch["instruction"][i]
                chosen = batch["chosen"][i]
                rejected = batch["rejected"][i]
                actual_batch_size += 1

                loss = compute_per_instance_dpo_loss(
                    lm=model,
                    lm_ref=model_ref,
                    tokenizer=tokenizer,
                    beta=beta,
                    prompt=prompt,
                    response_chosen=chosen,
                    response_rejected=rejected,
                )

                # Normalize loss for accumulation
                loss = loss / gradient_accumulation_steps
                step_loss += loss.item() # Accumulate item loss for logging

                # Backward pass
                loss.backward()

            # --- Gradient Accumulation Step ---
            if (processed_examples + actual_batch_size) % effective_batch_size == 0:
                # Clip gradients (optional but recommended)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad() # Reset gradients
                total_steps += 1
                epoch_iterator.set_description(f"Training Step {total_steps}")

                avg_step_loss = step_loss * gradient_accumulation_steps / actual_batch_size # Average loss for the step
                epoch_iterator.set_postfix({"loss": avg_step_loss})
                if use_wandb:
                    wandb.log({"train/loss": avg_step_loss, "step": total_steps, "examples": processed_examples + actual_batch_size})

            processed_examples += actual_batch_size

            # --- Periodic Evaluation ---
            current_time = time.time()
            if current_time - last_eval_time >= eval_interval_seconds:
                logging.info(f"Running validation at step {total_steps}...")
                avg_val_loss, val_accuracy = evaluate_dpo(
                    model, model_ref, tokenizer, val_dataset, beta, eval_batch_size, device
                )
                logging.info(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                if use_wandb:
                    wandb.log({"val/loss": avg_val_loss, "val/accuracy": val_accuracy, "step": total_steps})

                # --- Save Best Model --- based on validation accuracy
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    logging.info(f"New best validation accuracy: {best_val_accuracy:.4f}. Saving model...")
                    save_path = os.path.join(output_dir, "best_checkpoint")
                    model.save_pretrained(save_path)
                    tokenizer.save_pretrained(save_path)
                    logging.info(f"Model saved to {save_path}")

                last_eval_time = current_time
                model.train() # Ensure model is back in train mode

            # --- Periodic Checkpoint Saving (Optional) ---
            # if current_time - last_save_time >= save_interval_seconds:
            #     checkpoint_path = os.path.join(output_dir, f"checkpoint-{total_steps}")
            #     model.save_pretrained(checkpoint_path)
            #     tokenizer.save_pretrained(checkpoint_path)
            #     logging.info(f"Periodic checkpoint saved to {checkpoint_path}")
            #     last_save_time = current_time

        # End of epoch (or break due to time limit)
        if training_complete:
            break

    logging.info("Training finished.")
    # Final save?
    final_save_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logging.info(f"Final model saved to {final_save_path}")

    if use_wandb:
        wandb.finish()

# --- Argument Parsing --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using DPO.")
    parser.add_argument("--sft_model_path", type=str, required=True, help="Path to the pretrained SFT model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the trained DPO model.")
    parser.add_argument("--hh_dataset_dir", type=str, required=True, help="Directory containing the processed HH dataset (.jsonl files).")

    parser.add_argument("--beta", type=float, default=0.1, help="Beta hyperparameter for DPO.")
    parser.add_argument("--learning_rate", type=float, default=1e-6, help="Learning rate for RMSprop.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay for RMSprop.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Number of steps to accumulate gradients.")
    parser.add_argument("--train_batch_size", type=int, default=1, help="Batch size per device per step (micro-batch size).")
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Batch size for evaluation.")
    parser.add_argument("--max_train_time_minutes", type=int, default=30, help="Maximum training time in minutes.")
    parser.add_argument("--val_set_size", type=int, default=200, help="Number of examples for the validation set.")
    parser.add_argument("--eval_interval_seconds", type=int, default=120, help="Interval in seconds for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")

    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu).")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "bfloat16", "float16"], help="Data type for model loading.")
    parser.add_argument("--no_flash_attn", action="store_true", help="Disable Flash Attention 2.")

    parser.add_argument("--wandb_project", type=str, default="cs336_alignment_dpo", help="WandB project name. Set to empty string to disable.")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name.")

    args = parser.parse_args()

    # Adjust wandb_project if empty string is passed
    wandb_project_arg = args.wandb_project if args.wandb_project else None

    train_dpo(
        sft_model_path=args.sft_model_path,
        output_dir=args.output_dir,
        hh_dataset_dir=args.hh_dataset_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        max_train_time_minutes=args.max_train_time_minutes,
        val_set_size=args.val_set_size,
        eval_interval_seconds=args.eval_interval_seconds,
        seed=args.seed,
        device_str=args.device,
        dtype_str=args.dtype,
        use_flash_attn=not args.no_flash_attn,
        wandb_project=wandb_project_arg,
        wandb_run_name=args.wandb_run_name
    ) 