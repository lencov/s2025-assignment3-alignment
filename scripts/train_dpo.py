import os
import math
import torch
import wandb
import argparse
from tqdm import tqdm
from datetime import datetime
from dataclasses import dataclass, field
import time
import logging
import random
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase
from typing import Tuple

# Assuming these utility functions exist in your project structure
from cs336_alignment.dpo import compute_per_instance_dpo_loss, get_sequence_log_probs, format_alpaca
from scripts.load_hh_data import load_processed_hh_dataset # Use the HH loading script

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Increase timeout to prevent wandb errors
os.environ["WANDB__SERVICE_WAIT"] = "300"
# turn off watch to log faster
os.environ["WANDB_WATCH"] = "false"

@dataclass
class DPOConfig:
    # Paths
    sft_model_path: str = "./output/qwen-0.5b-sft-short" # Path to the starting SFT model
    hh_dataset_dir: str = "./data/hh"                  # Directory with processed HH data
    output_dir: str = "./output/qwen-0.5b-dpo"         # Where to save DPO checkpoints

    # DPO Hyperparameters
    beta: float = 0.1

    # Optimizer Hyperparameters
    learning_rate: float = 1e-6
    weight_decay: float = 0.0 # RMSprop usually doesn't use weight decay
    optimizer_alpha: float = 0.99 # RMSprop alpha

    # Training Setup
    train_batch_size: int = 1       # Micro-batch size per step
    gradient_accumulation_steps: int = 32 # Accumulate gradients for this many steps
    max_train_time_minutes: int = 30
    seed: int = 42

    # Validation Setup
    val_set_size: int = 200
    eval_batch_size: int = 4
    eval_interval_seconds: int = 120 # Evaluate every 2 minutes

    # Technical Details
    device: str = "cuda"           # Device to use ("cuda" or "cpu")
    dtype: str = "bfloat16"      # Data type ("bfloat16", "float16", "float32")
    use_flash_attn: bool = True    # Use Flash Attention 2 if available and dtype allows
    gradient_checkpointing: bool = False # Consider enabling if memory is tight, though likely ok on A100

    # Logging
    wandb_project: str | None = "cs336_alignment_dpo" # Project name, None or empty to disable
    wandb_run_name: str | None = None                 # Run name, defaults to timestamp
    log_interval_steps: int = 10                      # Log training loss every N steps

    # Calculated field (effective batch size)
    effective_batch_size: int = field(init=False)

    def __post_init__(self):
        self.effective_batch_size = self.train_batch_size * self.gradient_accumulation_steps

def get_optimizer(model: PreTrainedModel, config: DPOConfig) -> optim.Optimizer:
    """Sets up the RMSprop optimizer based on config."""
    return optim.RMSprop(
        model.parameters(),
        lr=config.learning_rate,
        alpha=config.optimizer_alpha,
        weight_decay=config.weight_decay
    )

@torch.no_grad()
def evaluate(model: PreTrainedModel, model_ref: PreTrainedModel, tokenizer: PreTrainedTokenizerBase,
             val_loader: DataLoader, beta: float, device: torch.device) -> Tuple[float, float]:
    """Evaluates the model on the validation set for DPO loss and implicit reward accuracy."""
    model.eval()
    # model_ref is already frozen and in eval mode

    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0

    pbar = tqdm(val_loader, desc="Validation")
    for batch in pbar:
        batch_losses = []
        batch_pi_log_probs_chosen = []
        batch_pi_log_probs_rejected = []

        # Process each item individually
        for i in range(len(batch["instruction"])):
            prompt = batch["instruction"][i]
            chosen = batch["chosen"][i]
            rejected = batch["rejected"][i]

            # --- Calculate Loss ---
            loss = compute_per_instance_dpo_loss(
                lm=model,
                lm_ref=model_ref,
                tokenizer=tokenizer,
                beta=beta,
                prompt=prompt,
                response_chosen=chosen,
                response_rejected=rejected,
            )
            batch_losses.append(loss.item())

            # --- Calculate implicit reward accuracy ---
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

            # Mask prompt tokens using the same logic as in compute_per_instance_dpo_loss
            # We need to compute the log prob of the response part only
            prompt_tokens = tokenizer(prompt, return_tensors="pt", padding=False, truncation=False)
            prompt_length = prompt_tokens.input_ids.shape[1]
            # Ensure masking doesn't go out of bounds if prompt is long
            labels[:, :min(prompt_length, labels.shape[1])] = -100

            # Get log probs for the *current policy model* only (response tokens)
            log_probs_lm = get_sequence_log_probs(model, tokenized_sequences.input_ids, labels, tokenized_sequences.attention_mask)
            log_probs_lm_chosen = log_probs_lm[0]
            log_probs_lm_rejected = log_probs_lm[1]

            batch_pi_log_probs_chosen.append(log_probs_lm_chosen.item())
            batch_pi_log_probs_rejected.append(log_probs_lm_rejected.item())

        # --- Aggregate batch results ---
        avg_batch_loss = sum(batch_losses) / len(batch_losses) if batch_losses else 0
        total_loss += avg_batch_loss * len(batch_losses)

        for pi_chosen, pi_rejected in zip(batch_pi_log_probs_chosen, batch_pi_log_probs_rejected):
            if pi_chosen > pi_rejected:
                correct_predictions += 1
        total_predictions += len(batch_losses)

        pbar.set_postfix({"avg_loss": f"{avg_batch_loss:.4f}"})

    # --- Calculate overall metrics ---
    avg_val_loss = total_loss / total_predictions if total_predictions > 0 else 0.0
    val_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    model.train() # Set model back to train mode
    return avg_val_loss, val_accuracy

def main(config: DPOConfig):
    """Main DPO training function, structured similarly to the example."""
    # --- Setup --- #
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    run_name = config.wandb_run_name or f"dpo-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_dir = os.path.join(config.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    if config.device == "cuda" and not torch.cuda.is_available():
        logging.warning("CUDA specified but not available. Using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(config.device)

    pt_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[config.dtype]
    logging.info(f"Using device: {device}, dtype: {pt_dtype}")

    # --- Load Model & Tokenizer --- #
    logging.info(f"Loading SFT model and tokenizer from: {config.sft_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(config.sft_model_path)
    if tokenizer.pad_token is None:
        logging.warning("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": pt_dtype,
        "use_cache": False,
    }
    if config.use_flash_attn and config.dtype != "float32":
        logging.info("Attempting to use Flash Attention 2")
        model_kwargs["attn_implementation"] = "flash_attention_2"
    elif config.use_flash_attn:
        logging.warning("Flash Attention requested but dtype is float32. Using default.")

    # Load models onto the *same* specified device
    logging.info(f"Loading policy model to {device}...")
    model = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    if config.gradient_checkpointing:
        logging.info("Enabling gradient checkpointing.")
        model.gradient_checkpointing_enable()
    model.to(device)
    model.train()

    logging.info(f"Loading reference model to {device}...")
    model_ref = AutoModelForCausalLM.from_pretrained(config.sft_model_path, **model_kwargs)
    model_ref.to(device)
    model_ref.eval()
    for param in model_ref.parameters(): # Freeze reference model
        param.requires_grad = False
    logging.info("Loaded policy and reference models.")

    # --- Load Dataset --- #
    logging.info(f"Loading HH dataset from: {config.hh_dataset_dir}")
    # Make sure the file_map inside load_hh_data corresponds to your file names
    full_dataset = load_processed_hh_dataset(data_dir=config.hh_dataset_dir)
    if not full_dataset:
        raise ValueError(f"Failed to load dataset from {config.hh_dataset_dir}. Check paths/files.")

    # --- Split Dataset --- #
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    val_indices = indices[:config.val_set_size]
    train_indices = indices[config.val_set_size:]
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    logging.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation examples.")

    # --- Data Loaders --- #
    # Simple collator: The DataLoader returns a list of dicts, we process individually
    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.eval_batch_size)

    # --- Optimizer --- #
    optimizer = get_optimizer(model, config)

    # --- Logging --- #
    use_wandb = False
    if config.wandb_project:
        try:
            wandb.init(project=config.wandb_project, name=run_name, config=vars(config))
            logging.info(f"Weights & Biases initialized for run: {run_name}")
            use_wandb = True
        except ImportError:
            logging.warning("wandb not installed, skipping Weights & Biases logging.")
        except Exception as e:
             logging.warning(f"Could not initialize WandB: {e}")

    # --- Training Loop --- #
    logging.info(f"Starting DPO training. Effective batch size: {config.effective_batch_size}")
    logging.info(f"Training for a maximum of {config.max_train_time_minutes} minutes.")

    start_time = time.time()
    last_eval_time = start_time
    total_steps = 0
    processed_examples_accum = 0 # Examples processed since last optimizer step
    total_processed_examples = 0 # Total examples processed overall
    best_val_accuracy = -1.0
    training_complete = False
    accumulated_loss = 0.0

    optimizer.zero_grad() # Initialize gradients

    while not training_complete:
        model.train()
        epoch_iterator = tqdm(train_loader, desc=f"Step {total_steps}")
        for batch in epoch_iterator:
            # Check training time limit
            elapsed_time = time.time() - start_time
            if elapsed_time > config.max_train_time_minutes * 60:
                logging.info(f"Training time limit ({config.max_train_time_minutes} minutes) reached.")
                training_complete = True
                break

            # --- Process Micro-Batch --- #
            actual_batch_size = len(batch["instruction"])
            micro_batch_loss_sum = 0.0

            for i in range(actual_batch_size):
                prompt = batch["instruction"][i]
                chosen = batch["chosen"][i]
                rejected = batch["rejected"][i]

                # Calculate loss for one instance
                loss = compute_per_instance_dpo_loss(
                    lm=model,
                    lm_ref=model_ref,
                    tokenizer=tokenizer,
                    beta=config.beta,
                    prompt=prompt,
                    response_chosen=chosen,
                    response_rejected=rejected,
                )

                # Normalize loss for accumulation and average calculation
                loss_normalized = loss / config.gradient_accumulation_steps
                micro_batch_loss_sum += loss.item() # Accumulate actual loss for step logging

                # Backward pass - accumulates gradients
                loss_normalized.backward()

            processed_examples_accum += actual_batch_size
            total_processed_examples += actual_batch_size
            accumulated_loss += micro_batch_loss_sum # Track sum of un-normalized losses for the step

            # --- Gradient Accumulation Step --- #
            if processed_examples_accum >= config.gradient_accumulation_steps:
                # Ensure we processed exactly the expected number for the step
                actual_accum_steps = processed_examples_accum / config.train_batch_size
                # Clip gradients
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad() # Reset gradients
                total_steps += 1

                # --- Logging --- #
                avg_step_loss = accumulated_loss / processed_examples_accum # Avg loss over accumulated examples
                epoch_iterator.set_description(f"Step {total_steps}")
                epoch_iterator.set_postfix({"loss": f"{avg_step_loss:.4f}", "grad_norm": f"{grad_norm:.4f}"})

                if total_steps % config.log_interval_steps == 0:
                     if use_wandb:
                        wandb.log({
                            "train/loss": avg_step_loss,
                            "train/grad_norm": grad_norm,
                            "step": total_steps,
                            "examples": total_processed_examples
                        })

                # Reset accumulators for next step
                processed_examples_accum = 0
                accumulated_loss = 0.0

                # --- Periodic Evaluation --- #
                current_time = time.time()
                if current_time - last_eval_time >= config.eval_interval_seconds:
                    logging.info(f"Running validation at step {total_steps}...")
                    avg_val_loss, val_accuracy = evaluate(
                        model, model_ref, tokenizer, val_loader, config.beta, device
                    )
                    logging.info(f"Validation Step {total_steps}: Loss={avg_val_loss:.4f}, Accuracy={val_accuracy:.4f}")
                    if use_wandb:
                        wandb.log({"val/loss": avg_val_loss, "val/accuracy": val_accuracy, "step": total_steps})

                    # --- Save Best Model --- #
                    if val_accuracy > best_val_accuracy:
                        best_val_accuracy = val_accuracy
                        logging.info(f"New best val accuracy: {best_val_accuracy:.4f}. Saving model...")
                        save_path = os.path.join(output_dir, "best_checkpoint")
                        model.save_pretrained(save_path)
                        tokenizer.save_pretrained(save_path)
                        # Save config too
                        with open(os.path.join(save_path, "dpo_config.json"), "w") as f:
                            import json
                            json.dump(vars(config), f, indent=4)
                        logging.info(f"Best model saved to {save_path}")
                    else:
                        logging.info(f"Validation accuracy ({val_accuracy:.4f}) did not improve from best ({best_val_accuracy:.4f})")


                    last_eval_time = current_time
                    model.train() # Ensure model is back in train mode

        # --- End of Epoch / Data Iteration --- #
        # The outer while loop handles the time limit
        if training_complete:
            break # Exit outer loop if time limit hit mid-epoch

    # --- Training Finished --- #
    logging.info("Training finished.")
    elapsed_total_time = time.time() - start_time
    logging.info(f"Total training time: {elapsed_total_time / 60:.2f} minutes")
    logging.info(f"Total steps completed: {total_steps}")

    # Final save
    final_save_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    with open(os.path.join(final_save_path, "dpo_config.json"), "w") as f:
        import json
        json.dump(vars(config), f, indent=4)
    logging.info(f"Final model saved to {final_save_path}")

    if use_wandb:
        wandb.finish()

# --- Argument Parsing and Main Execution --- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model using DPO.")

    # Add arguments from DPOConfig dataclass
    parser.add_argument("--sft_model_path", type=str, default=DPOConfig.sft_model_path, help="Path to the SFT model checkpoint.")
    parser.add_argument("--hh_dataset_dir", type=str, default=DPOConfig.hh_dataset_dir, help="Directory with HH data.")
    parser.add_argument("--output_dir", type=str, default=DPOConfig.output_dir, help="Directory to save DPO model.")
    parser.add_argument("--beta", type=float, default=DPOConfig.beta, help="DPO beta hyperparameter.")
    parser.add_argument("--learning_rate", type=float, default=DPOConfig.learning_rate, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=DPOConfig.weight_decay, help="Weight decay.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=DPOConfig.gradient_accumulation_steps, help="Gradient accumulation steps.")
    parser.add_argument("--train_batch_size", type=int, default=DPOConfig.train_batch_size, help="Micro-batch size.")
    parser.add_argument("--eval_batch_size", type=int, default=DPOConfig.eval_batch_size, help="Validation batch size.")
    parser.add_argument("--max_train_time_minutes", type=int, default=DPOConfig.max_train_time_minutes, help="Max training time (minutes).")
    parser.add_argument("--val_set_size", type=int, default=DPOConfig.val_set_size, help="Validation set size.")
    parser.add_argument("--eval_interval_seconds", type=int, default=DPOConfig.eval_interval_seconds, help="Evaluation interval (seconds).")
    parser.add_argument("--seed", type=int, default=DPOConfig.seed, help="Random seed.")
    parser.add_argument("--device", type=str, default=DPOConfig.device, help="Device (cuda/cpu).")
    parser.add_argument("--dtype", type=str, default=DPOConfig.dtype, choices=["float32", "bfloat16", "float16"], help="Data type.")
    parser.add_argument("--no_flash_attn", action="store_true", help="Disable Flash Attention 2.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing.")
    parser.add_argument("--wandb_project", type=str, default=DPOConfig.wandb_project, help="WandB project name (empty string or None to disable).")
    parser.add_argument("--wandb_run_name", type=str, default=DPOConfig.wandb_run_name, help="WandB run name.")
    parser.add_argument("--log_interval_steps", type=int, default=DPOConfig.log_interval_steps, help="Log training loss every N steps.")

    args = parser.parse_args()

    # Create config object from args
    config = DPOConfig(
        sft_model_path=args.sft_model_path,
        hh_dataset_dir=args.hh_dataset_dir,
        output_dir=args.output_dir,
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
        device=args.device,
        dtype=args.dtype,
        use_flash_attn=not args.no_flash_attn,
        gradient_checkpointing=args.gradient_checkpointing,
        wandb_project=args.wandb_project if args.wandb_project else None, # Handle empty string
        wandb_run_name=args.wandb_run_name,
        log_interval_steps=args.log_interval_steps
    )

    main(config) 