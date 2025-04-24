#!/usr/bin/env python3
"""
Fine-tune a pre-trained language model on instruction-following data using PackedSFTDataset.

Based on the training script from CS336 Assignment 4.

Example usage:

# Single GPU
python scripts/train_sft.py --model_name_or_path ../Qwen/Qwen2.5-0.5B \
    --train_dataset_path data/sft/train.jsonl \
    --val_dataset_path data/sft/test.jsonl \
    --output_dir ./output/qwen-0.5b-sft \
    --seq_length 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --train_steps 500 \
    --learning_rate 2e-5 \
    --device cuda \
    --dtype float32 \
    --eval_interval 100 \
    --eval_iters 50 \
    --wandb_project cs336_alignment_sft

# Multi-GPU (Single Node, 2 GPUs) - Ensure WORLD_SIZE, RANK, LOCAL_RANK are set by torchrun
torchrun --standalone --nproc_per_node=2 scripts/train_sft.py --model_name_or_path ../Qwen/Qwen2.5-0.5B \
    --train_dataset_path data/sft/train.jsonl \
    --val_dataset_path data/sft/test.jsonl \
    --output_dir ./output/qwen-0.5b-sft \
    --seq_length 512 \
    --batch_size 2 \
    --gradient_accumulation_steps 16 \
    --train_steps 500 \
    --learning_rate 2e-5 \
    --device cuda \
    --dtype float32 \
    --eval_interval 100 \
    --eval_iters 50 \
    --wandb_project cs336_alignment_sft
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import pathlib
import sys
import random # Added for seeding
from contextlib import nullcontext

import torch
import torch.nn.functional as F
import wandb
from cs336_alignment.sft_dataset import PackedSFTDataset, iterate_batches # Updated import
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer # Added imports
from transformers.optimization import get_cosine_schedule_with_warmup # Using HF scheduler

logger = logging.getLogger(__name__)

def train(
    model_name_or_path: str,
    train_dataset_path: str,
    val_dataset_path: str,
    output_dir: str,
    seq_length: int,
    attn_implementation: str | None, # Added argument
    batch_size: int,
    train_steps: int, # Total optimizer steps
    gradient_accumulation_steps: int,
    eval_iters: int,
    eval_interval: int, # Interval in micro-steps
    learning_rate: float,
    # lr_scheduler: str, # Replaced by HF scheduler
    warmup_ratio: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    grad_clip: float | None,
    device: str,
    compile_model: bool, # Renamed from compile
    dtype: str,
    wandb_project: str | None,
):
    # --- DDP Setup ---
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    if is_ddp:
        init_process_group(backend="nccl")
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        seed = 42 + ddp_rank # Ensure different seed per process
        is_master_process = ddp_rank == 0
        print(f"DDP Rank {ddp_rank}: Using device {device}, seed {seed}")
    else:
        seed = 42
        ddp_world_size = 1
        is_master_process = True
        print(f"Single process: Using device {device}, seed {seed}")

    torch.manual_seed(seed)
    # Consider seeding numpy/random if used elsewhere, e.g., in iterate_batches if shuffle=True relies on them
    random.seed(seed)
    # np.random.seed(seed) # If numpy is used for randomness

    if is_master_process:
        effective_batch_size = gradient_accumulation_steps * ddp_world_size * batch_size
        total_optimizer_steps = train_steps # Use the argument directly
        logger.info(f"Total effective batch size: {effective_batch_size}")
        logger.info(f"Target optimizer steps: {total_optimizer_steps}")
        # Create output dir
        os.makedirs(output_dir, exist_ok=True)

    # --- Tokenizer and Model Loading ---
    device_type = "cuda" if "cuda" in device else "cpu"
    torch_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    if is_master_process:
        logger.info(f"Loading tokenizer from: {model_name_or_path}")
        logger.info(f"Loading model from: {model_name_or_path}")
        logger.info(f"Using dtype: {torch_dtype}")
        if attn_implementation:
            logger.info(f"Using attn_implementation: {attn_implementation}")

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    # Handle padding token for consistency if missing
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.warning(f"tokenizer.pad_token was None, setting to eos_token ({tokenizer.pad_token})")
        else:
            # If both are None, adding a pad token might be necessary depending on usage
            logger.warning("tokenizer.pad_token and tokenizer.eos_token are both None.")
            # Potentially add a pad token here if needed by other parts of the code
            # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            # model.resize_token_embeddings(len(tokenizer)) # If adding tokens

    model_kwargs = {"torch_dtype": torch_dtype}
    if attn_implementation:
        # Only add if not None or empty string
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        **model_kwargs
    )
    model = model.to(device)

    # --- Data Loading ---
    if is_master_process:
         logger.info("Loading datasets...")
    train_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=train_dataset_path,
        seq_length=seq_length,
        shuffle=True # Shuffle training data each epoch potentially
    )
    # Val dataset shuffle is typically False for consistent evaluation
    val_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=val_dataset_path,
        seq_length=seq_length,
        shuffle=False
    )
    if is_master_process:
        logger.info(f"Train dataset size: {len(train_dataset)} sequences")
        logger.info(f"Validation dataset size: {len(val_dataset)} sequences")

    # --- Optimizer and LR Scheduler ---
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    params_to_decay = [p for _, p in param_dict.items() if p.dim() >= 2]
    params_to_not_decay = [p for _, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": params_to_decay, "weight_decay": weight_decay},
        {"params": params_to_not_decay, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optim_groups,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
    )

    # Using Hugging Face Cosine Scheduler
    num_warmup_steps = int(total_optimizer_steps * warmup_ratio)
    lr_scheduler_hf = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_optimizer_steps, # Use total optimizer steps
    )
    if is_master_process:
        logger.info(f"Using Cosine LR scheduler with {num_warmup_steps} warmup steps over {total_optimizer_steps} optimizer steps.")


    # --- Compile Model (Optional) ---
    if compile_model:
        if is_master_process: logger.info("Compiling model...")
        # Ensure torch version supports compile
        if hasattr(torch, 'compile'):
            torch.set_float32_matmul_precision("high")
            model = torch.compile(model)
            if is_master_process: logger.info("Model compiled.")
        else:
            if is_master_process: logger.warning("torch.compile not available in this PyTorch version. Skipping compilation.")
            compile_model = False # Ensure we don't try to access _orig_mod later

    # --- Wrap model in DDP --- 
    if is_ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=False) # Set find_unused_parameters based on model structure
        if is_master_process: logger.info("Model wrapped in DDP.")

    # --- AMP and GradScaler ---
    amp_ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=torch_dtype)
    )
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == "float16")) # Use recommended constructor

    # --- Training Loop ---
    train_iterator = iter(iterate_batches(train_dataset, batch_size=batch_size, shuffle=True))
    pbar = None
    if is_master_process:
        pbar = tqdm(range(total_optimizer_steps), desc="Optimizer Steps") # Progress bar over optimizer steps

    model.train()
    completed_optimizer_steps = 0
    # Loop indefinitely, relying on completed_optimizer_steps to break
    micro_step = 0
    while completed_optimizer_steps < total_optimizer_steps:
        # Determine if this is the last micro-step in the current accumulation cycle
        is_last_micro_step = (micro_step + 1) % gradient_accumulation_steps == 0

        try:
            batch = next(train_iterator)
        except StopIteration:
            # End of epoch, re-initialize iterator
            if is_master_process: logger.info(f"Epoch finished after {completed_optimizer_steps} optimizer steps. Restarting data iterator...")
            train_iterator = iter(iterate_batches(train_dataset, batch_size=batch_size, shuffle=True))
            batch = next(train_iterator)

        batch_x = batch["input_ids"].to(device)
        batch_y = batch["labels"].to(device)

        # DDP gradient sync context
        sync_context = model.no_sync if (is_ddp and not is_last_micro_step) else nullcontext

        with sync_context():
            with amp_ctx:
                outputs = model(batch_x)
                logits = outputs.logits
                # Calculate the loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
                loss = loss / gradient_accumulation_steps # Scale loss for accumulation

            # loss.backward() needs to be scaled for float16
            scaler.scale(loss).backward()

        # Perform optimizer step only after accumulating gradients
        if is_last_micro_step:
            if grad_clip is not None:
                scaler.unscale_(optimizer) # Unscale before clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            scaler.step(optimizer)
            scaler.update() # Update scaler for next iteration
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler_hf.step() # Step the HF scheduler after optimizer step
            completed_optimizer_steps += 1

            # --- Logging and Evaluation (only on optimizer steps) ---
            loss_float = loss.item() * gradient_accumulation_steps # Log un-scaled loss

            if is_master_process:
                current_lr = optimizer.param_groups[0]['lr']
                log_msg = f"Step {completed_optimizer_steps}/{total_optimizer_steps}, Loss: {loss_float:.4f}, LR: {current_lr:.6f}"
                if pbar:
                    pbar.set_description(log_msg)
                    pbar.update(1)
                else:
                    logger.info(log_msg)

                if wandb_project:
                    wandb.log({"train/loss": loss_float, "train/lr": current_lr}, step=completed_optimizer_steps)

                # --- Periodic Evaluation ---
                # Evaluate based on completed optimizer steps
                if completed_optimizer_steps > 0 and completed_optimizer_steps % eval_interval == 0:
                    eval_loss = estimate_dev_loss(
                        model=model,
                        tokenizer=tokenizer,
                        val_dataset_path=val_dataset_path,
                        seq_length=seq_length,
                        batch_size=batch_size,
                        eval_iters=eval_iters,
                        device=device,
                        amp_ctx=amp_ctx,
                        ddp_world_size=ddp_world_size,
                        is_master_process_local=is_master_process # Pass flag as argument
                    )
                    logger.info(f"Step {completed_optimizer_steps}: Estimated validation loss: {eval_loss:.4f}")
                    if wandb_project:
                        wandb.log({"val/loss": eval_loss}, step=completed_optimizer_steps)
                    model.train() # Set model back to train mode

        micro_step = (micro_step + 1) % gradient_accumulation_steps

        # Break the outer loop if target optimizer steps are reached
        if completed_optimizer_steps >= total_optimizer_steps:
            break

    if pbar and is_master_process:
        pbar.close()

    # --- Final Evaluation and Saving --- 
    if is_master_process:
        logger.info("Training finished. Running final validation...")
        final_dev_loss = estimate_dev_loss(
             model=model,
             tokenizer=tokenizer,
             val_dataset_path=val_dataset_path,
             seq_length=seq_length,
             batch_size=batch_size,
             eval_iters=eval_iters * 2, # Use more iters for final estimate?
             device=device,
             amp_ctx=amp_ctx,
             ddp_world_size=ddp_world_size,
             is_master_process_local=is_master_process # Pass flag as argument
        )
        logger.info(f"Final estimated validation loss: {final_dev_loss:.4f}")
        if wandb_project:
            wandb.log({"val/final_loss": final_dev_loss})

        # Save the model and tokenizer
        logger.info(f"Saving final model and tokenizer to {output_dir}")
        # Ensure we save the unwrapped model if using DDP or compile
        model_to_save = model.module if is_ddp else model
        if compile_model and hasattr(model_to_save, '_orig_mod'):
             # Access underlying model if compiled with DDP: model.module._orig_mod
             # Access underlying model if compiled without DDP: model._orig_mod
             model_to_save = model_to_save._orig_mod

        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info("Model and tokenizer saved.")

        if wandb_project:
            wandb.finish()

    if is_ddp:
        destroy_process_group()


@torch.no_grad()
def estimate_dev_loss(
    model: torch.nn.Module, # Can be DDP or compiled
    tokenizer: AutoTokenizer,
    val_dataset_path: str,
    seq_length: int,
    batch_size: int,
    eval_iters: int,
    device: str,
    amp_ctx: torch.amp.autocast | nullcontext, # Pass context manager
    ddp_world_size: int = 1, # For averaging loss across GPUs
    is_master_process_local: bool = True # Add parameter with default for non-DDP
):
    """Estimates validation loss."""
    if is_master_process_local:
        logger.info("Estimating validation loss...")
    # Ensure model is in eval mode
    was_training = model.training
    model.eval()

    # Initialize validation dataset and iterator for this evaluation run
    # Use shuffle=False for consistent validation results
    val_dataset = PackedSFTDataset(
        tokenizer=tokenizer,
        dataset_path=val_dataset_path,
        seq_length=seq_length,
        shuffle=False
    )
    if len(val_dataset) == 0:
        if is_master_process_local: logger.warning("Validation dataset is empty. Returning NaN loss.")
        # Need to sync across processes if DDP is used, even for NaN
        loss_tensor = torch.tensor(float('nan'), device=device)
        if ddp_world_size > 1:
            torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
        return loss_tensor.item()

    val_iterator = iter(iterate_batches(val_dataset, batch_size=batch_size, shuffle=False))

    total_loss = 0.0
    actual_iters = 0
    pbar_val = None
    if is_master_process_local:
        pbar_val = tqdm(range(eval_iters), desc="Validation Batches", leave=False)

    for k in range(eval_iters):
        try:
            batch = next(val_iterator)
            actual_iters += 1
            batch_x = batch["input_ids"].to(device)
            batch_y = batch["labels"].to(device)
            with amp_ctx:
                outputs = model(batch_x)
                logits = outputs.logits
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            total_loss += loss.item()
            if pbar_val and is_master_process_local: pbar_val.update(1)
        except StopIteration:
            if is_master_process_local: logger.warning(f"Validation iterator finished after {actual_iters} iterations (requested {eval_iters}).")
            break # End evaluation if dataset runs out
    if pbar_val and is_master_process_local: pbar_val.close()

    # Average loss over actual iterations
    avg_loss = total_loss / actual_iters if actual_iters > 0 else float('nan')

    # Average loss across DDP processes if needed
    loss_tensor = torch.tensor(avg_loss, device=device)
    if ddp_world_size > 1:
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.AVG)
    
    final_avg_loss = loss_tensor.item()

    # Restore model training state
    if was_training:
        model.train()
    if is_master_process_local:
        logger.info("Finished estimating validation loss.")
    return final_avg_loss


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", # Added name
        level=logging.INFO,
    )
    # Setup master process check early
    is_ddp_main = int(os.environ.get("RANK", -1)) != -1
    is_master_process_main = int(os.environ.get("RANK", 0)) == 0 if is_ddp_main else True

    parser = argparse.ArgumentParser(
        description="Fine-tune a pre-trained language model on instruction data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, # Use defaults help
    )
    # --- Required arguments ---
    parser.add_argument("--model_name_or_path", required=True, help="Path or name of the pre-trained model/tokenizer.")
    parser.add_argument("--train_dataset_path", required=True, help="Path to the training JSONL file.")
    parser.add_argument("--val_dataset_path", required=True, help="Path to the validation JSONL file.")
    parser.add_argument("--output_dir", required=True, help="Directory to save the fine-tuned model and tokenizer.")
    parser.add_argument("--seq_length", type=int, required=True, help="Sequence length for dataset packing.")

    # --- Training hyperparameters ---
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size per GPU per accumulation step.")
    parser.add_argument("--train_steps", type=int, default=1000, help="Total number of training steps (optimizer steps).")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Number of forward passes per optimizer step.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Peak learning rate.")
    # parser.add_argument("--lr_scheduler", type=str, choices=["constant", "cosine"], default="cosine", help="Learning rate scheduler type (uses HF cosine).") # Removed, using HF cosine
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of total steps for linear LR warmup.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="AdamW weight decay.") # Note: HF default is 0.0
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2.") # Note: HF default is 0.999
    parser.add_argument("--adam_eps", type=float, default=1e-8, help="AdamW epsilon.")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value (norm). Set to 0 or None to disable.")

    # --- Evaluation ---
    parser.add_argument("--eval_iters", type=int, default=100, help="Number of validation batches to evaluate.")
    parser.add_argument("--eval_interval", type=int, default=500, help="Evaluate every N training steps (optimizer steps).")

    # --- System / Hardware ---
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda, cpu).")
    parser.add_argument("--compile_model", action="store_true", help="Compile the model with torch.compile.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="float32", help="Compute dtype. float32 is recommended for compatibility unless using recent GPUs.")
    parser.add_argument("--attn_implementation", type=str, choices=["eager", "sdpa", "flash_attention_2"], default=None, help="Attention implementation override (e.g., flash_attention_2 for A100+). Defaults to None (auto).")

    # --- Logging ---
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name for logging.")

    args = parser.parse_args()

    # Convert grad_clip=0.0 to None
    if args.grad_clip == 0.0:
        args.grad_clip = None

    # ECE491B Specific Note / General Hardware Notes:
    if args.attn_implementation == "flash_attention_2":
        if is_master_process_main: logger.warning("FlashAttention-2 selected via --attn_implementation. Ensure it's installed (pip install flash-attn --no-build-isolation) and hardware is compatible (e.g., Ampere+).")
    if args.dtype == "bfloat16" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        if is_master_process_main: logger.warning("bfloat16 requested via --dtype but may not be supported or optimal on this hardware. Consider float16 or float32.")
    elif args.dtype == "float16" and args.device != "cpu":
        if is_master_process_main: logger.info("float16 selected via --dtype. Using GradScaler.")
    # No special note needed for float32 as it's the default and most compatible.


    if is_master_process_main:
        logger.info("Starting SFT Training Script...")
        logger.info("Arguments: %s", vars(args))
        # Check output dir
        if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
             logger.warning(f"Output directory {args.output_dir} already exists and is not empty. Files may be overwritten.")
             # Decide whether to overwrite or raise error
             # raise ValueError(f"Output directory {args.output_dir} exists and is not empty.")
        else:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.wandb_project:
            try:
                wandb.login()
                wandb.init(
                    project=args.wandb_project,
                    config=vars(args),
                    name=pathlib.Path(args.output_dir).name or "sft-run",
                )
                logger.info(f"WandB logging enabled for project: {args.wandb_project}")
            except Exception as e:
                logger.error(f"Failed to initialize WandB: {e}. Disabling WandB.")
                args.wandb_project = None # Disable WandB if init fails

    # --- Run Training ---
    train(
        model_name_or_path=args.model_name_or_path,
        train_dataset_path=args.train_dataset_path,
        val_dataset_path=args.val_dataset_path,
        output_dir=args.output_dir,
        seq_length=args.seq_length,
        attn_implementation=args.attn_implementation,
        batch_size=args.batch_size,
        train_steps=args.train_steps, # Pass total optimizer steps
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_iters=args.eval_iters,
        eval_interval=args.eval_interval, # Eval interval is in optimizer steps
        learning_rate=args.learning_rate,
        # lr_scheduler=args.lr_scheduler, # Removed
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_eps=args.adam_eps,
        grad_clip=args.grad_clip,
        device=args.device,
        compile_model=args.compile_model,
        dtype=args.dtype,
        wandb_project=args.wandb_project,
    )

    if is_master_process_main:
        logger.info("Finished running %s", sys.argv[0]) 