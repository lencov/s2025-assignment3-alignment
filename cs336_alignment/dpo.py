import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerBase, PreTrainedModel
from typing import Tuple, List, Dict
import os
import json
import random
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


PROMPT_FORMAT = (
    "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{prompt}\n\n### Response:\n{response}"
)

def get_sequence_logprob(
    policy: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor
) -> torch.Tensor:

    input_ids = input_ids.to(policy.device)
    labels = labels.to(policy.device)


    logits = policy(input_ids=input_ids).logits


    shifted_logits = logits[:, :-1, :].contiguous()


    log_probs = torch.nn.functional.log_softmax(shifted_logits, dim=-1)


    per_token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)


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


    if tokenizer.pad_token_id is None:
         if tokenizer.eos_token_id is not None:

             tokenizer.pad_token_id = tokenizer.eos_token_id
         else:

             tokenizer.add_special_tokens({'pad_token': '[PAD]'})
             logging.warning("Tokenizer missing pad and eos token. Added '[PAD]' as pad token.")


    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:

        tokenizer.add_special_tokens({'eos_token': '</s>'})
        eos_token_id = tokenizer.eos_token_id
        logging.warning("Tokenizer missing eos token. Added '</s>' as eos token.")


    chosen_sequence_str = PROMPT_FORMAT.format(prompt=prompt, response=chosen_response)
    rejected_sequence_str = PROMPT_FORMAT.format(prompt=prompt, response=rejected_response)

    chosen_input_ids = torch.tensor(tokenizer.encode(chosen_sequence_str) + [eos_token_id], dtype=torch.long).unsqueeze(0)
    rejected_input_ids = torch.tensor(tokenizer.encode(rejected_sequence_str) + [eos_token_id], dtype=torch.long).unsqueeze(0)


    chosen_labels = chosen_input_ids[:, 1:].clone()
    rejected_labels = rejected_input_ids[:, 1:].clone()


    with torch.no_grad():

        ref_chosen_logprob = get_sequence_logprob(reference_model, chosen_input_ids, chosen_labels)
        ref_rejected_logprob = get_sequence_logprob(reference_model, rejected_input_ids, rejected_labels)


    policy_chosen_logprob = get_sequence_logprob(policy_model, chosen_input_ids, chosen_labels)
    policy_rejected_logprob = get_sequence_logprob(policy_model, rejected_input_ids, rejected_labels)


    policy_device = policy_model.device
    ref_chosen_logprob = ref_chosen_logprob.to(policy_device)
    ref_rejected_logprob = ref_rejected_logprob.to(policy_device)


    pi_logratios = policy_chosen_logprob - policy_rejected_logprob
    ref_logratios = ref_chosen_logprob - ref_rejected_logprob

    logits = pi_logratios - ref_logratios

    loss = -torch.nn.functional.logsigmoid(beta * logits).mean()

    return loss


def get_dpo_dataset(
    data_dir: str,
    tokenizer: PreTrainedTokenizerBase,
    val_set_size: int = 200,
    seed: int = 42
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:


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