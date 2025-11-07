import argparse
import numpy as np
import json
import os
import torch
import wandb
from datetime import datetime
from functools import partial

from .model import BasicsTransformerLM
from .loss import cross_entropy
from .optimizers import AdamW, lr_cosine_schedule
from .data import random_sample, det_sample
from .timer import Timer


def train_parser(
    parser: argparse.ArgumentParser,
    defaults: dict,
    argv,
):
    for key, value in defaults.items():
        if key in ["total_tokens", "d_model", "num_layers", "num_heads"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=int)
        elif key in ["lr", "compute_budget"]:
            parser.add_argument(f"--{key}", dest=key, default=value, type=float)
        else:
            parser.add_argument(f"--{key}", dest=key, default=value)
    return parser.parse_args(argv)


def train(data, model, params, device):
    # Random sample a batch for training.
    batch, label = random_sample(data, params.batch_size, params.context_length, device)
    logits = model(batch)
    loss = cross_entropy(logits, label)
    return loss


def validation(data, model, params, device):
    model.eval()
    with torch.no_grad():
        # Iterate through the eval data for validation.
        batch_size, context_length = params.batch_size, params.context_length
        total_loss, num_instances = 0, 0
        for idx in range(0, data.size, batch_size * context_length):
            batch, label = det_sample(data, idx, batch_size, context_length, device)
            logits = model(batch)
            loss = cross_entropy(logits, label)
            total_loss += loss * label.shape[0]
            num_instances += label.shape[0]
    model.train()
    return total_loss / num_instances


def run(argv=None):
    torch.autograd.set_detect_anomaly(True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser()
    config = dict(
        train_data="../data/SlimPajama-627B-valid-chunk1-ids.bin", # NOTE: change it to train ids.
        validation_data="../data/SlimPajama-627B-valid-chunk1-ids.bin",
        batch_size=32,
        gradient_accumulation=4,
        vocab_size=1000,
        compute_budget=1e13,
        context_length=512,
        d_model=1024,
        num_layers=24,
        num_heads=16,
        d_ff=4*1024,
        attn_pdrop=0.1,
        residual_pdrop=0.1,
        lr=1e-3,
        betas=(0.9,0.95),
        eps=1e-8,
        weight_decay=0.01,
        grad_clip_norm=1.0,
        alpha_min=0.1*1e-3,
        t_w=0,
        t_c=19_000,
        ckpt_path="./ckpts/my_model.pt",
        dry_run=True,
    )
    config = train_parser(parser, defaults=config, argv=argv)
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
        attn_pdrop=config.attn_pdrop,
        residual_pdrop=config.residual_pdrop,
    ).to(device)
    # Calculate n_steps given compute.
    config.total_params = model.get_num_params()
    config.total_tokens = config.compute_budget // (6 * config.total_params)
    config.n_steps = config.total_tokens // (config.batch_size * config.gradient_accumulation * config.context_length)
    opt = AdamW(
        model.parameters(),
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
        lr_schedule=partial(
            lr_cosine_schedule,
            alpha_max=config.lr,
            alpha_min=config.alpha_min,
            t_w=config.t_w,
            t_c=config.n_steps,
        ),
    )
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dataset = np.memmap(os.path.join(current_dir, config.train_data), dtype=np.int32, mode='r')
    validation_dataset = np.memmap(os.path.join(current_dir, config.validation_data), dtype=np.int32, mode='r')
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if config.dry_run:
        print(config.n_steps)
        if config.n_steps >= 10 and config.n_steps <= 10_000:
            with open("/tmp/config.jsonl", "a") as f:
                json.dump(vars(config), f)
                f.write("\n")
        os._exit(0)
    
    # Training Loops.
    wandb.init(project="cs336", name=f"{current_time}-assignment3-scaling", config=config)
    for i in range(config.n_steps):
        with Timer("Training Loop") as t:
            opt.zero_grad()
            batch_loss = 0
            for _ in range(config.gradient_accumulation):
                loss = train(train_dataset, model, config, device) / config.gradient_accumulation
                loss.backward()
                batch_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
            opt.step()
            if i > 0 and i % 10 == 0:
                print(f"Step {i} Training Loss: {batch_loss}")
        wandb.log({
            "batch_loss": batch_loss,
            "batch_idx": i,
            "step_time": t.elapsed,
            "lr": opt.lr_schedule(i),
            "memory_allocated_gb": torch.cuda.memory_allocated(device) / 1e9,
            "memory_reserved_gb": torch.cuda.memory_reserved(device) / 1e9,
        })
        if i > 0 and i % 100 == 0:    
            valid_loss = validation(validation_dataset[:10000], model, config, device)
            print(f"Step {i} Validation Loss: {valid_loss}")
            wandb.log({
                "train_validation_loss": valid_loss.item(),
                "batch_idx": i,
            })
    valid_loss = validation(validation_dataset[:10000], model, config, device)
    print(f"Final Validation Loss: {valid_loss}")
    wandb.log({
        "eval_validation_loss": valid_loss.item(),
        "batch_idx": i,
    })
    torch.save(model, config.ckpt_path)

if __name__ == "__main__":
    run()