import os
import sys
import torch
import torch.distributed as dist
import deepspeed
import logging
import torch.multiprocessing as mp

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import time
from multiprocessing import Manager
from multi_gpu_save_ckpt import train_and_checkpoint, train_without_checkpoint, DummyDataset, setup_logging, load_checkpoint

if __name__ == "__main__":
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    manager = Manager()
    log_lock = manager.Lock()

    precision = "FP16"
    dtype = torch.float16

    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    checkpoint_dir = "/pscratch/sd/k/klhhhhh/deepspeed_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    deepspeed_config = {
        "train_batch_size": 256,
        "train_micro_batch_size_per_gpu": 8,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {
            "stage": 3,
            "contiguous_gradients": True,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": true
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": true
            }
        },
        "checkpoint": {
            "async_io": False  # ✅ 启用异步 checkpoint 读写
        },
        "logging": {"verbosity": 0},
        "bf16": {"enabled": False},
        "fp16": {"enabled": True}
    }

    my_logger = setup_logging(precision, "sync")
    model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=deepspeed_config)
    train_without_checkpoint(model, dataloader, dtype, precision, log_lock, my_logger)
    train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, dtype=dtype, precision=precision, log_lock=log_lock, my_logger=my_logger)
    # load_time = load_checkpoint(model, checkpoint_dir, precision, log_lock, my_logger)
    
    # with log_lock:
    #     my_logger.info(f"Final Checkpoint Load Time: {load_time:.2f}s")
    #     my_logger.handlers[0].flush()

    if dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0)
