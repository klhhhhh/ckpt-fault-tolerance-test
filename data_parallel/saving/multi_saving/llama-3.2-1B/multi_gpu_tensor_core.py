import os
import torch
import deepspeed
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import time

# ✅ **Dummy Dataset**
class DummyDataset(Dataset):
    def __init__(self, tokenizer, text="Hello world!", length=1000):
        tokenizer.pad_token = tokenizer.eos_token
        self.inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.inputs["input_ids"][0], self.inputs["attention_mask"][0]

# ✅ **DeepSpeed Checkpoint Save**
def save_checkpoint(engine, checkpoint_dir, step, mode="sync", executor=None):
    """
    Save checkpoint using DeepSpeed, supporting both Sync & Async modes.
    - ✅ ZeRO-3 automatically saves sharded model parameters, optimizer states, and gradients.
    - ✅ No need to manually save gradients, DeepSpeed manages it automatically.
    """
    rank = engine.local_rank  # Get GPU process ID
    start_time = time.time()

    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")

    if mode == "sync":
        engine.save_checkpoint(checkpoint_path)
    elif mode == "async":
        executor.submit(engine.save_checkpoint, checkpoint_path)

    checkpoint_time = time.time() - start_time
    print(f"[GPU {rank}] [Step {step}] Checkpoint saved at {checkpoint_path} (mode={mode}, time={checkpoint_time:.2f}s)")
    return checkpoint_time

# ✅ **DeepSpeed Checkpoint Load**
def load_checkpoint(engine, checkpoint_dir):
    """
    DeepSpeed automatically restores model parameters, optimizer states, and gradients (sharded by ZeRO-3).
    """
    rank = engine.local_rank  # Current GPU rank
    start_time = time.time()

    latest_ckpt = sorted(os.listdir(checkpoint_dir))[-1]  # Select the latest checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    engine.load_checkpoint(checkpoint_path)  # ✅ Automatically restore model, optimizer, and gradients

    load_time = time.time() - start_time
    print(f"[GPU {rank}] Checkpoint loaded from {checkpoint_path} in {load_time:.2f}s")

# ✅ **Training & Checkpointing**
def train_and_checkpoint(engine, dataloader, checkpoint_dir, checkpoint_interval, mode="sync"):
    """
    During training, each GPU only saves the parameters it is responsible for, supporting both Sync & Async modes, and ZeRO-3.
    """
    executor = ThreadPoolExecutor(max_workers=1) if mode == "async" else None
    total_train_time = 0
    total_checkpoint_time = 0

    start_train_time = time.time()

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(engine.device)
        attention_mask = attention_mask.to(engine.device)

        engine.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):  # ✅ Enable Tensor Core training
            outputs = engine.module(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        engine.backward(loss)
        engine.step()

        # ✅ **Periodically save checkpoint**
        if step % checkpoint_interval == 0 and step > 0:
            total_checkpoint_time += save_checkpoint(engine, checkpoint_dir, step, mode, executor)

    total_train_time = time.time() - start_train_time

    if mode == "async":
        executor.shutdown(wait=True)

    print(f"Training completed. Mode: {mode}, ZeRO-3 Enabled, Tensor Core: Enabled")
    print(f"Total training time: {total_train_time:.2f}s")
    print(f"Total checkpoint time: {total_checkpoint_time:.2f}s")

if __name__ == "__main__":
    # ✅ **DeepSpeed Configuration**
    deepspeed_config = {
        "train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {
            "stage": 3,  # ✅ ZeRO-3: Each GPU only saves its own parameters, optimizer states, and gradients
            # "offload_optimizer": {"device": "cpu"},
            # "offload_param": {"device": "cpu"},  # ✅ Optional: Offload parameters to CPU
            "contiguous_gradients": True,  # ✅ Make gradient memory allocation more efficient
        },
        "fp16": {"enabled": True},  # ✅ Enable FP16 training
    }

    # ✅ **Initialize Model**
    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ✅ **Dataloader**
    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=4)

    # ✅ **Optimizer**
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ✅ **DeepSpeed Initialization**
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config_params=deepspeed_config
    )

    # ✅ **Checkpoint Directory**
    checkpoint_dir = "/pscratch/sd/k/klhhhhh/deepspeed_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ✅ **Execute Training in Different Modes**
    for mode in ["sync", "async"]:
        train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, mode=mode)

    # ✅ **Load the Latest Checkpoint**
    load_checkpoint(model, checkpoint_dir)
