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
    使用 DeepSpeed 存储 Checkpoint，并支持 Sync & Async 模式。
    - Sync: engine.save_checkpoint()
    - Async: 通过线程池异步存储
    """
    rank = engine.local_rank  # 获取 GPU 进程 ID
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
    DeepSpeed 负责自动恢复模型参数、优化器状态和梯度。
    """
    rank = engine.local_rank  # 当前 GPU Rank
    start_time = time.time()

    latest_ckpt = sorted(os.listdir(checkpoint_dir))[-1]  # 选择最新的 Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    engine.load_checkpoint(checkpoint_path)

    load_time = time.time() - start_time
    print(f"[GPU {rank}] Checkpoint loaded from {checkpoint_path} in {load_time:.2f}s")

# ✅ **Training & Checkpointing**
def train_and_checkpoint(engine, dataloader, checkpoint_dir, checkpoint_interval, mode="sync"):
    """
    训练过程中，每个 GPU 只存储自己负责的部分参数，支持 Sync & Async，支持 Tensor Core。
    """
    executor = ThreadPoolExecutor(max_workers=1) if mode == "async" else None
    total_train_time = 0
    total_checkpoint_time = 0

    start_train_time = time.time()

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(engine.device)
        attention_mask = attention_mask.to(engine.device)

        engine.zero_grad()
        with torch.cuda.amp.autocast(dtype=torch.float16):  # ✅ 启用 Tensor Core 训练
            outputs = engine.module(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        engine.backward(loss)
        engine.step()

        # ✅ **定期存储 Checkpoint**
        if step % checkpoint_interval == 0 and step > 0:
            total_checkpoint_time += save_checkpoint(engine, checkpoint_dir, step, mode, executor)

    total_train_time = time.time() - start_train_time

    if mode == "async":
        executor.shutdown(wait=True)

    print(f"Training completed. Mode: {mode}, Tensor Core: Enabled")
    print(f"Total training time: {total_train_time:.2f}s")
    print(f"Total checkpoint time: {total_checkpoint_time:.2f}s")

if __name__ == "__main__":
    # ✅ **DeepSpeed 配置**
    deepspeed_config = {
        "train_batch_size": 4,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {
            "stage": 2,  # ✅ ZeRO Stage 2: 仅存储自己负责的参数
            "offload_optimizer": {"device": "cpu"},
        },
        "fp16": {"enabled": True},  # ✅ 训练时启用 FP16
    }

    # ✅ **初始化模型**
    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ✅ **Dataloader**
    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=4)

    # ✅ **Optimizer**
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ✅ **DeepSpeed 初始化**
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config_params=deepspeed_config
    )

    # ✅ **Checkpoint 目录**
    checkpoint_dir = "/pscratch/sd/k/klhhhhh/deepspeed_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ✅ **执行不同模式的训练**
    for mode in ["sync", "async"]:
        train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, mode=mode)

    # ✅ **加载最新的 Checkpoint**
    load_checkpoint(model, checkpoint_dir)
