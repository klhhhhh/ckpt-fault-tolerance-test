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
from multiprocessing import Manager, Process

# ✅ Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, tokenizer, text="Hello world!", length=1000):
        tokenizer.pad_token = tokenizer.eos_token
        self.inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.inputs["input_ids"][0], self.inputs["attention_mask"][0]

# ✅ 获取 Node Rank 和 Local Rank
def get_node_rank():
    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])
    elif "WORLD_SIZE" in os.environ and "RANK" in os.environ:
        num_nodes = int(os.environ.get("WORLD_SIZE", 1)) // torch.cuda.device_count()
        return int(os.environ["RANK"]) // num_nodes
    return 0 

def get_local_rank():
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    elif "RANK" in os.environ:
        return int(os.environ["RANK"]) % torch.cuda.device_count()
    return 0

# ✅ 自定义日志格式
class LogFormatter(logging.Formatter):
    def __init__(self, precision):
        super().__init__()
        self.precision = precision

    def format(self, record):
        record.node_rank = get_node_rank()
        record.local_rank = get_local_rank()
        record.precision = self.precision  
        return super().format(record)

def setup_logging(precision="FP32"):
    node_rank = get_node_rank()
    local_rank = get_local_rank()

    my_logger = logging.getLogger(f"my_logger_{node_rank}_{local_rank}")
    my_logger.setLevel(logging.INFO)

    os.makedirs(f"./logs/multi_processing/{precision}/", exist_ok=True)
    log_filename = f"./logs/multi_processing/{precision}/my_logs_node{node_rank}_gpu{local_rank}.log"

    my_handler = logging.FileHandler(log_filename, mode="w")
    formatter = LogFormatter(precision)
    
    my_handler.setFormatter(formatter)
    my_logger.addHandler(my_handler)

    return my_logger

# ✅ 训练（不存储 Checkpoint）
def train_without_checkpoint(engine, dataloader, dtype, precision, log_lock=None, my_logger=None):
    start_train_time = time.time()

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(engine.device)
        attention_mask = attention_mask.to(engine.device)

        engine.zero_grad()
        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = engine.module(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        engine.backward(loss)
        engine.step()

    total_train_time = time.time() - start_train_time

    with log_lock:
        my_logger.info(f"Training without checkpoint completed. Precision: {precision}")
        my_logger.info(f"Total training time (No Checkpoint): {total_train_time:.2f}s")
        my_logger.handlers[0].flush()

# ✅ DeepSpeed Checkpoint Load
def load_checkpoint(engine, checkpoint_dir, precision, log_lock=None, my_logger=None):
    rank = engine.local_rank
    start_time = time.time()

    latest_ckpt = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    engine.load_checkpoint(checkpoint_path)

    load_time = time.time() - start_time

    with log_lock:
        my_logger.info(f"[GPU {rank}] Checkpoint loaded from {checkpoint_path} "
                       f"(Precision: {precision}, Load Time: {load_time:.2f}s)")
        my_logger.handlers[0].flush()

    return load_time

# ✅ 进程异步存储 Checkpoint
def save_checkpoint_process(state_dict, checkpoint_path):
    """独立进程执行 DeepSpeed Checkpoint 存储"""
    torch.save(state_dict, checkpoint_path)

def save_checkpoint_async(engine, checkpoint_dir, step):
    """创建一个独立进程来存储 DeepSpeed Checkpoint"""
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
    
    state_dict = engine.state_dict()  # ✅ 只传 state_dict，避免 pickle 问题
    
    p = Process(target=save_checkpoint_process, args=(state_dict, checkpoint_path))
    p.start()
    
    return p

# ✅ 训练 + Checkpoint 逻辑
def train_and_checkpoint(engine, dataloader, checkpoint_dir, checkpoint_interval, dtype, precision, mode="sync", log_lock=None, my_logger=None):
    checkpoint_processes = []

    start_train_time = time.time()
    total_checkpoint_time = 0

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(engine.device)
        attention_mask = attention_mask.to(engine.device)

        engine.zero_grad()
        with torch.cuda.amp.autocast(dtype=dtype):
            outputs = engine.module(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        engine.backward(loss)
        engine.step()

        if step % checkpoint_interval == 0 and step > 0:
            checkpoint_start = time.time()
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")

            if mode == "async":
                p = save_checkpoint_async(engine, checkpoint_dir, step)
                checkpoint_processes.append(p)
            else:
                save_checkpoint_process(engine.state_dict(), checkpoint_path)

            checkpoint_time = time.time() - checkpoint_start
            total_checkpoint_time += checkpoint_time

            rank = engine.local_rank

            with log_lock:
                my_logger.info(f"[GPU {rank}] [Step {step}] Checkpoint saved at {checkpoint_path} "
                               f"(mode={mode}, Precision: {precision}, time={checkpoint_time:.2f}s)")
                my_logger.handlers[0].flush()

    total_train_time = time.time() - start_train_time

    with log_lock:
        my_logger.info(f"Training completed. Mode: {mode}, Precision: {precision}")
        my_logger.info(f"Total training time: {total_train_time:.2f}s")
        my_logger.info(f"Total checkpoint time: {total_checkpoint_time:.2f}s")
        my_logger.handlers[0].flush()

    # ✅ 等待所有异步 checkpoint 进程完成
    for p in checkpoint_processes:
        p.join()

if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    manager = Manager()
    log_lock = manager.Lock()

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
        },
        "logging": {"verbosity": 0},
        "bf16": {"enabled": False},
        "fp16": {"enabled": False}
    }

    for precision, dtype in [("BF16", torch.bfloat16), ("FP16", torch.float16), ("FP32", torch.float32)]:
        my_logger = setup_logging(precision)
        deepspeed_config["fp16"]["enabled"] = precision == "FP16"
        deepspeed_config["bf16"]["enabled"] = precision == "BF16"
        model, optimizer, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config_params=deepspeed_config)

        train_without_checkpoint(model, dataloader, dtype, precision, log_lock, my_logger)

        for mode in ["sync", "async"]:
            train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, dtype=dtype, precision=precision, mode=mode, log_lock=log_lock, my_logger=my_logger)

        load_checkpoint(model, checkpoint_dir, precision, log_lock, my_logger)

    if dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0)
