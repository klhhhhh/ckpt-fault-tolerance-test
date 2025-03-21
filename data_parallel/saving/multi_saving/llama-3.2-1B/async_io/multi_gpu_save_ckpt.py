import os
import sys
import torch
import torch.distributed as dist
import deepspeed
import logging
import time
from multiprocessing import Manager
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

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

def setup_logging(precision="FP32", mode="sync"):
    node_rank = get_node_rank()
    local_rank = get_local_rank()

    my_logger = logging.getLogger(f"my_logger_{node_rank}_{local_rank}")
    my_logger.setLevel(logging.INFO)

    os.makedirs(f"./logs/async_io/{mode}_{precision}/", exist_ok=True)
    log_filename = f"./logs/async_io/{mode}_{precision}/my_logs_node{node_rank}_gpu{local_rank}.log"

    my_handler = logging.FileHandler(log_filename, mode="w")
    formatter = LogFormatter(precision)
    
    my_handler.setFormatter(formatter)
    my_logger.addHandler(my_handler)

    return my_logger

# ✅ 训练（不存储 Checkpoint）
def train_without_checkpoint(engine, dataloader, dtype, precision, my_logger=None):
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


    my_logger.info(f"Training without checkpoint completed. Precision: {precision}")
    my_logger.info(f"Total training time (No Checkpoint): {total_train_time:.2f}s")
    my_logger.handlers[0].flush()

# ✅ 训练 + Checkpoint 逻辑
def train_and_checkpoint(engine, dataloader, checkpoint_dir, checkpoint_interval, dtype, precision, my_logger=None):
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

        # ✅ 触发 Checkpoint
        if step % checkpoint_interval == 0 and step > 0:
            checkpoint_start = time.time()
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")
            
            # ✅ 直接调用 DeepSpeed 的 save_checkpoint（内部已支持 async_io）
            engine.save_checkpoint(checkpoint_path)

            checkpoint_time = time.time() - checkpoint_start
            total_checkpoint_time += checkpoint_time

            rank = engine.local_rank

            my_logger.info(f"[GPU {rank}] [Step {step}] Checkpoint saved at {checkpoint_path} "
                            f"(Precision: {precision}, time={checkpoint_time:.2f}s)")
            my_logger.handlers[0].flush()

    total_train_time = time.time() - start_train_time


    my_logger.info(f"Training completed. Precision: {precision}")
    my_logger.info(f"Total training time: {total_train_time:.2f}s")
    my_logger.info(f"Total checkpoint time: {total_checkpoint_time:.2f}s")
    my_logger.handlers[0].flush()

# ✅  DeepSpeed Checkpoint Load
def load_checkpoint(engine, checkpoint_dir, precision, my_logger=None):
    rank = engine.local_rank
    start_time = time.time()

    latest_ckpt = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    engine.load_checkpoint(checkpoint_path)

    load_time = time.time() - start_time

    my_logger.info(f"[GPU {rank}] Checkpoint loaded from {checkpoint_path} "
                    f"(Precision: {precision}, Load Time: {load_time:.2f}s)")
    my_logger.handlers[0].flush()

    return load_time
