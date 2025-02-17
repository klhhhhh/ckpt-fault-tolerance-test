import os
import sys
import torch
import torch.distributed as dist
import deepspeed
import logging
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import time
from multiprocessing import Manager


# ✅ 自定义日志格式，添加 `NODE_RANK`、`LOCAL_RANK`、`PRECISION`
class LogFormatter(logging.Formatter):
    def __init__(self, precision):
        super().__init__()
        self.precision = precision

    def format(self, record):
        record.node_rank = get_node_rank()
        record.local_rank = get_local_rank()
        record.precision = self.precision  # 添加当前精度信息
        return super().format(record)

def setup_logging(precision="FP32"):
    """设置日志系统，确保日志不会被多个 GPU 进程截断，并且包含精度信息"""
    node_rank = get_node_rank()
    local_rank = get_local_rank()

    my_logger = logging.getLogger(f"my_logger_{node_rank}_{local_rank}")
    my_logger.setLevel(logging.INFO)

    os.makedirs("./logs", exist_ok=True)
    log_filename = f"./logs/my_logs_node{node_rank}_gpu{local_rank}.log"

    my_handler = logging.FileHandler(log_filename, mode="w")
    log_format = "%(asctime)s - Node: %(node_rank)s - GPU: %(local_rank)s - Precision: %(precision)s - %(levelname)s - %(message)s"
    formatter = LogFormatter(precision)
    
    my_handler.setFormatter(formatter)
    my_logger.addHandler(my_handler)

    return my_logger

# ✅ DeepSpeed Checkpoint Save
def save_checkpoint(engine, checkpoint_dir, step, precision, mode="sync", executor=None, log_lock=None, my_logger=None):
    rank = engine.local_rank
    start_time = time.time()
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")

    if mode == "sync":
        engine.save_checkpoint(checkpoint_path)
    elif mode == "async":
        executor.submit(engine.save_checkpoint, checkpoint_path)

    checkpoint_time = time.time() - start_time

    # ✅ 保护日志写入，防止行被截断
    with log_lock:
        my_logger.info(f"[GPU {rank}] [Step {step}] Checkpoint saved at {checkpoint_path} "
                       f"(mode={mode}, Precision: {precision}, time={checkpoint_time:.2f}s)")
        my_logger.handlers[0].flush()

    return checkpoint_time

# ✅ DeepSpeed Checkpoint Load (增加加载时间测试)
def load_checkpoint(engine, checkpoint_dir, precision, log_lock=None, my_logger=None):
    """DeepSpeed 自动恢复模型参数、优化器状态和梯度，并测量加载时间"""
    rank = engine.local_rank
    start_time = time.time()

    latest_ckpt = sorted(os.listdir(checkpoint_dir))[-1]  # 选择最新的 Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    engine.load_checkpoint(checkpoint_path)

    load_time = time.time() - start_time

    # ✅ 保护日志写入，防止行被截断
    with log_lock:
        my_logger.info(f"[GPU {rank}] Checkpoint loaded from {checkpoint_path} "
                       f"(Precision: {precision}, Load Time: {load_time:.2f}s)")
        my_logger.handlers[0].flush()

    return load_time

# ✅ Training & Checkpointing
def train_and_checkpoint(engine, dataloader, checkpoint_dir, checkpoint_interval, dtype, precision, mode="sync", log_lock=None, my_logger=None):
    executor = ThreadPoolExecutor(max_workers=1) if mode == "async" else None
    total_train_time = 0
    total_checkpoint_time = 0

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

        if step % checkpoint_interval == 0 and step > 0:
            total_checkpoint_time += save_checkpoint(engine, checkpoint_dir, step, precision, mode, executor, log_lock, my_logger)

    total_train_time = time.time() - start_train_time

    if mode == "async":
        executor.shutdown(wait=True)

    # ✅ 保护日志写入，防止行被截断
    with log_lock:
        my_logger.info(f"Training completed. Mode: {mode}, ZeRO-3 Enabled, Precision: {precision}, Tensor Core: Enabled")
        my_logger.info(f"Total training time: {total_train_time:.2f}s")
        my_logger.info(f"Total checkpoint time: {total_checkpoint_time:.2f}s")
        my_logger.handlers[0].flush()


if __name__ == "__main__":
    # ✅ 在 `if __name__ == "__main__"` 里创建锁，支持多进程共享
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

        for mode in ["sync", "async"]:
            train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, dtype=dtype, precision=precision, mode=mode, log_lock=log_lock, my_logger=my_logger)

        # ✅ **新增：训练后立即加载 Checkpoint 并测量时间**
        load_time = load_checkpoint(model, checkpoint_dir, precision, log_lock, my_logger)

        # ✅ **日志记录**
        with log_lock:
            my_logger.info(f"Final Checkpoint Load Time: {load_time:.2f}s")
            my_logger.handlers[0].flush()

    if dist.is_initialized():
        dist.destroy_process_group()

    sys.exit(0)
