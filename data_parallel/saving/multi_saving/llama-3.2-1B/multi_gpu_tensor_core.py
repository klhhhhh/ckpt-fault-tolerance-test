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

# 禁止 Hugging Face 的 Tokenizer 并行，防止 `fork()` 错误
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ✅ **获取 `NODE_RANK`**
def get_node_rank():
    """获取 DeepSpeed / torch.distributed 的 node rank"""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", -1))  # `RANK` 代表当前节点编号


# ✅ **日志配置**
class LogFormatter(logging.Formatter):
    """自定义日志格式，添加 `NODE_RANK`"""
    def format(self, record):
        record.node_rank = get_node_rank()
        return super().format(record)


def setup_logging():
    """设置日志系统，确保日志带有 `NODE_RANK`"""
    my_logger = logging.getLogger("my_logger")
    my_logger.setLevel(logging.INFO)

    # 日志格式：带 `NODE_RANK`
    log_format = "%(asctime)s - Node: %(node_rank)s - %(levelname)s - %(message)s"
    formatter = LogFormatter(log_format)

    # 日志写入文件
    my_handler = logging.FileHandler("my_logs_tensor_core.log", mode="w")
    my_handler.setFormatter(formatter)
    my_logger.addHandler(my_handler)

    return my_logger

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
    rank = engine.local_rank
    start_time = time.time()
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}")

    if mode == "sync":
        engine.save_checkpoint(checkpoint_path)
    elif mode == "async":
        executor.submit(engine.save_checkpoint, checkpoint_path)

    checkpoint_time = time.time() - start_time
    my_logger.info(f"[GPU {rank}] [Step {step}] Checkpoint saved at {checkpoint_path} (mode={mode}, time={checkpoint_time:.2f}s)")
    return checkpoint_time

# ✅ **DeepSpeed Checkpoint Load**
def load_checkpoint(engine, checkpoint_dir, mode):
    rank = engine.local_rank
    start_time = time.time()
    latest_ckpt = sorted(os.listdir(checkpoint_dir))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_ckpt)
    engine.load_checkpoint(checkpoint_path)
    load_time = time.time() - start_time
    my_logger.info(f"[GPU {rank}] {mode} Checkpoint loaded from {checkpoint_path} in {load_time:.2f}s")

# ✅ **Training & Checkpointing**
def train_and_checkpoint(engine, dataloader, checkpoint_dir, checkpoint_interval, dtype, mode="sync"):
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
            total_checkpoint_time += save_checkpoint(engine, checkpoint_dir, step, mode, executor)

    total_train_time = time.time() - start_train_time

    if mode == "async":
        executor.shutdown(wait=True)

    my_logger.info(f"Training completed. Mode: {mode}, ZeRO-3 Enabled, Tensor Core: Enabled")
    my_logger.info(f"Total training time: {total_train_time:.2f}s")
    my_logger.info(f"Total checkpoint time: {total_checkpoint_time:.2f}s")

if __name__ == "__main__":
    # ✅ **设置日志**
    my_logger = setup_logging()

    # ✅ **DeepSpeed Configuration**
    deepspeed_config = {
        "train_batch_size": 512,
        "train_micro_batch_size_per_gpu": 16,
        "gradient_accumulation_steps": 2,
        "zero_optimization": {
            "stage": 3,
            "contiguous_gradients": True,
        },
        "logging": {"verbosity": 0},
        "bf16": {"enabled": False},
        "fp16": {"enabled": False}
    }

    # ✅ **Initialize Model**
    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # ✅ **Dataloader**
    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=16)

    # ✅ **Optimizer**
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # ✅ **Checkpoint Directory**
    checkpoint_dir = "/pscratch/sd/k/klhhhhh/deepspeed_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if get_node_rank() == 0:
        my_logger.info("Percision: BF16, ZeRO-3: Enabled, Tensor Core: Enabled, Training started.")
    
    deepspeed_config["fp16"]["enabled"] = False
    deepspeed_config["bf16"]["enabled"] = True
    # ✅ **DeepSpeed Initialization**
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config_params=deepspeed_config
    )
    
    for mode in ["sync", "async"]:
        # ✅ **Training & Checkpointing**
        train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, dtype=torch.bfloat16, mode=mode)

        # ✅ **Load the Latest Checkpoint**
        load_checkpoint(model, checkpoint_dir)
    
    if get_node_rank() == 0:
        my_logger.info("Percision: FP16, ZeRO-3: Enabled, Tensor Core: Enabled, Training started.")
    
    deepspeed_config["fp16"]["enabled"] = True
    deepspeed_config["bf16"]["enabled"] = False
    # ✅ **DeepSpeed Initialization**
    model, optimizer, _, _ = deepspeed.initialize(
        model=model, optimizer=optimizer, config_params=deepspeed_config
    )

    for mode in ["sync", "async"]:
        # ✅ **Training & Checkpointing**
        train_and_checkpoint(model, dataloader, checkpoint_dir, checkpoint_interval=10, dtype=torch.float16, mode=mode)

        # ✅ **Load the Latest Checkpoint**
        load_checkpoint(model, checkpoint_dir)
    

    # ✅ **清理 PyTorch 分布式进程**
    if dist.is_initialized():
        dist.destroy_process_group()

    # ✅ **强制退出**
    sys.exit(0)