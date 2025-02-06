import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import time
from concurrent.futures import ThreadPoolExecutor

# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, tokenizer, text="Hello world!", length=1000):
        tokenizer.pad_token = tokenizer.eos_token
        self.inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.inputs["input_ids"][0], self.inputs["attention_mask"][0]

# **保存 Checkpoint**
def save_partial_checkpoint(model, optimizer, checkpoint_dir, precision, use_tensor_core, accelerator, step, mode="sync", executor=None):
    """
    每个 GPU 只保存自己负责的部分参数、optimizer 状态、gradient，支持 Sync & Async，支持 FP32 & FP16，支持 CUDA Core & Tensor Core。
    """
    rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes

    # 获取完整的 state_dict
    full_model_state = model.state_dict()
    full_optimizer_state = optimizer.state_dict()

    # **按 rank 进行切分**
    model_keys = list(full_model_state.keys())
    optimizer_keys = list(full_optimizer_state["state"].keys())

    split_model_keys = model_keys[rank::world_size]
    split_optimizer_keys = optimizer_keys[rank::world_size]

    partial_model_state = {k: full_model_state[k] for k in split_model_keys}
    partial_optimizer_state = {k: full_optimizer_state["state"][k] for k in split_optimizer_keys}

    # **手动拆分梯度**
    partial_gradients = {
        k: v.grad.clone() for k, v in model.named_parameters() if k in split_model_keys and v.grad is not None
    }

    # **精度转换**
    if precision == "fp16":
        partial_model_state = {k: v.half() for k, v in partial_model_state.items()}
        partial_gradients = {k: v.half() for k, v in partial_gradients.items()}

    # **Checkpoint 路径**
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_rank_{rank}.pt")

    start_time = time.time()

    # **Sync & Async**
    if mode == "sync":
        torch.save({
            "model_state_dict": partial_model_state,
            "optimizer_state_dict": partial_optimizer_state,
            "gradients": partial_gradients,
            "rank": rank,
            "world_size": world_size,
        }, checkpoint_path)
    elif mode == "async":
        executor.submit(torch.save, {
            "model_state_dict": partial_model_state,
            "optimizer_state_dict": partial_optimizer_state,
            "gradients": partial_gradients,
            "rank": rank,
            "world_size": world_size,
        }, checkpoint_path)

    checkpoint_time = time.time() - start_time
    print(f"[GPU {rank}]  [Step {step}] Checkpoint saved at {checkpoint_path} (mode={mode}, precision={precision}, tensor_core={use_tensor_core}, time={checkpoint_time:.2f}s)")
    return checkpoint_time

# **加载 Checkpoint**
def load_full_checkpoint(model, optimizer, checkpoint_dir, accelerator):
    """
    只在 `rank 0` 进程合并所有 Checkpoint 并恢复完整模型。
    """
    rank = accelerator.state.local_process_index
    world_size = accelerator.state.num_processes

    if rank == 0:
        full_model_state = {}
        full_optimizer_state = {}
        full_gradients = {}

        start_time = time.time()

        for i in range(world_size):
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_rank_{i}.pt")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

            full_model_state.update(checkpoint["model_state_dict"])
            full_optimizer_state.update(checkpoint["optimizer_state_dict"])
            full_gradients.update(checkpoint["gradients"])

        # **加载完整模型**
        model.load_state_dict(full_model_state)

        # **恢复 optimizer**
        optimizer.load_state_dict({"state": full_optimizer_state, "param_groups": optimizer.param_groups})

        # **恢复梯度**
        for name, param in model.named_parameters():
            if name in full_gradients:
                param.grad = full_gradients[name].to(param.device)

        load_time = time.time() - start_time
        print(f"[GPU {rank}] Full checkpoint loaded successfully! Load time: {load_time:.2f}s")

# **训练 & Checkpoint**
def train_and_checkpoint(model, dataloader, optimizer, accelerator, checkpoint_dir, checkpoint_interval, mode="sync", precision="fp32", use_tensor_core=False):
    """
    训练时，每个 GPU 只存储自己负责的部分参数，支持 Sync & Async，支持 FP32 & FP16，支持 CUDA Core & Tensor Core。
    """
    executor = ThreadPoolExecutor(max_workers=1) if mode == "async" else None

    total_train_time = 0
    total_checkpoint_time = 0

    start_train_time = time.time()
    
    for step, (input_ids, attention_mask) in enumerate(dataloader):

        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        optimizer.zero_grad()
        with accelerator.autocast(dtype=torch.float16 if use_tensor_core else torch.float32):
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        if step % checkpoint_interval == 0 and step > 0:
            total_checkpoint_time += save_partial_checkpoint(model, optimizer, checkpoint_dir, precision, use_tensor_core, accelerator, step, mode, executor)

    total_train_time = time.time() - start_train_time

    if mode == "async":
        executor.shutdown(wait=True)

    if accelerator.is_local_main_process:
        print(f"Training completed. Mode: {mode}, Precision: {precision}, Tensor Core: {use_tensor_core}")
        print(f"Total training time: {total_train_time:.2f}s")
        print(f"Total checkpoint time: {total_checkpoint_time:.2f}s")

if __name__ == "__main__":
    accelerator = Accelerator(mixed_precision="fp16")

    # **初始化模型**
    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # **Dataloader**
    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=4)

    # **Optimizer**
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # **Prepare for distributed training**
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # **Checkpoint 目录**
    checkpoint_dir = "/pscratch/sd/k/klhhhhh/checkpoint_exp"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # **执行不同模式的训练**
    for mode in ["sync", "async"]:
        for precision in ["fp32", "fp16"]:
            for use_tensor_core in [False, True]:
                train_and_checkpoint(model, dataloader, optimizer, accelerator, checkpoint_dir, checkpoint_interval=10, mode=mode, precision=precision, use_tensor_core=use_tensor_core)

    # **加载完整 Checkpoint**
    load_full_checkpoint(model, optimizer, checkpoint_dir, accelerator)
