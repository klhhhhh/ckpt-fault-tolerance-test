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


# Define checkpoint saving function (CUDA Core or Tensor Core)
def save_checkpoint(model, checkpoint_path, precision="fp32", use_tensor_core=False):
    """
    Save checkpoint using CUDA Core or Tensor Core.
    Args:
        model: The model to save.
        checkpoint_path: Path to save the checkpoint.
        precision: "fp16" or "fp32".
        use_tensor_core: If True, use Tensor Core; otherwise, use CUDA Core.
    """
    if precision == "fp16":
        state_dict = {k: v.half() for k, v in model.state_dict().items()}
    else:
        state_dict = {k: v.float() for k, v in model.state_dict().items()}

    if use_tensor_core:
        # Force Tensor Core usage during save
        with torch.cuda.amp.autocast(dtype=torch.float16):
            torch.save(state_dict, checkpoint_path)
    else:
        torch.save(state_dict, checkpoint_path)


# Asynchronous checkpoint saving
def save_checkpoint_async(model, checkpoint_path, executor, precision="fp32", use_tensor_core=False):
    future = executor.submit(save_checkpoint, model, checkpoint_path, precision, use_tensor_core)
    return future


# Training and checkpoint testing function
def train_and_checkpoint(model, dataloader, checkpoint_interval, checkpoint_path, precision, mode="sync", use_tensor_core=False, use_ckpt=True, accelerator=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    total_time = 0
    start_time = time.time()
    executor = ThreadPoolExecutor(max_workers=1) if mode == "async" else None

    for step, (input_ids, attention_mask) in enumerate(dataloader):
        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        optimizer.zero_grad()
        with accelerator.autocast():  # Mixed precision
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        # Checkpoint
        if step % checkpoint_interval == 0 and step > 0 and use_ckpt:
            checkpoint_start = time.time()
            if mode == "sync":
                save_checkpoint(model, checkpoint_path, precision, use_tensor_core)
            elif mode == "async":
                save_checkpoint_async(model, checkpoint_path, executor, precision, use_tensor_core)
            checkpoint_end = time.time()
            if accelerator.is_local_main_process:
                print(f"Step {step}: Checkpoint ({'Tensor Core' if use_tensor_core else 'CUDA Core'}) {precision.upper()} {mode} time: {checkpoint_end - checkpoint_start:.2f}s")
            total_time += checkpoint_end - checkpoint_start

    end_time = time.time()
    if mode == "async":
        executor.shutdown(wait=True)

    if accelerator.is_local_main_process:
        print(f"Total training time ({mode} mode, {'Tensor Core' if use_tensor_core else 'CUDA Core'}, {precision.upper()}): {end_time - start_time:.2f}s")
        print(f"Total checkpoint time ({mode} mode, {'Tensor Core' if use_tensor_core else 'CUDA Core'}, {precision.upper()}): {total_time:.2f}s")


if __name__ == "__main__":
    accelerator = Accelerator(mixed_precision="fp16")  # Enable mixed precision

    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"  # Replace with actual path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=4)
    model, dataloader = accelerator.prepare(model, dataloader)

    if accelerator.is_local_main_process:
        print("\n===== Do not use checkpoint=====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/sync_cuda_fp16.pt",
        precision="fp16",
        mode="sync",
        use_tensor_core=False,
        use_ckpt=False,
        accelerator=accelerator,
    )

    if accelerator.is_local_main_process:
        print("\n===== Do not use checkpoint=====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/sync_cuda_fp16.pt",
        precision="fp16",
        mode="sync",
        use_tensor_core=True,
        use_ckpt=False,
        accelerator=accelerator,
    )

    #CUDA Core (FP32)
    if accelerator.is_local_main_process:
        print("===== Synchronous Checkpoint Test (CUDA Core, FP32) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/sync_cuda_fp32.pt",
        precision="fp32",
        mode="sync",
        use_tensor_core=False,
        accelerator=accelerator,
    )

    # Tensor Core (FP32)
    if accelerator.is_local_main_process:
        print("\n===== Synchronous Checkpoint Test (Tensor Core, FP32) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/async_tensor_fp32.pt",
        precision="fp32",
        mode="sync",
        use_tensor_core=True,
        accelerator=accelerator,
    )

        # CUDA Core (FP32)
    if accelerator.is_local_main_process:
        print("===== Asynchronous Checkpoint Test (CUDA Core, FP32) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/sync_cuda_fp32.pt",
        precision="fp32",
        mode="async",
        use_tensor_core=False,
        accelerator=accelerator,
    )

    # Tensor Core (FP32)
    if accelerator.is_local_main_process:
        print("\n===== Asynchronous Checkpoint Test (Tensor Core, FP32) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/async_tensor_fp32.pt",
        precision="fp32",
        mode="async",
        use_tensor_core=True,
        accelerator=accelerator,
    )

    # CUDA Core (FP16)
    if accelerator.is_local_main_process:
        print("\n===== Synchronous Checkpoint Test (CUDA Core, FP16) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/sync_cuda_fp16.pt",
        precision="fp16",
        mode="sync",
        use_tensor_core=False,
        accelerator=accelerator,
    )

    # Tensor Core (FP16)
    if accelerator.is_local_main_process:
        print("\n===== Synchronous Checkpoint Test (Tensor Core, FP16) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/async_tensor_fp16.pt",
        precision="fp16",
        mode="sync",
        use_tensor_core=True,
        accelerator=accelerator,
    )

    # CUDA Core (FP16)
    if accelerator.is_local_main_process:
        print("\n===== Asynchronous Checkpoint Test (CUDA Core, FP16) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/sync_cuda_fp16.pt",
        precision="fp16",
        mode="async",
        use_tensor_core=False,
        accelerator=accelerator,
    )

    # Tensor Core (FP16)
    if accelerator.is_local_main_process:
        print("\n===== Asynchronous Checkpoint Test (Tensor Core, FP16) =====")
    train_and_checkpoint(
        model,
        dataloader,
        checkpoint_interval=10,
        checkpoint_path="/pscratch/sd/k/klhhhhh/checkpoint_exp/async_tensor_fp16.pt",
        precision="fp16",
        mode="async",
        use_tensor_core=True,
        accelerator=accelerator,
    )
