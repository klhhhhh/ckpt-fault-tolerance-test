import os
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import time

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

# Save checkpoint function
def save_checkpoint(model, optimizer, step, checkpoint_dir):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        # 保存梯度信息
        "gradients": {name: param.grad.clone() for name, param in model.named_parameters() if param.grad is not None},
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step}: {checkpoint_path}")

# Load checkpoint function
def load_checkpoint(model, optimizer, checkpoint_path):
    start_time = time.time()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint["step"]
    
    # 恢复梯度
    if "gradients" in checkpoint:
        for name, param in model.named_parameters():
            if name in checkpoint["gradients"]:
                param.grad = checkpoint["gradients"][name].to(param.device)

    load_time = time.time() - start_time
    print(f"Checkpoint loaded from {checkpoint_path} in {load_time:.2f}s")
    return step, load_time

# Training function
def train_and_checkpoint(model, dataloader, optimizer, accelerator, checkpoint_dir, checkpoint_interval):
    total_steps = 100  # Example total steps for training
    for step, (input_ids, attention_mask) in enumerate(dataloader):
        if step >= total_steps:
            break

        input_ids = input_ids.to(accelerator.device)
        attention_mask = attention_mask.to(accelerator.device)

        optimizer.zero_grad()

        with accelerator.autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        if step % checkpoint_interval == 0 and step > 0:
            save_checkpoint(model, optimizer, step, checkpoint_dir)

    print("Training completed.")

if __name__ == "__main__":
    accelerator = Accelerator(mixed_precision="fp16")

    # Initialize tokenizer and model
    model_name = "/pscratch/sd/k/klhhhhh/Huggingface_model/Llama-3.2-1B"  # Replace with actual path
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Dummy dataset and dataloader
    dataset = DummyDataset(tokenizer, length=2000)
    dataloader = DataLoader(dataset, batch_size=4)

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Prepare model and dataloader for distributed training
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Checkpoint directory
    checkpoint_dir = "/pscratch/sd/k/klhhhhh/checkpoint_exp"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Train and save checkpoints
    train_and_checkpoint(
        model,
        dataloader,
        optimizer,
        accelerator,
        checkpoint_dir,
        checkpoint_interval=10,
    )

    # Load a checkpoint and measure time
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step_10.pt")
    if os.path.exists(checkpoint_path):
        step, load_time = load_checkpoint(model, optimizer, checkpoint_path)
        print(f"Resumed from step {step}.")
