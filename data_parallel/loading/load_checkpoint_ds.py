import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from deepspeed.pipe import PipelineModule
import deepspeed
import time

# 定义数据集
class DummyDataset(Dataset):
    def __init__(self, tokenizer, text="Hello world!", length=1000):
        tokenizer.pad_token = tokenizer.eos_token
        self.inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.inputs["input_ids"][0], self.inputs["attention_mask"][0]

# 定义模型流水线阶段
class Stage1(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, x, attention_mask=None, labels=None):
        return self.model(x, attention_mask=attention_mask, labels=labels)

class Stage2(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, hidden_states):
        return self.model.lm_head(hidden_states)

# 构建流水线模型
def get_pipeline_model(model_name, pipeline_parallel_size):
    stages = [Stage1(model_name), Stage2(model_name)]
    model = PipelineModule(layers=stages, loss_fn=torch.nn.CrossEntropyLoss(), num_stages=pipeline_parallel_size)
    return model

# DeepSpeed 配置
deepspeed_config = {
    "train_batch_size": 8,
    "fp16": {"enabled": True},
    "zero_optimization": {
        "stage": 3,  # Zero Redundancy Optimizer Stage 3
        "offload_optimizer": {"device": "cpu"},
        "offload_param": {"device": "cpu"},
    },
    "pipeline": {
        "enable": True,
        "stages": 2  # 流水线并行阶段数
    },
    "tensor_parallel": {
        "size": 4  # 张量并行大小
    },
}

# 保存 checkpoint
def save_checkpoint(engine, checkpoint_path):
    """保存模型参数、优化器状态和梯度到 checkpoint"""
    if engine.global_rank == 0:
        print(f"Saving checkpoint to {checkpoint_path}")
    engine.save_checkpoint(checkpoint_path)

# 加载 checkpoint 并测量加载时间
def load_checkpoint(engine, checkpoint_path):
    """加载 checkpoint 并测量时间"""
    start_time = time.time()
    success, _, _ = engine.load_checkpoint(checkpoint_path)
    end_time = time.time()

    if engine.global_rank == 0:
        if success:
            print(f"Checkpoint loaded successfully from {checkpoint_path}")
        else:
            print(f"Failed to load checkpoint from {checkpoint_path}")
        print(f"Checkpoint load time: {end_time - start_time:.2f}s")

# 训练函数
def train_model(engine, dataloader, num_epochs, checkpoint_interval, checkpoint_path):
    for epoch in range(num_epochs):
        for step, (input_ids, attention_mask) in enumerate(dataloader):
            input_ids = input_ids.to(engine.local_rank)
            attention_mask = attention_mask.to(engine.local_rank)

            loss = engine(input_ids, attention_mask=attention_mask)
            engine.backward(loss)
            engine.step()

            if step % checkpoint_interval == 0 and step > 0:
                save_checkpoint(engine, checkpoint_path)

        # 加载 checkpoint 以验证功能和加载时间
        if epoch == 0 and step == checkpoint_interval:
            load_checkpoint(engine, checkpoint_path)

# 主函数
if __name__ == "__main__":
    # 配置模型和数据
    model_name = "/path/to/llama"  # 替换为实际模型路径
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    dataset = DummyDataset(tokenizer, length=5000)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 初始化模型
    model = get_pipeline_model(model_name, pipeline_parallel_size=2)

    # 初始化 DeepSpeed
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        config_params=deepspeed_config,
        model_parameters=model.parameters()
    )

    # 训练模型并保存 checkpoint
    train_model(
        engine,
        dataloader,
        num_epochs=2,
        checkpoint_interval=50,
        checkpoint_path="/path/to/checkpoints"
    )
