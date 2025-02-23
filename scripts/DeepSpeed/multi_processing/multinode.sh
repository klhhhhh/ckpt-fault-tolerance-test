## DeepSpeed will automatically start based on the `hostfile`, and by default, it will use all GPUs on the current node. In contrast, `torchrun` requires the `--nproc_per_node` argument to specify how many GPUs to use.  
## DeepSpeed starts on a single node and automatically launches programs on other nodes via SSH, whereas `torchrun` requires manually starting the program on each node.  
## DeepSpeed follows the legacy behavior of `torch.distributed.launch` and automatically adds a `--local_rank` argument to your program, while `torchrun` retrieves `local_rank` from the `LOCAL_RANK` environment variable.  
## Therefore, DeepSpeed essentially automates the execution of `torch.distributed.launch` on each node and sets the `--nproc_per_node` parameter according to the `hostfile`.
## And we do not use srun command, directly use the command below on master node to run the program.
deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200348" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/multi_processing/train_fp16.py &> multi_processing_FP16_output.log