srun deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200292" \
    --master_port=6510 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/multi_gpu_tensor_core.py &> output.log
