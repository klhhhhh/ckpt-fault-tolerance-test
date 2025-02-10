deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200421" \
    --master_port=29500 \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/multi_gpu_tensor_core.py
