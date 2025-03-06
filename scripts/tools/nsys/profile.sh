nsys profile -o async_fp16_profile.txt deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200476" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "FP16" "async" &> nsys_async_io_async_FP16_output.log