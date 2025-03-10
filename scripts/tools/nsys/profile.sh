##ALL gpu processes are profiled
nsys profile -o async_fp16_profile \
    --trace=cuda,nvtx,osrt \
    deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200357" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "FP16" "async" &> nsys_async_io_async_FP16_output.log

##only for rank 0 gpu 0
CUDA_VISIBLE_DEVICES=0 RANK=0 nsys --capture-range=cudaProfilerApi --output=async_fp16_profile \
    --trace=cuda,nvtx,osrt \
    deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200357" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test_profile.py "FP16" "async" &> nsys_async_io_async_FP16_output.log
