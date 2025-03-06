#fp16
deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200357" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "FP16" "async" &> async_io_async_FP16_output.log

deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200440" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "FP16" "sync"&> async_io_sync_FP16_output.log

#bf16
deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200357" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "BF16" "async" &> async_io_async_BF16_output.log

deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200440" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "BF16" "sync"&> async_io_sync_BF16_output.log

#fp32
deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200357" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "FP32" "async" &> async_io_async_FP32_output.log

deepspeed --num_nodes=4 --num_gpus=4 \
    --master_addr="nid200440" \
    --master_port=29501 \
    --hostfile="/global/homes/k/klhhhhh/ckpt-fault-tolerance-test/scripts/hostfile" \
    /global/homes/k/klhhhhh/ckpt-fault-tolerance-test/data_parallel/saving/multi_saving/llama-3.2-1B/async_io/train_test.py "FP32" "sync"&> async_io_sync_F32_output.log
