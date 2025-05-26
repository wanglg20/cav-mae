import subprocess
import time
import os

# 要监控的 GPU ID
gpu_ids = [4, 5, 6, 7]
check_interval = 30  # 每30秒检查一次
memory_threshold_mb = 1024  # 显存使用小于1GB视为可用

# 检查指定GPU显存是否全部低于阈值
def all_gpus_available(gpu_ids, threshold):
    try:
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader']
        )
        mem_usage = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        for gid in gpu_ids:
            if mem_usage[gid] >= threshold:
                return False
        return True
    except Exception as e:
        print(f"检测GPU状态时出错: {e}")
        return False

# 训练命令
train_cmd = """
export CUDA_VISIBLE_DEVICES=4,5,6,7 && \
cd /data/wanglinge/project/cav-mae/src && \
exp_dir=./exp/trainmae-busy-k700-cav-mae-sync-lr5e-5-bs60-normFalse-c0.01-p1.0-tpFalse-mr-unstructured-0.75 && \
mkdir -p $exp_dir && \
PYTHONWARNINGS=ignore torchrun --nproc_per_node=4 run_cavmae_pretrain.py --model cav-mae-sync --dataset audioset \
--data-train /data/wanglinge/project/cav-mae/src/data/info/k700_train.json \
--data-val /data/wanglinge/project/cav-mae/src/data/info/k700_val.json \
--exp-dir $exp_dir --label-csv /data/wanglinge/project/cav-mae/src/data/info/k700_class.csv \
--n_class 700 --lr 5e-5 --n-epochs 45 --batch-size 60 --save_model True --mixup 0.0 --bal None \
--lrscheduler_start 10 --lrscheduler_decay 0.5 --lrscheduler_step 5 --dataset_mean -5.081 --dataset_std 4.4849 \
--target_length 1000 --noise True --warmup True --lr_adapt False --norm_pix_loss False \
--pretrain_path /data/wanglinge/project/cav-mae/src/weight/init/ori_mae_11.pth \
--mae_loss_weight 1.0 --contrast_loss_weight 0.01 --tr_pos False --masking_ratio 0.75 \
--mask_mode unstructured
"""

# 主循环
print(f"开始监控GPU {gpu_ids}，显存低于 {memory_threshold_mb}MB 时启动训练...")
while True:
    if all_gpus_available(gpu_ids, memory_threshold_mb):
        print("检测到所有指定GPU显存使用低于阈值，开始训练...")
        subprocess.call(train_cmd, shell=True)
        break
    else:
        print("GPU仍在使用中，等待中...")
        time.sleep(check_interval)
