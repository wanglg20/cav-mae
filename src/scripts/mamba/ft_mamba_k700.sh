export CUDA_VISIBLE_DEVICES=0,1,2,3

model=CrossMambaFT
ftmode=multi

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
pretrain_path=/data/wanglinge/project/cav-mae/src/exp/trainmamba-k700-CrossMamba-lr5e-5-bs50-c0.01-p10.0-m0.75/models/model.25.pth
freeze_base=True
head_lr=100 # newly initialized ft layers uses 100 times larger than the base lr

bal=None
lr=5e-5
epoch=30
lrscheduler_start=5
lrscheduler_decay=0.5
lrscheduler_step=1
wa=True
wa_start=3
wa_end=15
lr_adapt=True              # Set to True if you are not doing Recurrence
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=False
freqm=48
timem=192
mixup=0.0
batch_size=12
label_smooth=0.1

dataset=k700
dataset_split=k700_train_valid
tr_data=/data/wanglinge/project/cav-mae/src/data/info/k700/${dataset_split}.json
te_data=/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid_1.json
label_csv=/data/wanglinge/project/cav-mae/src/data/info/k700/k700_class.csv
cd /data/wanglinge/project/cav-mae/src
# Clear Python cache to avoid parameter mismatch issues
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
exp_dir=./exp/mamba_ft_${dataset_split}_freeze{$freeze_base}_lr_${lr}-bs${batch_size}-h${head_lr}
mkdir -p $exp_dir
train_frame=/data/wanglinge/dataset/k700/frames_16
val_frame=/data/wanglinge/dataset/k700/frames_16

PYTHONWARNINGS=ignore torchrun --master_port=29505 --nproc_per_node=4 ./run_finetune_mamba.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 700 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
--label_smooth ${label_smooth} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
--loss CE --metrics acc --warmup True \
--wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
--pretrain_path ${pretrain_path} --ftmode ${ftmode} \
--freeze_base ${freeze_base} --head_lr ${head_lr} \
--num-workers 4 --pooling \
--raw_data k700 --train_frame_root ${train_frame} --val_frame_root ${val_frame} \
# --use_wandb --wandb_run_name mamba_ft_k700_algined_dataset\
# --resume --wandb_id  oasix4fx\