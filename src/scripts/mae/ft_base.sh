export CUDA_VISIBLE_DEVICES=0,1

model=cav-mae-ft
ftmode=multimodal

# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
pretrain_path=/data/wanglinge/project/cav-mae/src/exp/trainmae-audioset-cav-mae-lr5e-5-bs60-normTrue-c0.01-p1.0-tpFalse-mr-unstructured-0.75/models/audio_model.9.pth

freeze_base=False
head_lr=100 # newly initialized ft layers uses 100 times larger than the base lr

bal=None
lr=5e-5
epoch=25
lrscheduler_start=5
lrscheduler_decay=0.5
lrscheduler_step=1
wa=False
wa_start=3
wa_end=15
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.0
batch_size=30
label_smooth=0.1

dataset=audioset
tr_data=/data/wanglinge/project/cav-mae/src/data/info/k700_val.json
te_data=/data/wanglinge/project/cav-mae/src/data/info/k700_val.json
label_csv=/data/wanglinge/project/cav-mae/src/data/info/k700_class.csv
exp_dir=./exp/mae_ft-all_frames-norm_${lr}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
cd /data/wanglinge/project/cav-mae/src
mkdir -p $exp_dir


PYTHONWARNINGS=ignore torchrun --nproc_per_node=2 run_cavmae_ft.py --model ${model} --dataset ${dataset} \
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
--num-workers 3 --pooling \
--use_wandb