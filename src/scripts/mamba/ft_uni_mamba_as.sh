export CUDA_VISIBLE_DEVICES=0,1
model=UniMamba
ftmode=multi
cur_dir=$(pwd)
pretrain_path=/home/chenyingying/tmp/cav-mae/src/exp/audio_mamba-as2M-UniMamba-lr5e-5-bs400-p10.0-m0.75/models/model.2.pth
freeze_base=False
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

dataset=audioset
tr_data=/data/wanglinge/project/cav-mae/src/data/info/as/data/balanced_train_segments_valid.json
te_data=/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments_valid.json
label_csv=/data/wanglinge/project/cav-mae/src/data/info/as/data/as_label.csv
cd /home/chenyingying/tmp/cav-mae/src

exp_dir=./exp/audio_mamba_ft-as200k-${model}-lr${lr}-bs${batch_size}-p${ts_loss_weight}-m${masking_ratio}
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
--raw_data audioset --train_frame_root ${train_frame} --val_frame_root ${val_frame} \
--modality audio \
# --use_wandb --wandb_run_name audio_mamba_ft_as200k \