export CUDA_VISIBLE_DEVICES=1,2,3,4
model=UniMamba
masking_ratio=0.75
contrastive_loss_weight=0.01
ts_loss_weight=10.0
bal=None
lr=5e-5
epoch=25    
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=False
mixup=0.0
batch_size=400
lr_adapt=False
dataset=audioset
tr_data=/data/wanglinge/project/cav-mae/src/data/info/as/data/unbalanced_train_segments_valid.json
te_data=/data/wanglinge/project/cav-mae/src/data/info/as/data/eval_segments_valid.json
label_csv=/data/wanglinge/project/cav-mae/src/data/info/as/data/as_label.csv
cd /home/chenyingying/tmp/cav-mae/src
clap_path=/data/wanglinge/project/cav-mae/src/weight/teacher/clap.pth
exp_dir=./exp/audio_mamba-as2M-${model}-lr${lr}-bs${batch_size}-p${ts_loss_weight}-m${masking_ratio}
mkdir -p $exp_dir


train_frame=/data/wanglinge/dataset/k700/frames_16
val_frame=/data/wanglinge/dataset/k700/frames_16


PYTHONWARNINGS=ignore torchrun --nproc_per_node=4 run_pretrain_mamba.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 700 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--mixup ${mixup} --bal ${bal} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--lr_adapt ${lr_adapt} \
--ts_loss_weight ${ts_loss_weight} --contrastive_loss_weight ${contrastive_loss_weight}  --masking_ratio ${masking_ratio} \
--train_frame_root ${train_frame} --val_frame_root ${val_frame} \
--modality audio \
--use_wandb \