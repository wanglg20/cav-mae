export CUDA_VISIBLE_DEVICES=4,5,6,7


model=cav-mae-sync
masking_ratio=0.75
mask_mode=unstructured # or time, or freq, or tf
contrast_loss_weight=0.01
mae_loss_weight=1.0
tr_pos=False
norm_pix_loss=False
pretrain_path=/data/wanglinge/project/cav-mae/src/weight/init/ori_mae_11.pth

bal=None
lr=5e-5
epoch=25    
lrscheduler_start=10
lrscheduler_decay=0.5
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1000
noise=True
mixup=0.0
batch_size=128
lr_adapt=False

dataset=audioset
tr_data=/data/wanglinge/project/cav-mae/src/data/info/k700_train.json
te_data=/data/wanglinge/project/cav-mae/src/data/info/k700_val.json
label_csv=/data/wanglinge/project/cav-mae/src/data/info/k700_class.csv
cd /data/wanglinge/project/cav-mae/src
exp_dir=./exp/trainmae-k700-${model}-lr${lr}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}
mkdir -p $exp_dir


PYTHONWARNINGS=ignore torchrun --nproc_per_node=4 run_cavmae_pretrain.py --model ${model} --dataset ${dataset} \
--data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
--label-csv ${label_csv} --n_class 700 \
--lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
--mixup ${mixup} --bal ${bal} \
--lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
--dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
--lr_adapt ${lr_adapt} \
--norm_pix_loss ${norm_pix_loss} \
--pretrain_path ${pretrain_path} \
--mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} \
--tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --mask_mode ${mask_mode} \
--use_wandb \
# --wandb_id 4l19d6ib