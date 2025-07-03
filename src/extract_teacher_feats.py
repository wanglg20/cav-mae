import torch
from torch import nn
from models.teacher import clip_b16
from transformers.models.clap.modeling_clap import ClapAudioModelOutput, ClapAudioPatchEmbed, ClapAudioStage, ClapAudioPatchMerging
from transformers.models.clap.modeling_clap import ClapAudioModel
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor
import dataloader
import h5py
import os
from tqdm import tqdm


def main():
    # arguments
    data_val = '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid.json'
    data_train = '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train_valid.json'
    label_csv='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_class.csv'
    target_length = 1024
    im_res = 224
    dataset_mean=-5.081
    dataset_std=4.4849
    noise = False
    val_frame_root = '/data/wanglinge/dataset/k700/frames_16'
    train_frame_root = '/data/wanglinge/dataset/k700/frames_16'


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher_v = clip_b16(
      pretrained=True,
      clip_norm_type='l2',
      input_resolution=224,
      return_attn=True,
      clip_return_layer=1,
      clip_return_interval=1,
      clip_return_cls=True
    )
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
    clap_encoder = clap_model.audio_model
    weight_path = '/data/wanglinge/project/cav-mae/src/weight/teacher/clap.pth'
    clap_encoder.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    teacher_a = clap_encoder

    audio_conf = {'num_mel_bins': 64, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'k700', 
                  'mode':'train', 'mean':dataset_mean, 'std':dataset_std, 'noise':noise, 'label_smooth': 0, 'im_res': im_res}
    val_audio_conf = {'num_mel_bins': 64, 'target_length': target_length, 'freqm': 0, 'timem': 0, 'mixup': 0, 'dataset': 'k700',
                  'mode':'eval', 'mean': dataset_mean, 'std': dataset_std, 'noise': False, 'im_res': im_res}
    # dataset

    # 设置模型为eval模式
    teacher_v.eval()
    teacher_a.eval()
    
    # 提取验证集特征
    print("Extracting validation set features...")
    extract_features_for_split(data_val, val_frame_root, '/data/wanglinge/dataset/k700/features/val', 
                             val_audio_conf, teacher_v, teacher_a, device, label_csv)
    
    # 提取训练集特征
    # print("Extracting training set features...")
    # extract_features_for_split(data_train, train_frame_root, '/data/wanglinge/dataset/k700/features/train', 
    #                          audio_conf, teacher_v, teacher_a, device, label_csv)
    
    print("Feature extraction completed!")

if __name__ == "__main__":
    main()

class TeacherFeatureSaver:
    """分片保存teacher特征的类"""
    def __init__(self, save_dir, videos_per_shard=5000):
        self.save_dir = save_dir
        self.videos_per_shard = videos_per_shard
        self.current_shard = 0
        self.current_shard_count = 0
        self.current_file = None
        self.video_to_shard_mapping = {}
        
        os.makedirs(save_dir, exist_ok=True)
        self._open_new_shard()
    
    def _open_new_shard(self):
        """打开新的分片文件"""
        if self.current_file is not None:
            self.current_file.close()
        
        shard_file = os.path.join(self.save_dir, f"teacher_features_shard_{self.current_shard:04d}.h5")
        self.current_file = h5py.File(shard_file, 'w')
        self.current_shard_count = 0
    
    def save_features(self, video_id, clap_target, clip_target, clip_attn):
        """保存单个视频的特征"""
        if self.current_shard_count >= self.videos_per_shard:
            self.current_shard += 1
            self._open_new_shard()
        
        # 在当前分片中创建视频组
        video_group = self.current_file.create_group(video_id)
        video_group.create_dataset('clap_target', data=clap_target.cpu().numpy(), compression='gzip')
        video_group.create_dataset('clip_target', data=clip_target.cpu().numpy(), compression='gzip')
        video_group.create_dataset('clip_attn', data=clip_attn.cpu().numpy(), compression='gzip')
        
        # 记录映射关系
        self.video_to_shard_mapping[video_id] = self.current_shard
        self.current_shard_count += 1
    
    def finalize(self):
        """完成保存并生成映射文件"""
        if self.current_file is not None:
            self.current_file.close()
        
        # 保存video_id到分片的映射
        import json
        mapping_file = os.path.join(self.save_dir, "video_to_shard_mapping.json")
        with open(mapping_file, 'w') as f:
            json.dump(self.video_to_shard_mapping, f)
        
        print(f"Saved {len(self.video_to_shard_mapping)} videos across {self.current_shard + 1} shards")

def save_teacher_features(sample_idx, clap_target, clip_target, clip_attn, sample_data, saver):
    """保存teacher模型的特征到分片"""
    # 解析sample_data获取video_id
    if isinstance(sample_data, dict):
        video_id = sample_data['video_id']
    else:
        # numpy array format
        video_id = sample_data[2]
    
    try:
        saver.save_features(video_id, clap_target, clip_target, clip_attn)
    except Exception as e:
        print(f"Error saving features for {video_id}: {e}")

def extract_features_for_split(data_path, frame_root, save_dir, audio_conf, teacher_v, teacher_a, device, label_csv):
    """为指定数据集分割提取特征"""
    dataset = dataloader.AudiosetDataset(data_path, label_csv=label_csv, num_frames=16,
        audio_conf=audio_conf, modality='both', vision='video', raw='k700', 
        use_mask=True, video_frame_dir=frame_root)
    
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=False
    )
    
    is_longer_tensor = torch.tensor([True], dtype=torch.bool, device=device)
    output_attentions_tensor = torch.tensor([True], dtype=torch.bool, device=device)
    
    # 创建分片保存器
    saver = TeacherFeatureSaver(save_dir, videos_per_shard=5000)
    
    try:
        for i, (a_input, v_input, labels, mask, mask_v, mask_a) in enumerate(tqdm(data_loader)):
            B, T, C, H, W = v_input.shape
            clap_input = a_input.unsqueeze(1).to(device)  # B, 1, 1024, 64
            audio_outputs = teacher_a(clap_input, is_longer=is_longer_tensor, output_attentions=output_attentions_tensor)
            clap_target, clap_attn = audio_outputs.last_hidden_state, audio_outputs.attentions[-1]

            v_input = v_input.permute(0, 2, 1, 3, 4).to(device)  # B, C, T, H, W
            clip_target, clip_attn = teacher_v(v_input) # K, B, 1961, 768
            
            # 保存特征
            save_teacher_features(i, clap_target, clip_target, clip_attn, 
                                dataset.data[i], saver)
            
            if i % 100 == 0:
                print(f"Processed {i}/{len(data_loader)} samples")
    finally:
        # 确保最后关闭文件并保存映射
        saver.finalize()