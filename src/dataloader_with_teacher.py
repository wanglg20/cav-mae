# -*- coding: utf-8 -*-
# @Time    : 08/06/25 11:03 AM
# @Author  : Linge Wang

import h5py
import os
import torch
import numpy as np
import json
from dataloader import AudiosetDataset


class AudiosetDatasetWithTeacher(AudiosetDataset):
    """
    扩展的AudiosetDataset，可以同时加载teacher模型的特征
    """
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, vision='image', align=False, 
                 num_frames=10, audio_seg_len=4, modality='both', raw='k700', use_mask=False, 
                 num_v_patches=196, num_a_patches=64, mask_ratio=0.75, video_frame_dir=None,
                 teacher_feat_dir=None, use_teacher_features=False):
        """
        Additional Params:
        :param teacher_feat_dir: 预提取的teacher特征存储目录
        :param use_teacher_features: 是否使用teacher特征
        """
        super().__init__(dataset_json_file, audio_conf, label_csv, vision, align, 
                        num_frames, audio_seg_len, modality, raw, use_mask, 
                        num_v_patches, num_a_patches, mask_ratio, video_frame_dir)
        
        self.teacher_feat_dir = teacher_feat_dir
        self.use_teacher_features = use_teacher_features
        self.video_to_shard_mapping = None
        self.shard_files = {}  # 缓存打开的分片文件
        
        if self.use_teacher_features and teacher_feat_dir:
            self._load_shard_mapping()
            self._check_teacher_features()
    
    def _load_shard_mapping(self):
        """加载video_id到分片的映射"""
        mapping_file = os.path.join(self.teacher_feat_dir, "video_to_shard_mapping.json")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r') as f:
                self.video_to_shard_mapping = json.load(f)
            print(f"Loaded shard mapping with {len(self.video_to_shard_mapping)} videos")
        else:
            # 如果没有映射文件，尝试寻找单个h5文件的旧格式
            self.video_to_shard_mapping = None
            print("No shard mapping found, falling back to single file mode")
    
    def _get_shard_file(self, shard_id):
        """获取分片文件句柄，使用缓存避免重复打开"""
        if shard_id not in self.shard_files:
            shard_path = os.path.join(self.teacher_feat_dir, f"teacher_features_shard_{shard_id:04d}.h5")
            if os.path.exists(shard_path):
                self.shard_files[shard_id] = h5py.File(shard_path, 'r')
            else:
                return None
        return self.shard_files[shard_id]
    
    def __del__(self):
        """析构函数，关闭所有打开的分片文件"""
        for shard_file in self.shard_files.values():
            try:
                shard_file.close()
            except:
                pass
    
    def _check_teacher_features(self):
        """检查teacher特征文件是否存在"""
        if not os.path.exists(self.teacher_feat_dir):
            raise FileNotFoundError(f"Teacher feature directory not found: {self.teacher_feat_dir}")
        
        # 统计可用的特征文件数量
        available_features = 0
        if self.video_to_shard_mapping:
            # 分片模式
            for i in range(min(100, self.num_samples)):  # 只检查前100个样本避免过慢
                datum = self.decode_data(self.data[i])
                video_id = datum['video_id']
                if video_id in self.video_to_shard_mapping:
                    available_features += 1
            print(f"Shard mode: Found {available_features}/100 sample features in mapping")
        else:
            # 单文件模式（向后兼容）
            for i in range(min(100, self.num_samples)):
                datum = self.decode_data(self.data[i])
                video_id = datum['video_id']
                feat_file = os.path.join(self.teacher_feat_dir, f"{video_id}.h5")
                if os.path.exists(feat_file):
                    available_features += 1
            print(f"Single file mode: Found {available_features}/100 sample feature files")
    
    def load_teacher_features(self, video_id):
        """加载teacher特征"""
        if self.video_to_shard_mapping and video_id in self.video_to_shard_mapping:
            # 分片模式
            shard_id = self.video_to_shard_mapping[video_id]
            shard_file = self._get_shard_file(shard_id)
            
            if shard_file is None:
                print(f"Warning: Shard file not found for video {video_id}")
                return self._get_zero_features()
            
            try:
                video_group = shard_file[video_id]
                teacher_features = {
                    'clap_target': torch.from_numpy(video_group['clap_target'][:]).float(),
                    'clip_target': torch.from_numpy(video_group['clip_target'][:]).float(), 
                    'clip_attn': torch.from_numpy(video_group['clip_attn'][:]).float()
                }
                return teacher_features
            except Exception as e:
                print(f"Error loading features for {video_id} from shard {shard_id}: {e}")
                return self._get_zero_features()
        else:
            # 单文件模式（向后兼容）
            feat_file = os.path.join(self.teacher_feat_dir, f"{video_id}.h5")
            
            if not os.path.exists(feat_file):
                print(f"Warning: Teacher feature file not found: {feat_file}")
                return self._get_zero_features()
            
            try:
                with h5py.File(feat_file, 'r') as f:
                    teacher_features = {
                        'clap_target': torch.from_numpy(f['clap_target'][:]).float(),
                        'clip_target': torch.from_numpy(f['clip_target'][:]).float(), 
                        'clip_attn': torch.from_numpy(f['clip_attn'][:]).float()
                    }
                    return teacher_features
            except Exception as e:
                print(f"Error loading teacher features from {feat_file}: {e}")
                return self._get_zero_features()
    
    def _get_zero_features(self):
        """返回零特征作为fallback"""
        return {
            'clap_target': torch.zeros(1, 512, 768),  # 根据实际形状调整
            'clip_target': torch.zeros(1, 1961, 768),
            'clip_attn': torch.zeros(1, 12, 1961, 1961)
        }
    
    def __getitem__(self, index):
        # 调用父类方法获取原始数据
        if self.use_mask:
            fbank, image, label_indices, mask, mask_v, mask_a = super().__getitem__(index)
        else:
            fbank, image, label_indices = super().__getitem__(index)
        
        # 如果需要teacher特征，加载它们
        if self.use_teacher_features and self.teacher_feat_dir:
            datum = self.decode_data(self.data[index])
            video_id = datum['video_id']
            teacher_features = self.load_teacher_features(video_id)
            
            if self.use_mask:
                return (fbank, image, label_indices, mask, mask_v, mask_a, 
                       teacher_features['clap_target'], teacher_features['clip_target'], teacher_features['clip_attn'])
            else:
                return (fbank, image, label_indices, 
                       teacher_features['clap_target'], teacher_features['clip_target'], teacher_features['clip_attn'])
        else:
            if self.use_mask:
                return fbank, image, label_indices, mask, mask_v, mask_a
            else:
                return fbank, image, label_indices


class MemoryEfficientAudiosetDataset(AudiosetDatasetWithTeacher):
    """
    内存高效版本，使用缓存优化特征加载
    """
    def __init__(self, *args, **kwargs):
        self._cache_size = kwargs.pop('cache_size', 100)
        super().__init__(*args, **kwargs)
        self._feat_cache = {}  # 简单的LRU缓存
    
    def load_teacher_features(self, video_id):
        """带缓存的特征加载"""
        if video_id in self._feat_cache:
            return self._feat_cache[video_id]
        
        # 如果缓存满了，清除最老的条目
        if len(self._feat_cache) >= self._cache_size:
            # 简单的FIFO策略
            oldest_key = next(iter(self._feat_cache))
            del self._feat_cache[oldest_key]
        
        # 加载特征
        features = super().load_teacher_features(video_id)
        self._feat_cache[video_id] = features
        return features


if __name__ == '__main__':
    def test_teacher_dataset():
        """测试带teacher特征的数据集"""
        audio_conf = {'num_mel_bins': 64, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 
                     'dataset': 'audioset', 'mode':'train', 'mean':-5.081, 'std':4.4849,
                     'noise':True, 'label_smooth': 0, 'im_res': 224}
        
        dataset = AudiosetDatasetWithTeacher(
            '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train_valid.json', 
            audio_conf, 
            num_frames=16,
            label_csv='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_class.csv',  
            modality='both', 
            raw='k700', 
            vision='video', 
            use_mask=True, 
            video_frame_dir='/data/wanglinge/dataset/k700/frames_16',
            teacher_feat_dir='./teacher_features/train',
            use_teacher_features=True
        )
        
        print('dataset length is {:d}'.format(len(dataset)))
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        
        for i, batch in enumerate(loader):
            if len(batch) == 9:  # with teacher features and mask
                fbank, image, label_indices, mask, mask_v, mask_a, clap_target, clip_target, clip_attn = batch
                print(f"fbank: {fbank.shape}")
                print(f"image: {image.shape}")
                print(f"clap_target: {clap_target.shape}")
                print(f"clip_target: {clip_target.shape}")
                print(f"clip_attn: {clip_attn.shape}")
            else:
                print(f"Batch length: {len(batch)}")
            
            if i > 2:
                break
    
    test_teacher_dataset()
