#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析和验证teacher特征的工具脚本
"""

import os
import h5py
import numpy as np
import torch
import argparse
from tqdm import tqdm
import json

def analyze_feature_file(filepath):
    """分析单个特征文件"""
    try:
        with h5py.File(filepath, 'r') as f:
            info = {
                'video_id': f.attrs.get('video_id', 'unknown'),
                'sample_idx': f.attrs.get('sample_idx', -1),
                'clap_target_shape': f['clap_target'].shape,
                'clip_target_shape': f['clip_target'].shape, 
                'clip_attn_shape': f['clip_attn'].shape,
                'file_size_mb': os.path.getsize(filepath) / (1024*1024)
            }
            return info
    except Exception as e:
        return {'error': str(e), 'filepath': filepath}

def analyze_feature_directory(feature_dir):
    """分析特征目录"""
    print(f"Analyzing features in: {feature_dir}")
    
    if not os.path.exists(feature_dir):
        print(f"Directory not found: {feature_dir}")
        return
    
    h5_files = [f for f in os.listdir(feature_dir) if f.endswith('.h5')]
    print(f"Found {len(h5_files)} feature files")
    
    if len(h5_files) == 0:
        return
    
    # 分析前几个文件
    sample_files = h5_files[:min(10, len(h5_files))]
    sample_info = []
    
    for filename in tqdm(sample_files, desc="Analyzing sample files"):
        filepath = os.path.join(feature_dir, filename)
        info = analyze_feature_file(filepath)
        sample_info.append(info)
    
    # 打印统计信息
    print("\n=== Feature Analysis ===")
    for i, info in enumerate(sample_info):
        if 'error' in info:
            print(f"File {i+1}: ERROR - {info['error']}")
        else:
            print(f"File {i+1}: {info['video_id']}")
            print(f"  CLAP target: {info['clap_target_shape']}")
            print(f"  CLIP target: {info['clip_target_shape']}")
            print(f"  CLIP attention: {info['clip_attn_shape']}")
            print(f"  File size: {info['file_size_mb']:.2f} MB")
    
    # 计算总大小
    total_size = sum(os.path.getsize(os.path.join(feature_dir, f)) for f in h5_files)
    total_size_gb = total_size / (1024**3)
    print(f"\nTotal directory size: {total_size_gb:.2f} GB")
    print(f"Average file size: {total_size/len(h5_files)/(1024*1024):.2f} MB")

def verify_feature_completeness(json_file, feature_dir):
    """验证特征文件的完整性"""
    print(f"Verifying feature completeness...")
    print(f"JSON file: {json_file}")
    print(f"Feature dir: {feature_dir}")
    
    # 读取数据集信息
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    total_samples = len(data['data'])
    missing_features = []
    
    for sample in tqdm(data['data'], desc="Checking completeness"):
        video_id = sample['video_id']
        feat_file = os.path.join(feature_dir, f"{video_id}.h5")
        
        if not os.path.exists(feat_file):
            missing_features.append(video_id)
    
    print(f"\n=== Completeness Report ===")
    print(f"Total samples: {total_samples}")
    print(f"Missing features: {len(missing_features)}")
    print(f"Coverage: {(total_samples - len(missing_features))/total_samples*100:.2f}%")
    
    if missing_features:
        print(f"\nFirst 10 missing features:")
        for vid in missing_features[:10]:
            print(f"  {vid}")

def test_feature_loading():
    """测试特征加载性能"""
    from dataloader_with_teacher import AudiosetDatasetWithTeacher
    
    print("Testing feature loading performance...")
    
    audio_conf = {'num_mel_bins': 64, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 
                 'dataset': 'audioset', 'mode':'eval', 'mean':-5.081, 'std':4.4849,
                 'noise':False, 'label_smooth': 0, 'im_res': 224}
    
    dataset = AudiosetDatasetWithTeacher(
        '/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid.json', 
        audio_conf, 
        num_frames=16,
        label_csv='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_class.csv',  
        modality='both', 
        raw='k700', 
        vision='video', 
        use_mask=True, 
        video_frame_dir='/data/wanglinge/dataset/k700/frames_16',
        teacher_feat_dir='./teacher_features/val',
        use_teacher_features=True
    )
    
    # 测试加载几个样本
    import time
    
    load_times = []
    for i in range(min(10, len(dataset))):
        start_time = time.time()
        sample = dataset[i]
        load_time = time.time() - start_time
        load_times.append(load_time)
        
        print(f"Sample {i}: {load_time:.3f}s")
    
    print(f"\nAverage load time: {np.mean(load_times):.3f}s")
    print(f"Max load time: {np.max(load_times):.3f}s")

def main():
    parser = argparse.ArgumentParser(description='Analyze teacher features')
    parser.add_argument('--feature_dir', type=str, default='./teacher_features/val',
                       help='Feature directory to analyze')
    parser.add_argument('--json_file', type=str, 
                       default='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid.json',
                       help='JSON file to check completeness against')
    parser.add_argument('--action', type=str, choices=['analyze', 'verify', 'test'], 
                       default='analyze', help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'analyze':
        analyze_feature_directory(args.feature_dir)
    elif args.action == 'verify':
        verify_feature_completeness(args.json_file, args.feature_dir)
    elif args.action == 'test':
        test_feature_loading()

if __name__ == "__main__":
    main()
