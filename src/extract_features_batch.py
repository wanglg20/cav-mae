#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import torch
from extract_teacher_feats import main

def parse_args():
    parser = argparse.ArgumentParser(description='Extract teacher features for CAV-MAE')
    parser.add_argument('--data_train', type=str, 
                       default='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train_valid.json',
                       help='Training data JSON file path')
    parser.add_argument('--data_val', type=str,
                       default='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_val_valid.json', 
                       help='Validation data JSON file path')
    parser.add_argument('--label_csv', type=str,
                       default='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_class.csv',
                       help='Label CSV file path')
    parser.add_argument('--train_frame_root', type=str,
                       default='/data/wanglinge/dataset/k700/frames_16',
                       help='Training video frames root directory')
    parser.add_argument('--val_frame_root', type=str,
                       default='/data/wanglinge/dataset/k700/frames_16',
                       help='Validation video frames root directory')
    parser.add_argument('--output_dir', type=str,
                       default='./teacher_features',
                       help='Output directory for teacher features')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    
    return parser.parse_args()

def check_disk_space(output_dir, estimated_size_gb=100):
    """检查磁盘空间是否足够"""
    import shutil
    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free // (1024**3)
    
    if free_gb < estimated_size_gb:
        print(f"Warning: Only {free_gb}GB free space available, but {estimated_size_gb}GB estimated needed")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"Disk space check passed: {free_gb}GB available")

def main_extract():
    args = parse_args()
    
    # 检查输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
    
    # 检查磁盘空间
    check_disk_space(args.output_dir)
    
    # 检查CUDA可用性
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = 'cpu'
    
    print(f"Using device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # 运行特征提取
    try:
        main()
        print("Feature extraction completed successfully!")
    except Exception as e:
        print(f"Error during feature extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main_extract()
