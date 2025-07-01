import csv
import os

target_path = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/videos/vol_00'
video_list = os.listdir(target_path)[:100]
output_file = 'data/k700_test.csv'
with open(output_file, 'w') as f:
    for video in video_list:
        video_path = os.path.join(target_path, video)
        f.write(f"{video_path}\n")
