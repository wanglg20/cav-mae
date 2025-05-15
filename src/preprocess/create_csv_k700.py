import os 

data_info = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/kinetics700_train_valid_videos.txt'
video_list = []
root = '/data/wanglinge/dataset/OpenMMLab___Kinetics_700/raw/Kinetics_700/videos'
output_file = 'data/k700_train.csv'
label_list = []

with open(data_info, 'r') as f:
    lines = f.readlines()
    for line in lines:
        video_path = line.strip().split(' ')[0]
        label = line.strip().split(' ')[1]
        video_list.append(video_path)
        label_list.append(label)

with open(output_file, 'w') as f:
    for i, video in enumerate(video_list):
        video_path = os.path.join(root, video)
        label = label_list[i]
        f.write(f"{video_path}\n")