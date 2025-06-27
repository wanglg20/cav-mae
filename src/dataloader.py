# -*- coding: utf-8 -*-
# @Time    : 08/06/25 11:03 AM
# @Author  : Linge Wang


# original author info:
# -*- coding: utf-8 -*-
# @Time    : 6/19/21 12:23 AM
# @Author  : Yuan Gong
# @Affiliation  : Massachusetts Institute of Technology
# @Email   : yuangong@mit.edu
# @File    : dataloader_old.py

# modified from:
# Author: David Harwath
# with some functions borrowed from https://github.com/SeanNaren/deepspeech.pytorch

import csv
import json
import os.path
import torchaudio
import numpy as np
import torch
import torch.nn.functional
from torch.utils.data import Dataset
import random
import torchvision.transforms as T
from PIL import Image
import PIL


def rand_mask_generate(num_frames, num_patches, mask_ratio):
    num_mask_per_frame = int(num_patches * mask_ratio)

    rand_vals = torch.rand((num_frames, num_patches), device='cpu')
    _, indices = torch.topk(rand_vals, k=num_mask_per_frame, dim=1, largest=False)
    mask = torch.zeros((num_frames, num_patches), dtype=torch.bool, device='cpu')
    mask.scatter_(1, indices, True)
    # mask = mask.flatten()
    return mask


def mask_expand2d(mask, expand_ratio=2):
    """
    Expand the mask in both dimensions by a factor of expand_ratio.
    :param mask: 3D boolean tensor, shape (Frame, Freq, Time_stamp)
    :param expand_ratio: int, factor by which to expand the mask
    :return: 2D boolean tensor with expanded mask
    """
    if expand_ratio <= 1:
        return mask
    # Repeat the mask in both dimensions
    mask_expanded = mask.repeat_interleave(expand_ratio, dim=1).repeat_interleave(expand_ratio, dim=2)
    return mask_expanded

def make_index_dict(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            # index_lookup[row['mid']] = row['index']
            index_lookup[row['id']] = int(row['id'])
            line_count += 1
    return index_lookup


def make_index_dict_ori(label_csv):
    index_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            index_lookup[row['mid']] = row['index']
            line_count += 1
    return index_lookup


def make_name_dict(label_csv):
    name_lookup = {}
    with open(label_csv, 'r') as f:
        csv_reader = csv.DictReader(f)
        line_count = 0
        for row in csv_reader:
            #name_lookup[row['index']] = row['display_name']
            name_lookup[row['id']] = row['name']
            line_count += 1
    return name_lookup

def lookup_list(index_list, label_csv):
    label_list = []
    table = make_name_dict(label_csv)
    for item in index_list:
        label_list.append(table[item])
    return label_list

def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class AudiosetDataset(Dataset):
    def __init__(self, dataset_json_file, audio_conf, label_csv=None, vision='image', align = False, 
                 num_frames=10, audio_seg_len = 4, modality='both', raw = 'k700', use_mask=False, 
                 num_v_patches=196, num_a_patches=64, mask_ratio=0.75, video_frame_dir=None):
        """
        Dataset that manages audio recordings
        :param audio_conf: Dictionary containing the audio loading and preprocessing settings
        :param dataset_json_file


        Additional Params:
        :param modality: which modality to be loaded(audio, vision, both)
        :param vision: form of vision modality(image / video)
        :param align: whether to align the selected frame and the audio spectrum 
        :param num_frames: frames of original video
        :param raw: raw dataset: k700 / audioset 
        """
        self.use_mask = use_mask
        self.num_v_patches = num_v_patches
        self.num_a_patches = num_a_patches
        self.mask_ratio = mask_ratio
        self.video_frame_dir = video_frame_dir
        
        self.raw = raw
        self.datapath = dataset_json_file
        with open(dataset_json_file, 'r') as fp:
            data_json = json.load(fp)

        self.data = data_json['data']
        self.data = self.pro_data(self.data)
        print('Dataset has {:d} samples'.format(self.data.shape[0]))
        self.num_samples = self.data.shape[0]
        self.audio_conf = audio_conf
        self.label_smooth = self.audio_conf.get('label_smooth', 0.0)
        print('Using Label Smoothing: ' + str(self.label_smooth))
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm', 0)
        self.timem = self.audio_conf.get('timem', 0)
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'), self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup', 0)
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print('use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))

        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise', False)
        if self.noise == True:
            print('now use noise augmentation')
        else:
            print('not use noise augmentation')
        if self.raw == 'k700':
            self.index_dict = make_index_dict(label_csv)
        else:
            self.index_dict = make_index_dict_ori(label_csv)
        self.label_num = len(self.index_dict)
        print('number of classes is {:d}'.format(self.label_num))

        self.target_length = self.audio_conf.get('target_length')

        # train or eval
        self.mode = self.audio_conf.get('mode')
        print('now in {:s} mode.'.format(self.mode))

        # set the frame to use in the eval mode, default value for training is -1 which means random frame
        self.frame_use = self.audio_conf.get('frame_use', -1)
        # by default, 10 frames are used
        self.total_frame = self.audio_conf.get('total_frame', 10)
        print('now use frame {:d} from total {:d} frames'.format(self.frame_use, self.total_frame))

        # by default, all models use 224*224, other resolutions are not tested
        self.im_res = self.audio_conf.get('im_res', 224)
        print('now using {:d} * {:d} image input'.format(self.im_res, self.im_res))
        self.preprocess = T.Compose([
            T.Resize(self.im_res, interpolation=PIL.Image.BICUBIC),
            T.CenterCrop(self.im_res),
            T.ToTensor(),
            T.Normalize(
                mean=[0.4850, 0.4560, 0.4060],
                std=[0.2290, 0.2240, 0.2250]
            )])
        
        # -------------------------
        # modified properties
        self.vision = vision        # if use all frames
        self.align = align          # if align audio and frame, refer to https://arxiv.org/abs/2505.01237 for details
        self.num_frames = num_frames
        self.audio_seg_len = audio_seg_len
        self.spec_seg_len = int(self.target_length / num_frames)
        self.spec_window_size_half = int(0.5*self.spec_seg_len * audio_seg_len)
        if self.vision != 'image':
            self.mixup = 0      # if not image, set mixup to 0
        self.modality = modality
        # -------------------------

    # change python list to numpy array to avoid memory leak.
    def pro_data(self, data_json):
        if self.raw == 'k700':
            for i in range(len(data_json)):
                data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id'], data_json[i]['video_path']]
        else:
            for i in range(len(data_json)):
                data_json[i] = [data_json[i]['wav'], data_json[i]['labels'], data_json[i]['video_id']]
        data_np = np.array(data_json, dtype=str)
        return data_np

    # reformat numpy data to original json format, make it compatible with old code
    def decode_data(self, np_data):
        datum = {}
        datum['wav'] = np_data[0]
        datum['labels'] = np_data[1]
        datum['video_id'] = np_data[2]
        if self.raw == 'k700':
            datum['video_path'] = np_data[3]
        return datum

    def get_image(self, filename, filename2=None, mix_lambda=1):
        if filename2 == None:
            img = Image.open(filename)
            image_tensor = self.preprocess(img)
            return image_tensor
        else:
            img1 = Image.open(filename)
            image_tensor1 = self.preprocess(img1)

            img2 = Image.open(filename2)
            image_tensor2 = self.preprocess(img2)

            image_tensor = mix_lambda * image_tensor1 + (1 - mix_lambda) * image_tensor2
            return image_tensor

    def _wav2fbank(self, filename, filename2=None, mix_lambda=-1):
        # no mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)   
            #print("original waveform shape is ", waveform.shape) # 1, 160173
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        try:
            fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False, window_type='hanning', num_mel_bins=self.melbins, dither=0.0, frame_shift=10)
            
        except:
            fbank = torch.zeros([512, 128]) + 0.01
            print('there is a loading error')
        #print("fbank shape is ", fbank.shape)   #(1000, 128)
        target_length = self.target_length
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            fbank = fbank[0:target_length, :]

        return fbank

    def randselect_img(self, video_id, video_path):
        if self.mode == 'eval':
            # if not specified, use the middle frame
            if self.frame_use == -1:
                frame_idx = int((self.total_frame) / 2)
            else:
                frame_idx = self.frame_use
        else:
            frame_idx = random.randint(0, self.num_frames-1)

        while os.path.exists(video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg') == False and frame_idx >= 1:
            print('frame {:s} {:d} does not exist'.format(video_id, frame_idx))
            frame_idx -= 1
        out_path = video_path + '/frame_' + str(frame_idx) + '/' + video_id + '.jpg'
        #print(out_path)
        return out_path
    


    def align_img_spec(self, fbank, frame_idx):
        # align the audio and video
        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        # frame_idx is the index of the image
        S_center = int(self.spec_seg_len * (0.5 + frame_idx))
        S_start = S_center - self.spec_window_size_half
        S_end = S_center + self.spec_window_size_half
        fbank_len = fbank.shape[0]
        if S_start < 0:
            fbank = fbank[0:self.spec_window_size_half * 2]
        elif S_end > fbank_len:
            fbank = fbank[fbank_len-self.spec_window_size_half * 2:fbank_len]
        else:
            fbank = fbank[S_start:S_end]
        return fbank

    def __getitem__(self, index):

        if random.random() < self.mixup:
            datum = self.data[index]
            datum = self.decode_data(datum)
            if self.video_frame_dir != None:
                datum['video_path'] = self.video_frame_dir  # overwrite the video path if specified
            mix_sample_idx = random.randint(0, self.num_samples-1)
            mix_datum = self.data[mix_sample_idx]
            mix_datum = self.decode_data(mix_datum)
            # get the mixed fbank
            mix_lambda = np.random.beta(10, 10)

            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += mix_lambda * (1.0 - self.label_smooth)
            for label_str in mix_datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] += (1.0 - mix_lambda) * (1.0 - self.label_smooth)
            label_indices = torch.FloatTensor(label_indices)

            try:
                fbank = self._wav2fbank(datum['wav'], mix_datum['wav'], mix_lambda)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')
            if self.modality != 'audioonly':
                try:
                    path1 = self.randselect_img(datum['video_id'], datum['video_path'])
                    path2 = self.randselect_img(mix_datum['video_id'], datum['video_path'])
                    image = self.get_image(path1, path2, mix_lambda)
                except:
                    image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                    print('there is an error in loading image')
            else:
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
    

        else:
            #print("args:", self.modality)
            datum = self.data[index]
            datum = self.decode_data(datum)
            if self.video_frame_dir != None:
                datum['video_path'] = self.video_frame_dir  # overwrite the video path if specified
            # label smooth for negative samples, epsilon/label_num
            label_indices = np.zeros(self.label_num) + (self.label_smooth / self.label_num)
            try:
                fbank = self._wav2fbank(datum['wav'], None, 0)
            except:
                fbank = torch.zeros([self.target_length, 128]) + 0.01
                print('there is an error in loading audio')

            if self.modality == 'audioonly':
                image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
            else:
                if self.vision == 'image':
                    try:
                        path1 = self.randselect_img(datum['video_id'], datum['video_path'])
                        image = self.get_image(path1, None, 0)
                    except:
                        image = torch.zeros([3, self.im_res, self.im_res]) + 0.01
                        if self.modality != 'audioonly':
                            print('there is an error in loading image')
                        
                elif self.vision == 'video':
                    frame_id = range(self.num_frames)
                    path1 = self.randselect_img(datum['video_id'], datum['video_path'])
                    image_paths = [datum['video_path'] + '/frame_' + str(frame_id[i]) + '/' + datum['video_id'] + '.jpg' for i in range(self.num_frames)]
                    images = [self.get_image(image_paths[i], None, 0) for i in range(self.num_frames)]
                    image = torch.stack(images, dim=0)
                else:
                    raise ValueError('vision should be image or video')
            for label_str in datum['labels'].split(','):
                label_indices[int(self.index_dict[label_str])] = 1.0 - self.label_smooth
            label_indices = torch.FloatTensor(label_indices)

        # SpecAug, not do for eval set
        freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
        timem = torchaudio.transforms.TimeMasking(self.timem)
        fbank = torch.transpose(fbank, 0, 1)
        fbank = fbank.unsqueeze(0)
        if self.freqm != 0:
            fbank = freqm(fbank)
        if self.timem != 0:
            fbank = timem(fbank)
        fbank = fbank.squeeze(0)
        fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if self.skip_norm == False:
            fbank = (fbank - self.norm_mean) / (self.norm_std)
        # skip normalization the input ONLY when you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True:
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-self.target_length, self.target_length), 0)

        # align audio spec and frame idx
        if self.align and self.modality != 'audioonly':
            image_path = path1
            frame_dir = image_path.strip(os.sep).split(os.sep)[-2]
            frame_idx = int(frame_dir[-1])
            fbank = self.align_img_spec(fbank, frame_idx)
        # fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]

        if self.use_mask:
            zeros = torch.zeros(self.num_frames, 1).bool()
            mask_v = rand_mask_generate(self.num_frames, self.num_v_patches, self.mask_ratio)
            mask_a_ori = rand_mask_generate(self.num_frames, 64 // self.num_frames, self.mask_ratio)
            mask_a = mask_a_ori.reshape(self.num_frames, 2, 2)
            mask_a = mask_expand2d(mask_a, expand_ratio=2)  # Frame, Freq, Time, 
            mask_a = mask_a.reshape(self.num_frames, -1)
            mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=1)
            return fbank, image, label_indices, mask, mask_v, mask_a_ori
        return fbank, image, label_indices

    def __len__(self):
        return self.num_samples
    
if __name__ == '__main__':
    # test the dataloader
    audio_conf = {'num_mel_bins': 64, 'target_length': 1024, 'freqm': 0, 'timem': 0, 'mixup': 0.0, 'dataset': 'audioset', 'mode':'train', 'mean':-5.081, 'std':4.4849,
              'noise':True, 'label_smooth': 0, 'im_res': 224}
    dataset = AudiosetDataset('/data/wanglinge/project/cav-mae/src/data/info/k700/k700_train_valid.json', audio_conf, num_frames=16,
                               label_csv='/data/wanglinge/project/cav-mae/src/data/info/k700/k700_class.csv',  modality='both', 
                               raw='k700', vision='video', use_mask=True, video_frame_dir='/data/wanglinge/dataset/k700/frames_16')
    print('dataset length is {:d}'.format(len(dataset)))
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    loss = torch.nn.CrossEntropyLoss()
    from traintest_ft import *
    for i, (fbank, image, label_indices, mask, mask_v, mask_a) in enumerate(loader):
        print(fbank.shape)      # B, 1024, 128
        print(image.shape)      # B, 10, 3, 224, 224
        print(label_indices.shape) # B, 700, torch.sum() = 1
        print(mask.shape)
        print(mask_v.shape)
        print(mask_a.shape)    # B, 16, 4
        print("present idx:", i, end='\r')
        break
        if i > 1000:
            break