import torch
from torch import nn
from models.teacher import clip_b16
from transformers.models.clap.modeling_clap import ClapAudioModelOutput, ClapAudioPatchEmbed, ClapAudioStage, ClapAudioPatchMerging
from transformers.models.clap.modeling_clap import ClapAudioModel
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor
import dataloader


def get_audio_feats(teacher_a, audio_input):
    """
    Extract audio features using the teacher model.
    """
    if isinstance(teacher_a, ClapAudioModel):
        # For ClapAudioModel
        audio_output = teacher_a(audio_input)
        return audio_output.last_hidden_state
    elif isinstance(teacher_a, ClapModel):
        # For ClapModel
        audio_output = teacher_a.audio_model(audio_input)
        return audio_output.last_hidden_state
    else:
        raise ValueError("Unsupported teacher model type for audio feature extraction.")
    
    

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

    val_set = dataloader.AudiosetDataset(data_val, label_csv=label_csv, num_frames=16,
        audio_conf=val_audio_conf, modality='both', vision='video', raw='k700', 
        use_mask=True, video_frame_dir=val_frame_root)
    
