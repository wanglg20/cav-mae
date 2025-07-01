import warnings
import torch
import time
from typing import Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)
from models.mamba_pretrain import CrossMamba, CrossMambaFT, UniModalMamba, UniModalMamba_FT
from models.videomamba_pretrain import VisionMamba, videomamba_middle_pretrain
from models.teacher import clip_b16
from dataloader import rand_mask_generate, mask_expand2d
from transformers.models.clap.modeling_clap import ClapAudioModelOutput, ClapAudioPatchEmbed, ClapAudioStage, ClapAudioPatchMerging
from transformers.models.clap.modeling_clap import ClapAudioModel
from transformers import CLIPModel, CLIPProcessor, ClapModel, ClapProcessor
from transformers import MambaModel, MambaConfig
# FLOPs计算工具导入
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    print("Warning: thop not available. Installing...")
    THOP_AVAILABLE = False

try:
    from fvcore.nn import FlopCountMode, flop_count
    FVCORE_AVAILABLE = True
except ImportError:
    print("Warning: fvcore not available.")
    FVCORE_AVAILABLE = False

try:
    from ptflops import get_model_complexity_info
    PTFLOPS_AVAILABLE = True
except ImportError:
    print("Warning: ptflops not available.")
    PTFLOPS_AVAILABLE = False

def test_cross_mamba(Flop_test = True):
    print("\n" + "=" * 50)
    print("Testing CrossMamba (Pre-training)")
    model = CrossMamba(
        num_frames=16,
        audio_length=1024,
        )
    print("MambaPretrain model created successfully.")
    print(model.patch_embed_v.num_patches, model.patch_embed_a.num_patches)
    
    import torch
    B = 2
    v = torch.randn(B, 3, 16, 224, 224)  # Video input
    a = torch.randn(B, 64, 1024)
    zeros = torch.zeros(16, 1).bool()
    mask_v = rand_mask_generate(16, 196, 0.75)
    mask_a = rand_mask_generate(16, 64 // 16, 0.75)
    mask_a_ori = mask_a.reshape(16, 2, 2)
    mask_a = mask_expand2d(mask_a_ori, expand_ratio=2)  # Frame, Freq, Time, 
    mask_a = mask_a.reshape(16, -1)
    mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=1)
    mask = mask.reshape(1, -1)
    mask = mask.repeat(B, 1)  # Repeat for batch size
    mask_v = mask_v.reshape(1, -1).repeat(B, 1)
    mask_a_ori = mask_a_ori.reshape(1, -1).repeat(B, 1)
      # Add ones at the beginning and end
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    v = v.to(device)
    a = a.to(device)
    mask = mask.to(device)
    if Flop_test:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params: {} M'.format(n_parameters / 1e6))

        # 计算FLOPs
        print("\nCalculating FLOPs for CrossMamba...")
        inputs = (v, a, mask)

        flops_results = {}

        # THOP method
        thop_results = calculate_flops_thop(model, inputs)
        if thop_results[0] is not None:
            flops_results['THOP'] = thop_results[:2]
            print(f"THOP - FLOPs: {thop_results[2]}, Params: {thop_results[3]}")

        # FVCore method
        fvcore_flops = calculate_flops_fvcore(model, inputs)
        if fvcore_flops is not None:
            flops_results['FVCore'] = fvcore_flops
            print(f"FVCore - FLOPs: {fvcore_flops/1e9:.2f}G")

        print_flops_summary("CrossMamba", flops_results, n_parameters)

        # 性能基准测试
        print("Running inference speed benchmark...")
        avg_time, fps = benchmark_inference_speed(model, inputs, num_runs=50, warmup_runs=5)
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Inference FPS: {fps:.2f}")
    
    clip_vis = model.forward_features(v, a, mask)
    x_clip, x_clap, global_v, global_a = model(v, a, mask)

    print("CLIP output shape:", x_clip.shape)           #1, 1, 784, 512
    print("CLAP output shape:", x_clap.shape)           #1, 1, 64, 512
    print("Global Video shape:", global_v.shape)         #1, 16
    print("Global Audio shape:", global_a.shape)         #1, 16,
    pred_clap = x_clap[:, :, ::4, :]
    print("Pred CLAP shape:", pred_clap.shape)         #1, 1, 16, 512
    teacher_clip = torch.randn(B, 196*16, 768).to(device)  # Simulated teacher output
    teacher_clap = torch.randn(B, 64, 768).to(device)
    
    mask = mask.reshape(B, 16, -1)
    mask_v = mask_v.reshape(B, -1)
    mask_a = mask_a_ori.reshape(B, -1)
    target_clip = teacher_clip[~mask_v].reshape(B, -1, 768)
    target_clap = teacher_clap[~mask_a].reshape(B, -1, 768)
    print("Target CLIP shape:", target_clip.shape)     # 1, 196*16, 768
    print("Target CLAP shape:", target_clap.shape)     # 1, 16, 768

def test_clip_teacher():
    teacher_model = clip_b16(
      pretrained=True,
      clip_norm_type='l2',
      input_resolution=224,
      return_attn=True,
      clip_return_layer=1,
      clip_return_interval=1,
      clip_return_cls=True
    )
    # weight = '/data/wanglinge/project/cav-mae/src/weight/teacher/vit_b16.pth'
    # teacher_model.load_state_dict(torch.load(weight, map_location='cpu'), strict=True)
    mask = torch.cat([
        torch.ones(1, 10 * int(14 * 14 * 0.75)),
        torch.zeros(1, 10 * int(14 * 14 * 0.25)),
    ], dim=-1).bool()
    mask = mask.repeat(2, 1)
    x = torch.randn(2, 3, 16, 224, 224)  
    out = teacher_model(x)
    print(out[0].shape)  # K, B, 1961, 768
    print(out[1].shape)  # Attention weights shape  B*T, 49


def test_clap_teacher():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to(device)
    clap_encoder = clap_model.audio_model
    weight_path = '/data/wanglinge/project/cav-mae/src/weight/teacher/clap.pth'
    clap_encoder.load_state_dict(torch.load(weight_path, map_location=device), strict=True)
    teacher_model = clap_encoder
    audio = torch.randn(1, 1, 1024, 64).to(device)  # 假设的输入音频特征, B
    teacher_model = teacher_model.to(device)
    out = teacher_model(audio, is_longer=torch.tensor([1]).bool().to(device), output_attentions=torch.tensor([True]).bool().to(device), return_dict=True)
    clap_feat = out.last_hidden_state  # 1, 768, 2, 32
    clap_attn = out.attentions[-1]  # 1, 32, 64, 64
    print(clap_feat.shape)  # 1, 768, 2, 32
# def test_clap_teacher():
    


def test_cross_mamba_ft(Flop_test = True):
    # Test CrossMambaFT (for fine-tuning)
    print("\n" + "=" * 50)
    print("Testing CrossMambaFT (Fine-tuning)")
    print("=" * 50)
    model_ft = CrossMambaFT(num_classes=700, fc_drop_rate=0.1)
    print("CrossMambaFT model created successfully.")
    print(f"Number of classes: {model_ft.num_classes}")

    prob_keys = [k for k, v in model_ft.named_parameters() if v.requires_grad and k.startswith('head')]
    print("Probing parameters:")
    print(prob_keys)
    import torch
    device = torch.device("cuda")

    # Test inputs
    v = torch.randn(1, 3, 16, 224, 224)  # Video input
    a = torch.randn(1, 64, 1024)         # Audio input
    
    # Test mask for pre-training
    ones = torch.ones(1, 10)
    mask = torch.cat([
        ones,
        torch.ones(1, 10 * int(14 * 14 * 0.75)),
        torch.zeros(1, 10 * int(14 * 14 * 0.25)),
        ones,
        ones,     
        torch.ones(1, 10 * int(6 * 8 * 0.75)),      # 6 = 960 / 10 / 16
        torch.zeros(1, 10 * int(6 * 8 * 0.25)),
        ones, 
    ], dim=-1).to(torch.bool)

    # Move to device
    model_ft = model_ft.to(device)
    v = v.to(device)
    a = a.to(device)
    mask = mask.to(device)
    if Flop_test:
        # 计算参数数量
        n_parameters = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
        print('Number of params: {} M'.format(n_parameters / 1e6))
        
        # 计算FLOPs
        print("\nCalculating FLOPs for CrossMambaFT...")
        inputs = (v, a)
        
        flops_results = {}
        
        # THOP method
        thop_results = calculate_flops_thop(model_ft, inputs)
        if thop_results[0] is not None:
            flops_results['THOP'] = thop_results[:2]
            print(f"THOP - FLOPs: {thop_results[2]}, Params: {thop_results[3]}")
        
        # FVCore method
        fvcore_flops = calculate_flops_fvcore(model_ft, inputs)
        if fvcore_flops is not None:
            flops_results['FVCore'] = fvcore_flops
            print(f"FVCore - FLOPs: {fvcore_flops/1e9:.2f}G")
        
        print_flops_summary("CrossMambaFT", flops_results, n_parameters)
        
        # 性能基准测试
        print("Running inference speed benchmark...")
        avg_time, fps = benchmark_inference_speed(model_ft, inputs, num_runs=50, warmup_runs=5)
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Inference FPS: {fps:.2f}")
    
    # Test fine-tuning model
    print("\nTesting fine-tuning forward pass...")
    try:
        with torch.no_grad():
            outputs = model_ft(v, a)
            print(f"Video logits shape: {outputs['logits'].shape}")
            print(f"Video features shape: {outputs['feat_v'].shape}")
            print(f"Audio features shape: {outputs['feat_a'].shape}")
            print("Fine-tuning forward pass successful!")
    except Exception as e:
        print(f"Fine-tuning forward pass failed: {e}")


def test_unimodal_mamba(modality = 'audio'):
    print("\n" + "=" * 50)
    print("Testing uni-modality mamba, current modality is: {}".format(modality))
    print("=" * 50)
    model = UniModalMamba()
    device = torch.device("cuda")
    B = 2  
    # Test inputs
    v = torch.randn(B, 3, 16, 224, 224)  # Video input
    a = torch.randn(B, 64, 1024)         # Audio input
    
    # Test mask for pre-training
    ones = torch.ones(1, 16, 1).bool()
    zeros = torch.zeros(1, 16, 1).bool()
    mask_v = rand_mask_generate(16, 196, 0.75)
    mask_a = rand_mask_generate(16, 4, 0.75)
    mask_a_ori = mask_a.reshape(16, 2, 2)
    mask_a = mask_expand2d(mask_a_ori, expand_ratio = 2)    # 16, 4, 4
    mask_a = mask_a.reshape(16, -1)
    mask_v = mask_v.unsqueeze(0)
    mask_a = mask_a.unsqueeze(0)
    if modality == 'video':
        mask_v = torch.cat([
            zeros,
            mask_v,
            zeros
        ], dim=-1)
        input_mask = mask_v
        input_x = v
    else:
        mask_a = torch.cat([
            zeros,
            mask_a, 
            zeros
        ], dim=-1)
        input_mask = mask_a
        input_x = a

    model = model.to(device)
    input_x = input_x.to(device)
    input_mask = input_mask.to(device)
    input_mask = input_mask.reshape(1, -1)
    input_mask = input_mask.repeat(B, 1)
    
    # 计算参数数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of params: {} M'.format(n_parameters / 1e6))
    
    # 计算FLOPs
    print(f"\nCalculating FLOPs for UniModalMamba ({modality})...")
    inputs = (input_x, input_mask)
    
    flops_results = {}
    
    # THOP method
    thop_results = calculate_flops_thop(model, inputs)
    if thop_results[0] is not None:
        flops_results['THOP'] = thop_results[:2]
        print(f"THOP - FLOPs: {thop_results[2]}, Params: {thop_results[3]}")
    
    # FVCore method
    fvcore_flops = calculate_flops_fvcore(model, inputs)
    if fvcore_flops is not None:
        flops_results['FVCore'] = fvcore_flops
        print(f"FVCore - FLOPs: {fvcore_flops/1e9:.2f}G")
    
    print_flops_summary(f"UniModalMamba ({modality})", flops_results, n_parameters)
    
    # 性能基准测试
    print("Running inference speed benchmark...")
    avg_time, fps = benchmark_inference_speed(model, inputs, num_runs=50, warmup_runs=5)
    print(f"Average inference time: {avg_time*1000:.2f}ms")
    print(f"Inference FPS: {fps:.2f}")
    
    # Test forward pass of Unimodal Mamba Model:
    print(f"input shape: {input_x.shape}, input mask shape: {input_mask.shape}")
    outputs = model(input_x, input_mask)
    print(f"Output Feats:", outputs['global_features'].shape)
    print(f"Output Recosn Feats:", outputs['features'].shape)

    print(f"modality {modality} forward pass successful!")


def test_vision_mamba(Flop_test = True):
    print("\n" + "=" * 50)
    print("Testing VisionMamba")
    print("=" * 50)
    
    num_frames = 16
    img_size = 224
    model = VisionMamba(
        img_size=224,
        patch_size=16,
        depth=32,                       # 默认depth，实际可根据模型定义调整
        embed_dim=768,                  # 对应clip_decoder_embed_dim
        channels=3,
        drop_path_rate=0.4,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        initializer_cfg=None,
        fused_add_norm=True,
        rms_norm=True,
        residual_in_fp32=True,
        bimamba=True,
        kernel_size=1,
        num_frames=16,
        device=None,
        dtype=None,
        use_checkpoint=False,
        checkpoint_num=0,
        clip_decoder_embed_dim=768,
        clip_output_dim=512,
        clip_norm_type='l2',
        clip_return_layer=1,
        clip_student_return_interval=1,
    )
    #model = videomamba_middle_pretrain(num_frames=num_frames)
    mask = torch.cat([
        torch.ones(1, 16 * int(14 * 14 * 0.75)),
        torch.zeros(1, 16 * int(14 * 14 * 0.25)),
    ], dim=-1).to(torch.bool)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    mask = mask.to(device)
    img = torch.rand(1, 3, num_frames, img_size, img_size).to(device)
    
    if Flop_test:
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Number of params: {} M'.format(n_parameters / 1e6))
        
        print("\nCalculating FLOPs for VisionMamba...")
        inputs = (img, mask)
        flops_results = {}
        # THOP method
        thop_results = calculate_flops_thop(model, inputs)
        if thop_results[0] is not None:
            flops_results['THOP'] = thop_results[:2]
            print(f"THOP - FLOPs: {thop_results[2]}, Params: {thop_results[3]}")
        
        # FVCore method
        fvcore_flops = calculate_flops_fvcore(model, inputs)
        if fvcore_flops is not None:
            flops_results['FVCore'] = fvcore_flops
            print(f"FVCore - FLOPs: {fvcore_flops/1e9:.2f}G")
        print_flops_summary("CrossMambaFT", flops_results, n_parameters)
        # Performance benchmark
        print("Running inference speed benchmark...")
        avg_time, fps = benchmark_inference_speed(model, inputs, num_runs=50, warmup_runs=5)
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Inference FPS: {fps:.2f}")
    
    output = model(img, mask)
    print(f"Output shape: {output[0].shape}")
    print("VisionMamba forward pass successful!")


def calculate_flops_thop(model, inputs):
    """使用thop库计算FLOPs"""
    if not THOP_AVAILABLE:
        return None, None
    
    try:
        if isinstance(inputs, (list, tuple)):
            flops, params = profile(model, inputs=inputs, verbose=False)
        else:
            flops, params = profile(model, inputs=(inputs,), verbose=False)
        
        flops_readable = clever_format([flops], "%.3f")
        params_readable = clever_format([params], "%.3f")
        return flops, params, flops_readable[0], params_readable[0]
    except Exception as e:
        print(f"THOP calculation failed: {e}")
        return None, None, None, None


def calculate_flops_fvcore(model, inputs):
    """使用fvcore库计算FLOPs"""
    if not FVCORE_AVAILABLE:
        return None
    
    try:
        if isinstance(inputs, (list, tuple)):
            flop_dict = flop_count(model, inputs, supported_ops=None)
        else:
            flop_dict = flop_count(model, (inputs,), supported_ops=None)
        
        total_flops = sum(flop_dict.values())
        return total_flops
    except Exception as e:
        print(f"FVCore calculation failed (likely due to Triton/JIT compatibility): {e}")
        return None


def calculate_model_complexity(model, input_shape, input_constructor=None):
    """使用ptflops库计算模型复杂度"""
    if not PTFLOPS_AVAILABLE:
        return None, None
    
    try:
        macs, params = get_model_complexity_info(
            model, 
            input_shape, 
            input_constructor=input_constructor,
            as_strings=True,
            print_per_layer_stat=False,
            verbose=False
        )
        return macs, params
    except Exception as e:
        print(f"PTFlops calculation failed: {e}")
        return None, None


def print_flops_summary(model_name: str, flops_results: Dict[str, Any], params_count: int = None):
    """打印FLOPs计算结果摘要"""
    print(f"\n{'='*60}")
    print(f"FLOPs Analysis for {model_name}")
    print(f"{'='*60}")
    
    if params_count is not None:
        print(f"Parameters: {params_count:,} ({params_count/1e6:.2f}M)")
    
    for method, results in flops_results.items():
        if results is not None:
            if isinstance(results, tuple) and len(results) >= 2:
                flops, params = results[0], results[1]
                if flops is not None:
                    print(f"{method} FLOPs: {flops:,} ({flops/1e9:.2f}G)")
                if params is not None:
                    print(f"{method} Params: {params:,} ({params/1e6:.2f}M)")
            else:
                print(f"{method} FLOPs: {results:,} ({results/1e9:.2f}G)")
        else:
            print(f"{method}: Not available")
    
    print(f"{'='*60}\n")

def benchmark_inference_speed(model, inputs, num_runs=100, warmup_runs=10):
    model.eval()
    device = next(model.parameters()).device

    # Ensure inputs on correct device
    if isinstance(inputs, (list, tuple)):
        inputs = [x.to(device) for x in inputs]
    else:
        inputs = inputs.to(device)

    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)

    # Accurate timing using CUDA events
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    timings = []

    with torch.no_grad():
        for _ in range(num_runs):
            starter.record()
            _ = model(*inputs) if isinstance(inputs, (list, tuple)) else model(inputs)
            ender.record()
            torch.cuda.synchronize()
            elapsed_time_ms = starter.elapsed_time(ender)
            timings.append(elapsed_time_ms)

    avg_time_ms = sum(timings) / len(timings)
    fps = 1000.0 / avg_time_ms
    return avg_time_ms / 1000, fps  # Return in seconds




def test_all_models_flops():
    """测试所有模型的FLOPs并进行比较"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE FLOPs ANALYSIS FOR ALL MODELS")
    print("=" * 80)
    
    results_summary = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Test CrossMamba
    try:
        print("\n1. Testing CrossMamba (Pre-training)...")
        model = CrossMamba(num_frames=16, audio_length=1024)
        model = model.to(device)
        
        B = 1  # Use batch size 1 for FLOPs calculation
        v = torch.randn(B, 3, 16, 224, 224).to(device)
        a = torch.randn(B, 64, 1024).to(device)
        
        # Create mask
        zeros = torch.zeros(16, 1).bool()
        mask_v = rand_mask_generate(16, 196, 0.75)
        mask_a = rand_mask_generate(16, 64 // 16, 0.75)
        mask_a_ori = mask_a.reshape(16, 2, 2)
        mask_a = mask_expand2d(mask_a_ori, expand_ratio=2)
        mask_a = mask_a.reshape(16, -1)
        mask = torch.cat([zeros, mask_v, zeros, zeros, mask_a, zeros], dim=1)
        mask = mask.reshape(1, -1).repeat(B, 1).to(device)
        
        inputs = (v, a, mask)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        thop_results = calculate_flops_thop(model, inputs)
        fvcore_flops = calculate_flops_fvcore(model, inputs)
        
        results_summary['CrossMamba'] = {
            'params': n_params,
            'thop_flops': thop_results[0] if thop_results[0] else None,
            'fvcore_flops': fvcore_flops
        }
        
        print(f"CrossMamba - Params: {n_params/1e6:.2f}M")
        if thop_results[0]:
            print(f"CrossMamba - THOP FLOPs: {thop_results[0]/1e9:.2f}G")
        if fvcore_flops:
            print(f"CrossMamba - FVCore FLOPs: {fvcore_flops/1e9:.2f}G")
            
    except Exception as e:
        print(f"CrossMamba test failed: {e}")
        results_summary['CrossMamba'] = {'error': str(e)}
    
    # 2. Test CrossMambaFT
    try:
        print("\n2. Testing CrossMambaFT (Fine-tuning)...")
        model_ft = CrossMambaFT(num_classes=700, fc_drop_rate=0.1).to(device)
        
        B = 1
        v = torch.randn(B, 3, 16, 224, 224).to(device)
        a = torch.randn(B, 64, 1024).to(device)
        inputs = (v, a)
        
        n_params = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        thop_results = calculate_flops_thop(model_ft, inputs)
        fvcore_flops = calculate_flops_fvcore(model_ft, inputs)
        
        results_summary['CrossMambaFT'] = {
            'params': n_params,
            'thop_flops': thop_results[0] if thop_results[0] else None,
            'fvcore_flops': fvcore_flops
        }
        
        print(f"CrossMambaFT - Params: {n_params/1e6:.2f}M")
        if thop_results[0]:
            print(f"CrossMambaFT - THOP FLOPs: {thop_results[0]/1e9:.2f}G")
        if fvcore_flops:
            print(f"CrossMambaFT - FVCore FLOPs: {fvcore_flops/1e9:.2f}G")
            
    except Exception as e:
        print(f"CrossMambaFT test failed: {e}")
        results_summary['CrossMambaFT'] = {'error': str(e)}
    
    # 3. Test UniModalMamba (Audio)
    try:
        print("\n3. Testing UniModalMamba (Audio)...")
        model_uni_a = UniModalMamba().to(device)
        
        B = 1
        a = torch.randn(B, 64, 1024).to(device)
        
        # Create audio mask
        zeros = torch.zeros(1, 16, 1).bool()
        mask_a = rand_mask_generate(16, 4, 0.75)
        mask_a_ori = mask_a.reshape(16, 2, 2)
        mask_a = mask_expand2d(mask_a_ori, expand_ratio=2)
        mask_a = mask_a.reshape(16, -1)
        mask_a = mask_a.unsqueeze(0)
        mask_a = torch.cat([zeros, mask_a, zeros], dim=-1)
        mask_a = mask_a.reshape(1, -1).repeat(B, 1).to(device)
        
        inputs = (a, mask_a)
        n_params = sum(p.numel() for p in model_uni_a.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        thop_results = calculate_flops_thop(model_uni_a, inputs)
        fvcore_flops = calculate_flops_fvcore(model_uni_a, inputs)
        
        results_summary['UniModalMamba_Audio'] = {
            'params': n_params,
            'thop_flops': thop_results[0] if thop_results[0] else None,
            'fvcore_flops': fvcore_flops
        }
        
        print(f"UniModalMamba (Audio) - Params: {n_params/1e6:.2f}M")
        if thop_results[0]:
            print(f"UniModalMamba (Audio) - THOP FLOPs: {thop_results[0]/1e9:.2f}G")
        if fvcore_flops:
            print(f"UniModalMamba (Audio) - FVCore FLOPs: {fvcore_flops/1e9:.2f}G")
            
    except Exception as e:
        print(f"UniModalMamba (Audio) test failed: {e}")
        results_summary['UniModalMamba_Audio'] = {'error': str(e)}
    
    # 4. Test UniModalMamba (Video)
    try:
        print("\n4. Testing UniModalMamba (Video)...")
        model_uni_v = UniModalMamba().to(device)
        
        B = 1
        v = torch.randn(B, 3, 16, 224, 224).to(device)
        
        # Create video mask
        zeros = torch.zeros(1, 16, 1).bool()
        mask_v = rand_mask_generate(16, 196, 0.75)
        mask_v = mask_v.unsqueeze(0)
        mask_v = torch.cat([zeros, mask_v, zeros], dim=-1)
        mask_v = mask_v.reshape(1, -1).repeat(B, 1).to(device)
        
        inputs = (v, mask_v)
        n_params = sum(p.numel() for p in model_uni_v.parameters() if p.requires_grad)
        
        # Calculate FLOPs
        thop_results = calculate_flops_thop(model_uni_v, inputs)
        fvcore_flops = calculate_flops_fvcore(model_uni_v, inputs)
        
        results_summary['UniModalMamba_Video'] = {
            'params': n_params,
            'thop_flops': thop_results[0] if thop_results[0] else None,
            'fvcore_flops': fvcore_flops
        }
        
        print(f"UniModalMamba (Video) - Params: {n_params/1e6:.2f}M")
        if thop_results[0]:
            print(f"UniModalMamba (Video) - THOP FLOPs: {thop_results[0]/1e9:.2f}G")
        if fvcore_flops:
            print(f"UniModalMamba (Video) - FVCore FLOPs: {fvcore_flops/1e9:.2f}G")
            
    except Exception as e:
        print(f"UniModalMamba (Video) test failed: {e}")
        results_summary['UniModalMamba_Video'] = {'error': str(e)}
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("FINAL COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<25} {'Params (M)':<12} {'THOP FLOPs (G)':<15} {'FVCore FLOPs (G)':<15}")
    print("-" * 80)
    
    for model_name, results in results_summary.items():
        if 'error' in results:
            print(f"{model_name:<25} {'ERROR':<12} {'ERROR':<15} {'ERROR':<15}")
        else:
            params_str = f"{results['params']/1e6:.2f}" if results['params'] else "N/A"
            thop_str = f"{results['thop_flops']/1e9:.2f}" if results['thop_flops'] else "N/A"
            fvcore_str = f"{results['fvcore_flops']/1e9:.2f}" if results['fvcore_flops'] else "N/A"
            print(f"{model_name:<25} {params_str:<12} {thop_str:<15} {fvcore_str:<15}")
    
    return results_summary

def test_hf_mamba_model(Flop_test=True):
    print("\n" + "=" * 50)
    print("Testing HuggingFace MambaModel")
    print("=" * 50)

    config = MambaConfig(
        d_model=768,
        n_layer=8,
        vocab_size=32000,  # required if using LM head
        residual_in_fp32=True,
        fused_add_norm=True,
    )
    model = MambaModel(config)

    # 输入构造
    B, L = 1, 1024  # batch size, sequence length
    x = torch.randint(0, config.vocab_size, (B, L))  # token ids
    attention_mask = torch.ones(B, L)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    x = x.to(device)
    attention_mask = attention_mask.to(device)

    # 参数量
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: {:.2f}M'.format(n_parameters / 1e6))

    if Flop_test:
        inputs = (x,)

        flops_results = {}
        # THOP
        thop_results = calculate_flops_thop(model, inputs)
        if thop_results[0] is not None:
            flops_results['THOP'] = thop_results[:2]
            print(f"THOP - FLOPs: {thop_results[2]}, Params: {thop_results[3]}")
        # FVCore
        fvcore_flops = calculate_flops_fvcore(model, inputs)
        if fvcore_flops is not None:
            flops_results['FVCore'] = fvcore_flops
            print(f"FVCore - FLOPs: {fvcore_flops / 1e9:.2f}G")

        print_flops_summary("HuggingFace MambaModel", flops_results, n_parameters)

        # 推理速度
        print("Running inference speed benchmark...")
        avg_time, fps = benchmark_inference_speed(model, inputs, num_runs=50, warmup_runs=5)
        print(f"Average inference time: {avg_time*1000:.2f} ms")
        print(f"Inference FPS: {fps:.2f}")

    # 正向传播
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=x, attention_mask=attention_mask)
    hidden_states = outputs.last_hidden_state
    print("Output shape:", hidden_states.shape)  # (B, L, d_model)


def test_unimodal_mambaFT(modality = 'audio', Flop_test=True):
    print("\n" + "=" * 50)
    print("Testing uni-modality mamba FT, current modality is: {}".format(modality))
    print("=" * 50)
    model_ft = UniModalMamba_FT(num_classes=527)
    B = 2  
    device = torch.device("cuda")
    # Test inputs
    v = torch.randn(B, 3, 16, 224, 224)  # Video input
    a = torch.randn(B, 64, 1024)         # Audio input
    model_ft = model_ft.to(device)
    v = v.to(device)
    a = a.to(device)
    if modality == 'video':
        input_x = v
    else:
        input_x = a

    if Flop_test:
        # 计算参数数量
        n_parameters = sum(p.numel() for p in model_ft.parameters() if p.requires_grad)
        print('Number of params: {} M'.format(n_parameters / 1e6))
        
        # 计算FLOPs
        print("\nCalculating FLOPs for CrossMambaFT...")
        inputs = input_x
        
        flops_results = {}
        
        # THOP method
        thop_results = calculate_flops_thop(model_ft, inputs)
        if thop_results[0] is not None:
            flops_results['THOP'] = thop_results[:2]
            print(f"THOP - FLOPs: {thop_results[2]}, Params: {thop_results[3]}")
        
        # FVCore method
        fvcore_flops = calculate_flops_fvcore(model_ft, inputs)
        if fvcore_flops is not None:
            flops_results['FVCore'] = fvcore_flops
            print(f"FVCore - FLOPs: {fvcore_flops/1e9:.2f}G")
        
        print_flops_summary("CrossMambaFT", flops_results, n_parameters)
        
        # 性能基准测试
        print("Running inference speed benchmark...")
        avg_time, fps = benchmark_inference_speed(model_ft, inputs, num_runs=50, warmup_runs=5)
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Inference FPS: {fps:.2f}")
    # Test forward pass of Unimodal
    print(f"input shape: {input_x.shape}")
    outputs = model_ft(input_x)
    print(f"Output Feats:", outputs['global_feature'].shape)
    print(f"Pred logits:", outputs['logits'].shape)


if __name__ == "__main__":
    # 选择测试模式
    import sys
    
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
    else:
        test_mode = "flops"  # 默认测试FLOPs
    
    if test_mode == "clip":
        test_clip_teacher()
    elif test_mode == "clap":
        test_clap_teacher()
    elif test_mode == "cross_mamba":
        test_cross_mamba()
    elif test_mode == "cross_mamba_ft":
        test_cross_mamba_ft()
    elif test_mode == "d":
        test_unimodal_mamba(modality='audio')
    elif test_mode == "uni_video":
        test_unimodal_mamba(modality='video')
    elif test_mode == "vision_mamba":
        test_vision_mamba()
    elif test_mode == "hf_mamba":
        test_hf_mamba_model()
    elif test_mode == "uni_mamba_ft":
        test_unimodal_mambaFT(modality='audio')
    elif test_mode == "flops" or test_mode == "all":
        # 运行完整的FLOPs测试
        print("For comprehensive FLOPs analysis, please run:")
        print("python test_flops.py")
        print("\nOr run individual model tests with FLOPs analysis:")
        test_cross_mamba()
    else:
        print("Available test modes:")
        print("  clip - Test CLIP teacher model")
        print("  clap - Test CLAP teacher model")    
        print("  cross_mamba - Test CrossMamba model")
        print("  cross_mamba_ft - Test CrossMambaFT model")
        print("  uni_audio - Test UniModalMamba (audio)")
        print("  uni_video - Test UniModalMamba (video)")
        print("  flops/all - Show FLOPs test instructions")
        print("\nUsage: python model_pipeline.py [test_mode]")
        print("For comprehensive FLOPs analysis: python test_flops.py")
