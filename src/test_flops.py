#!/usr/bin/env python3
"""
FLOPs测试脚本 - 用于测试CAV-MAE项目中各个模型的FLOPs
"""

import warnings
import torch
import time
import sys
import subprocess
from typing import Dict, Any

warnings.filterwarnings("ignore", category=FutureWarning)

# 导入模型
from models.mamba_pretrain import CrossMamba, CrossMambaFT, UniModalMamba
from dataloader import rand_mask_generate, mask_expand2d

def install_packages():
    """安装FLOPs计算所需的包"""
    packages = ['thop', 'fvcore', 'ptflops']
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✓ {package} installed successfully")
            except subprocess.CalledProcessError:
                print(f"✗ Failed to install {package}")

def import_flops_libraries():
    """动态导入FLOPs计算库"""
    libraries = {}
    
    try:
        from thop import profile, clever_format
        libraries['thop'] = {'profile': profile, 'clever_format': clever_format}
        print("✓ THOP library imported successfully")
    except ImportError:
        print("✗ THOP library not available")
        libraries['thop'] = None
    
    try:
        from fvcore.nn import FlopCountMode, flop_count
        libraries['fvcore'] = {'flop_count': flop_count}
        print("✓ FVCore library imported successfully")
    except ImportError:
        print("✗ FVCore library not available")
        libraries['fvcore'] = None
    
    try:
        from ptflops import get_model_complexity_info
        libraries['ptflops'] = {'get_model_complexity_info': get_model_complexity_info}
        print("✓ PTFlops library imported successfully")
    except ImportError:
        print("✗ PTFlops library not available")
        libraries['ptflops'] = None
    
    return libraries

def calculate_flops_thop(model, inputs, libs):
    """使用thop库计算FLOPs"""
    if libs['thop'] is None:
        return None, None, None, None
    
    try:
        profile = libs['thop']['profile']
        clever_format = libs['thop']['clever_format']
        
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

def calculate_flops_fvcore(model, inputs, libs):
    """使用fvcore库计算FLOPs"""
    if libs['fvcore'] is None:
        return None
    
    try:
        flop_count = libs['fvcore']['flop_count']
        
        if isinstance(inputs, (list, tuple)):
            flop_dict = flop_count(model, inputs, supported_ops=None)
        else:
            flop_dict = flop_count(model, (inputs,), supported_ops=None)
        
        total_flops = sum(flop_dict.values())
        return total_flops
    except Exception as e:
        print(f"FVCore calculation failed: {e}")
        return None

def benchmark_inference_speed(model, inputs, num_runs=50, warmup_runs=5):
    """测试模型推理速度"""
    model.eval()
    device = next(model.parameters()).device
    
    # 确保输入在正确的设备上
    if isinstance(inputs, (list, tuple)):
        inputs = [inp.to(device) if hasattr(inp, 'to') else inp for inp in inputs]
    else:
        inputs = inputs.to(device) if hasattr(inputs, 'to') else inputs
    
    # 预热
    with torch.no_grad():
        for _ in range(warmup_runs):
            try:
                if isinstance(inputs, (list, tuple)):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)
            except:
                pass
    
    # 正式测试
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            try:
                if isinstance(inputs, (list, tuple)):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)
            except:
                pass
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    fps = 1.0 / avg_time
    
    return avg_time, fps

def test_model_flops(model_name, model, inputs, libs):
    """测试单个模型的FLOPs"""
    print(f"\n{'='*60}")
    print(f"Testing {model_name}")
    print(f"{'='*60}")
    
    # 计算参数数量
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    
    # THOP FLOPs计算
    thop_results = calculate_flops_thop(model, inputs, libs)
    if thop_results[0] is not None:
        print(f"THOP FLOPs: {thop_results[2]} ({thop_results[0]/1e9:.2f}G)")
        print(f"THOP Params: {thop_results[3]} ({thop_results[1]/1e6:.2f}M)")
    else:
        print("THOP FLOPs: Not available")
    
    # FVCore FLOPs计算
    fvcore_flops = calculate_flops_fvcore(model, inputs, libs)
    if fvcore_flops is not None:
        print(f"FVCore FLOPs: {fvcore_flops:,} ({fvcore_flops/1e9:.2f}G)")
    else:
        print("FVCore FLOPs: Not available")
    
    # 推理速度测试
    print("\nBenchmarking inference speed...")
    try:
        avg_time, fps = benchmark_inference_speed(model, inputs)
        print(f"Average inference time: {avg_time*1000:.2f}ms")
        print(f"Inference FPS: {fps:.2f}")
    except Exception as e:
        print(f"Speed benchmark failed: {e}")
    
    return {
        'params': n_params,
        'thop_flops': thop_results[0] if thop_results[0] else None,
        'fvcore_flops': fvcore_flops,
        'avg_time': avg_time if 'avg_time' in locals() else None,
        'fps': fps if 'fps' in locals() else None
    }

def test_all_models():
    """测试所有模型的FLOPs"""
    print("CAV-MAE Models FLOPs Analysis")
    print("="*80)
    
    # 安装和导入库
    print("Setting up FLOPs calculation libraries...")
    install_packages()
    libs = import_flops_libraries()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {}
    
    # 1. Test CrossMamba
    try:
        print(f"\n{'-'*20} CrossMamba (Pre-training) {'-'*20}")
        model = CrossMamba(num_frames=16, audio_length=1024).to(device)
        
        B = 1
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
        results['CrossMamba'] = test_model_flops('CrossMamba', model, inputs, libs)
        
    except Exception as e:
        print(f"CrossMamba test failed: {e}")
        results['CrossMamba'] = {'error': str(e)}
    
    # 2. Test CrossMambaFT
    try:
        print(f"\n{'-'*20} CrossMambaFT (Fine-tuning) {'-'*20}")
        model = CrossMambaFT(num_classes=700, fc_drop_rate=0.1).to(device)
        
        B = 1
        v = torch.randn(B, 3, 16, 224, 224).to(device)
        a = torch.randn(B, 64, 1024).to(device)
        inputs = (v, a)
        
        results['CrossMambaFT'] = test_model_flops('CrossMambaFT', model, inputs, libs)
        
    except Exception as e:
        print(f"CrossMambaFT test failed: {e}")
        results['CrossMambaFT'] = {'error': str(e)}
    
    # 3. Test UniModalMamba (Audio)
    try:
        print(f"\n{'-'*20} UniModalMamba (Audio) {'-'*20}")
        model = UniModalMamba().to(device)
        
        B = 1
        a = torch.randn(B, 64, 1024).to(device)
        
        zeros = torch.zeros(1, 16, 1).bool()
        mask_a = rand_mask_generate(16, 4, 0.75)
        mask_a_ori = mask_a.reshape(16, 2, 2)
        mask_a = mask_expand2d(mask_a_ori, expand_ratio=2)
        mask_a = mask_a.reshape(16, -1)
        mask_a = mask_a.unsqueeze(0)
        mask_a = torch.cat([zeros, mask_a, zeros], dim=-1)
        mask_a = mask_a.reshape(1, -1).repeat(B, 1).to(device)
        
        inputs = (a, mask_a)
        results['UniModalMamba_Audio'] = test_model_flops('UniModalMamba (Audio)', model, inputs, libs)
        
    except Exception as e:
        print(f"UniModalMamba (Audio) test failed: {e}")
        results['UniModalMamba_Audio'] = {'error': str(e)}
    
    # 4. Test UniModalMamba (Video)
    try:
        print(f"\n{'-'*20} UniModalMamba (Video) {'-'*20}")
        model = UniModalMamba().to(device)
        
        B = 1
        v = torch.randn(B, 3, 16, 224, 224).to(device)
        
        zeros = torch.zeros(1, 16, 1).bool()
        mask_v = rand_mask_generate(16, 196, 0.75)
        mask_v = mask_v.unsqueeze(0)
        mask_v = torch.cat([zeros, mask_v, zeros], dim=-1)
        mask_v = mask_v.reshape(1, -1).repeat(B, 1).to(device)
        
        inputs = (v, mask_v)
        results['UniModalMamba_Video'] = test_model_flops('UniModalMamba (Video)', model, inputs, libs)
        
    except Exception as e:
        print(f"UniModalMamba (Video) test failed: {e}")
        results['UniModalMamba_Video'] = {'error': str(e)}
    
    # Print comprehensive summary
    print_summary(results)
    
    return results

def print_summary(results):
    """打印结果摘要"""
    print("\n" + "=" * 100)
    print("COMPREHENSIVE FLOPS ANALYSIS SUMMARY")
    print("=" * 100)
    
    header = f"{'Model':<25} {'Params (M)':<12} {'THOP FLOPs (G)':<15} {'FVCore FLOPs (G)':<18} {'Avg Time (ms)':<15} {'FPS':<10}"
    print(header)
    print("-" * 100)
    
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name:<25} {'ERROR':<12} {'ERROR':<15} {'ERROR':<18} {'ERROR':<15} {'ERROR':<10}")
        else:
            params_str = f"{result['params']/1e6:.2f}" if result['params'] else "N/A"
            thop_str = f"{result['thop_flops']/1e9:.2f}" if result['thop_flops'] else "N/A"
            fvcore_str = f"{result['fvcore_flops']/1e9:.2f}" if result['fvcore_flops'] else "N/A"
            time_str = f"{result['avg_time']*1000:.2f}" if result['avg_time'] else "N/A"
            fps_str = f"{result['fps']:.2f}" if result['fps'] else "N/A"
            
            print(f"{model_name:<25} {params_str:<12} {thop_str:<15} {fvcore_str:<18} {time_str:<15} {fps_str:<10}")
    
    print("=" * 100)

if __name__ == "__main__":
    try:
        results = test_all_models()
        print("\n✓ FLOPs analysis completed successfully!")
    except Exception as e:
        print(f"\n✗ FLOPs analysis failed: {e}")
        import traceback
        traceback.print_exc()
