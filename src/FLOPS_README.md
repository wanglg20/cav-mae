# CAV-MAE Models FLOPs Analysis

本项目提供了用于测试CAV-MAE中各个模型FLOPs（浮点运算次数）的工具。

## 安装依赖

首先确保安装了FLOPs计算所需的库：

```bash
pip install thop fvcore ptflops
```

或者运行脚本时会自动安装。

## 使用方法

### 1. 完整FLOPs分析（推荐）

运行独立的FLOPs测试脚本：

```bash
python test_flops.py
```

这将测试所有模型并生成完整的比较报告。

### 2. 单独测试特定模型

使用model_pipeline.py测试特定模型：

```bash
# 测试CrossMamba模型（带FLOPs计算）
python model_pipeline.py cross_mamba

# 测试CrossMambaFT模型（带FLOPs计算）
python model_pipeline.py cross_mamba_ft

# 测试UniModalMamba音频模型（带FLOPs计算）
python model_pipeline.py uni_audio

# 测试UniModalMamba视频模型（带FLOPs计算）
python model_pipeline.py uni_video

# 测试CLIP教师模型
python model_pipeline.py clip

# 测试CLAP教师模型
python model_pipeline.py clap
```

## 测试的模型

1. **CrossMamba (Pre-training)**: 跨模态预训练模型
   - 输入：视频 (3×16×224×224) + 音频 (64×1024) + 掩码
   
2. **CrossMambaFT (Fine-tuning)**: 跨模态微调模型
   - 输入：视频 (3×16×224×224) + 音频 (64×1024)
   - 输出：700类分类结果
   
3. **UniModalMamba (Audio)**: 单模态音频模型
   - 输入：音频 (64×1024) + 掩码
   
4. **UniModalMamba (Video)**: 单模态视频模型
   - 输入：视频 (3×16×224×224) + 掩码

## FLOPs计算方法

本项目使用三种不同的库来计算FLOPs，以确保结果的准确性：

1. **THOP**: 基于PyTorch的FLOPs分析工具
2. **FVCore**: Facebook研究团队开发的计算机视觉核心库
3. **PTFlops**: 专门用于PyTorch模型的FLOPs计算

## 输出结果

测试完成后，你将看到包含以下信息的详细报告：

- **Parameters**: 模型参数数量（以百万为单位）
- **THOP FLOPs**: 使用THOP库计算的FLOPs（以十亿为单位）
- **FVCore FLOPs**: 使用FVCore库计算的FLOPs（以十亿为单位）
- **Average Time**: 平均推理时间（毫秒）
- **FPS**: 每秒处理帧数

### 示例输出

```
================================================================================================
COMPREHENSIVE FLOPS ANALYSIS SUMMARY
================================================================================================
Model                     Params (M)   THOP FLOPs (G)  FVCore FLOPs (G)   Avg Time (ms)   FPS       
---------------------------------------------------------------------------------------------
CrossMamba                123.45       67.89           65.43              45.67           21.90     
CrossMambaFT             125.67       69.12           66.78              47.23           21.17     
UniModalMamba_Audio       89.34        23.45           22.98              15.43           64.84     
UniModalMamba_Video       91.56        45.67           44.32              28.91           34.59     
================================================================================================
```

## 注意事项

1. **GPU推荐**: 建议在GPU环境下运行测试以获得准确的推理速度结果
2. **内存需求**: 某些模型可能需要较大的GPU内存
3. **批处理大小**: FLOPs计算使用批处理大小为1，这是标准做法
4. **模型权重**: 测试使用随机初始化的权重，不影响FLOPs计算结果

## 故障排除

如果遇到导入错误：

```bash
# 手动安装依赖
pip install thop fvcore ptflops

# 如果仍有问题，尝试更新pip
pip install --upgrade pip
pip install thop fvcore ptflops
```

如果遇到CUDA内存不足：

```bash
# 可以修改test_flops.py中的批处理大小
B = 1  # 已经是最小值
```

## 扩展使用

你可以轻松修改脚本来测试其他模型或不同的输入尺寸：

1. 在`test_flops.py`中添加新的模型测试函数
2. 修改输入张量的尺寸以测试不同分辨率的影响
3. 调整批处理大小来测试批处理对FLOPs的影响（注意：FLOPs通常与批处理大小成线性关系）

## 联系信息

如有问题或建议，请联系项目维护者。
