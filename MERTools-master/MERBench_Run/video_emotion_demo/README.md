# Video Emotion Recognition Demo

这个目录包含了一个基于 AttentionRobustV2 模型的视频情绪识别演示 Demo。

## 特点
- **轻量化/中量级特征提取**:
  - **Text**: `bert-base-chinese` (768维) 替代了 Baichuan-13B
  - **Audio**: `chinese-hubert-base` (768维) 替代了 chinese-hubert-large
  - **Video**: `openai/clip-vit-base-patch32` (512维) 替代了 clip-vit-large

## 目录结构
- `demo.py`: 主运行脚本，负责视频处理、特征提取和模型推理。
- `feature_extractor.py`: 封装了 HuggingFace 模型的特征提取逻辑。

## 环境准备
确保已安装以下依赖（已在环境中安装）：
```bash
pip install librosa opencv-python-headless transformers torch numpy
```
以及系统需安装 `ffmpeg` (已安装)。

## 运行方法

### 1. 准备视频
准备一个包含人声的视频文件（例如 `test.mp4`）。

### 2. 运行 Demo
```bash
python demo.py --video_path /path/to/your/video.mp4 --text "这里是视频里说的话"
```
*注意：由于环境中没有自动语音识别 (ASR) 工具，建议手动提供文本参数 `--text` 以获得更准确的情绪识别结果。如果不提供，将使用空文本特征。*

### 3. 关于模型权重
**重要提示**：由于更换了特征提取器（输入维度和特征空间发生了变化），**不能直接使用**之前基于 Baichuan/Large 模型训练的权重文件。
本 Demo 默认使用**随机初始化**的模型参数运行，仅用于演示 pipeline 的流程。输出的预测结果是**随机**的。

如果要获得真实结果，需要：
1. 使用本 Demo 中的 `FeatureExtractor` 对整个数据集（MER2023）提取特征。
2. 使用提取的新特征重新训练 `AttentionRobustV2` 模型。
3. 使用训练好的权重运行本 Demo。
