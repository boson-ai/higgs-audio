# Higgs Audio Installation Guide

This guide provides comprehensive instructions for installing and setting up Higgs Audio v2, including the optional Gradio web interface.

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start](#quick-start)
- [Detailed Installation](#detailed-installation)
- [Gradio Web Interface](#gradio-web-interface)
- [Troubleshooting](#troubleshooting)
- [Advanced Configuration](#advanced-configuration)

## System Requirements

### Minimum Requirements
- **OS**: Linux, macOS, or Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB (16GB recommended)
- **Storage**: 10GB free space

### Recommended Requirements
- **GPU**: NVIDIA GPU with 24GB+ VRAM (for optimal performance)
- **RAM**: 32GB or more
- **Storage**: SSD with 20GB+ free space

### GPU Support
- **CUDA**: 11.8 or higher
- **PyTorch**: 2.0.0 or higher with CUDA support

## Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio
```

### 2. Install Core Dependencies
```bash
pip install -r requirements.txt
pip install -e .
```

### 3. Test Installation
```bash
python examples/generation.py \
  --transcript "Hello, this is a test." \
  --out_path test_output.wav
```

## Detailed Installation

### Option 1: Direct Installation (Recommended)

```bash
# Clone repository
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "from boson_multimodal import BosonMultimodal; print('Installation successful!')"
```

### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv higgs_audio_env
source higgs_audio_env/bin/activate  # On Windows: higgs_audio_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 3: Conda Environment

```bash
# Create conda environment
conda create -y --prefix ./conda_env --override-channels --strict-channel-priority --channel "conda-forge" "python==3.10.*"
conda activate ./conda_env

# Install dependencies
pip install -r requirements.txt
pip install -e .

# Deactivate when done
conda deactivate
conda remove -y --prefix ./conda_env --all
```

### Option 4: Docker (Recommended for GPU)

```bash
# Pull NVIDIA PyTorch container
docker pull nvcr.io/nvidia/pytorch:25.02-py3

# Run container with GPU support
docker run --gpus all --ipc=host --net=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm nvcr.io/nvidia/pytorch:25.02-py3 bash

# Inside container, clone and install
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio
pip install -r requirements.txt
pip install -e .
```

## Gradio Web Interface

The Gradio web interface provides a user-friendly way to interact with Higgs Audio without command-line tools.

### Installation

1. **Install Core Dependencies** (if not already done):
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

2. **Install Gradio Dependencies**:
   ```bash
   pip install -r requirements-gradio.txt
   ```

3. **For GPU Support** (optional):
   ```bash
   # For CUDA 12.4
   pip install torch==2.6.0+cu124 torchvision==0.21.0 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   
   # For CUDA 12.6
   pip install torch==2.6.0+cu126 torchvision==0.21.0 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
   ```

### Usage

1. **Start the Web Interface**:
   ```bash
   # Full mode (recommended)
   python examples/gradio_app.py --mode full
   
   # Light mode (for limited resources)
   python examples/gradio_app.py --mode light
   
   # Demo mode (for quick testing)
   python examples/gradio_app.py --mode demo
   ```

2. **Access the Interface**:
   - Open your web browser
   - Navigate to `http://127.0.0.1:7860`
   - Start generating speech!

### Interface Modes

- **Full Mode**: All features enabled, maximum parameter ranges
- **Light Mode**: Reduced features, optimized for limited resources
- **Demo Mode**: Limited text length, auto-generation, best for demonstrations

## Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues

**Problem**: CUDA not found or GPU not detected
```bash
# Solution: Install CPU-only PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

**Problem**: CUDA out of memory
```bash
# Solutions:
# 1. Use light mode in Gradio
python examples/gradio_app.py --mode light

# 2. Reduce batch size
# 3. Close other GPU applications
```

#### 2. Model Download Issues

**Problem**: Model download fails
```bash
# Solution: Check internet connection and try again
# The model will be downloaded automatically on first use
```

#### 3. Memory Issues

**Problem**: System runs out of memory
```bash
# Solutions:
# 1. Use light mode
# 2. Close unnecessary applications
# 3. Use shorter text inputs
# 4. Increase system swap space
```

#### 4. Port Conflicts

**Problem**: Port 7860 already in use
```bash
# Solution: Use different port
python examples/gradio_app.py --port 7861
```

### Performance Optimization

#### GPU Optimization
```bash
# Install appropriate CUDA version
# For CUDA 12.4
pip install torch==2.6.0+cu124 torchvision==0.21.0 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.6
pip install torch==2.6.0+cu126 torchvision==0.21.0 torchaudio==2.6.0+cu126 --index-url https://download.pytorch.org/whl/cu126
```

#### Memory Optimization
- Use light mode for systems with limited RAM
- Close unnecessary applications
- Use shorter text inputs
- Consider using CPU mode if GPU memory is insufficient

## Advanced Configuration

### Environment Variables

```bash
# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Set model cache directory
export HF_HOME=/path/to/cache

# Set logging level
export LOG_LEVEL=INFO
```

### Custom Voice Prompts

1. **Add Your Own Voices**:
   ```bash
   # Place .wav files in examples/voice_prompts/
   cp your_voice.wav examples/voice_prompts/
   ```

2. **Voice Requirements**:
   - Format: WAV, MP3, or FLAC
   - Duration: 3-30 seconds
   - Quality: Clear, noise-free audio
   - Content: Natural speech, no music or effects

### Batch Processing

For processing multiple files:

```bash
# Create a batch script
for file in transcripts/*.txt; do
    python examples/generation.py \
        --transcript "$(cat $file)" \
        --out_path "output/$(basename $file .txt).wav" \
        --temperature 0.3
done
```

## Support

### Getting Help

1. **Check the Documentation**:
   - [Main README](README.md)
   - [Gradio README](examples/README_gradio.md)
   - [Examples](examples/)

2. **Common Solutions**:
   - Restart the application
   - Check system resources
   - Verify internet connection
   - Update dependencies

3. **Report Issues**:
   - Create an issue on GitHub
   - Include system information and error messages
   - Provide steps to reproduce the problem

### System Information

To help with troubleshooting, include this information:

```bash
# Python version
python --version

# PyTorch version and CUDA support
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# GPU information
nvidia-smi

# System memory
free -h  # Linux
# or
systeminfo | findstr "Total Physical Memory"  # Windows
```

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details. 