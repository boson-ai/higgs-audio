# 🚀 Higgs Audio V2 - Installation Guide

Complete guide for installing and running Higgs Audio V2 with Gradio web interface.

## 📋 System Requirements

### Minimum Requirements
- **RAM**: 8GB (16GB+ recommended)
- **Storage**: 10GB+ free space
- **Python**: 3.8+ (3.10+ recommended)
- **OS**: Windows 10/11, macOS, or Linux

### Recommended Requirements
- **RAM**: 32GB+
- **GPU**: NVIDIA GPU with 24GB+ VRAM
- **Storage**: 20GB+ free space
- **Network**: Stable internet connection for model downloads

## 🛠️ Installation Options

### Option 1: Quick Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio

# Install all dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Option 2: Using Virtual Environment

```bash
# Create virtual environment
python -m venv higgs_audio_env

# Activate environment
# Windows:
higgs_audio_env\Scripts\activate
# macOS/Linux:
source higgs_audio_env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Option 3: Using Conda

```bash
# Create conda environment
conda create -n higgs_audio python=3.10
conda activate higgs_audio

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

## 📦 Package Dependencies

### Core AI/ML Libraries
- `torch` - PyTorch for deep learning
- `transformers>=4.45.1,<4.47.0` - HuggingFace transformers
- `accelerate>=0.26.0` - Accelerated training
- `descript-audio-codec` - Audio codec processing

### Audio Processing
- `librosa` - Audio analysis and processing
- `torchvision` - Computer vision utilities
- `torchaudio` - Audio processing for PyTorch
- `pydub` - Audio file manipulation
- `soundfile` - Audio file I/O

### Data Processing
- `pandas` - Data manipulation
- `pydantic` - Data validation
- `dacite` - Data class creation
- `json_repair` - JSON repair utilities
- `vector_quantize_pytorch` - Vector quantization

### Web Interface (Gradio)
- `gradio>=4.0.0` - Web interface framework

### System Monitoring
- `psutil` - System and process utilities

### Development Tools
- `loguru` - Advanced logging
- `ruff==0.12.2` - Fast Python linter
- `omegaconf` - Configuration management
- `click` - Command line interface

### Language Processing
- `langid` - Language identification
- `jieba` - Chinese text segmentation

### Additional Dependencies
- `tqdm` - Progress bars
- `pyyaml` - YAML parser
- `boto3==1.35.36` - AWS SDK
- `s3fs` - S3 file system

## 🎯 Usage Options

### 1. Command Line Interface

```bash
# Basic text-to-speech
python examples/generation.py --transcript "Hello world" --out_path output.wav

# Voice cloning
python examples/generation.py --transcript "Hello world" --ref_audio belinda --out_path cloned.wav

# Multi-speaker conversation
python examples/generation.py --transcript examples/transcript/multi_speaker/en_argument.txt --out_path conversation.wav
```

### 2. Gradio Web Interface

![Higgs Audio V2 Gradio Interface](docs/gradio_ui_screenshot.png)

The web interface provides an intuitive way to use Higgs Audio V2 without command line knowledge.

#### Full Interface (16GB+ RAM)
```bash
python examples/gradio_app.py
```

#### Light Interface (8GB+ RAM)
```bash
python examples/gradio_app_light.py
```

#### Demo Interface (No model loading)
```bash
python examples/gradio_demo.py
```

Then open your browser to: `http://localhost:7860`

### 3. Python API

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message

# Initialize the model
serve_engine = HiggsAudioServeEngine(
    "bosonai/higgs-audio-v2-generation-3B-base",
    "bosonai/higgs-audio-v2-tokenizer",
    device="cuda"  # or "cpu"
)

# Generate audio
messages = [Message(role="user", content="Hello, world!")]
output = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=0.3
)
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Memory Errors
```
❌ Error: The paging file is too small for this operation to complete.
```
**Solutions:**
- Close other applications to free up RAM
- Use the light interface: `python examples/gradio_app_light.py`
- Use the demo interface: `python examples/gradio_demo.py`
- Restart your computer
- Use a machine with more RAM

#### 2. CUDA/GPU Issues
```
❌ Error: CUDA out of memory
```
**Solutions:**
- Use CPU mode: `device="cpu"`
- Reduce batch size or sequence length
- Use a GPU with more VRAM
- Close other GPU-intensive applications

#### 3. Missing Dependencies
```
❌ Error: No module named 'gradio'
```
**Solutions:**
```bash
pip install gradio
pip install -r requirements.txt
```

#### 4. Model Download Issues
```
❌ Error: Failed to download model
```
**Solutions:**
- Check internet connection
- Ensure sufficient disk space (10GB+)
- Try downloading manually from HuggingFace
- Use a VPN if needed

### Performance Optimization

#### For CPU Users
- Use the demo interface for testing
- Close unnecessary applications
- Consider upgrading RAM

#### For GPU Users
- Ensure CUDA is properly installed
- Use GPU with 24GB+ VRAM for optimal performance
- Monitor GPU memory usage

## 📊 System Information

### Check Your System
```python
import torch
import psutil

# Check GPU
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Check RAM
memory = psutil.virtual_memory()
print(f"Total RAM: {memory.total / (1024**3):.1f} GB")
print(f"Available RAM: {memory.available / (1024**3):.1f} GB")
```

## 🎵 Available Voices

The system includes various voice options:
- `belinda` - Female voice
- `broom_salesman` - Male voice
- `chadwick` - Male voice
- `en_man` - English male voice
- `en_woman` - English female voice
- And many more in `examples/voice_prompts/`

## 📚 Additional Resources

- **Documentation**: `examples/README_gradio.md`
- **Examples**: `examples/` directory
- **Voice Prompts**: `examples/voice_prompts/`
- **Transcripts**: `examples/transcript/`

## 🤝 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Try the demo interface first
3. Check system requirements
4. Open an issue on GitHub with system details

## 📄 License

This project is licensed under the Apache 2.0 License. 