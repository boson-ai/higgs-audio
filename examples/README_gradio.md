# Higgs Audio Gradio UI

A web-based interface for text-to-speech generation using Higgs Audio v2. This interface provides an easy-to-use web UI for generating high-quality speech from text.

## Features

- **Text-to-Speech Generation**: Convert text to natural-sounding speech
- **Voice Selection**: Choose from available voice prompts
- **Parameter Tuning**: Adjust temperature, top-p, and seed for different outputs
- **Multiple Modes**: Full, light, and demo modes for different use cases
- **Real-time Preview**: Listen to generated audio immediately
- **Error Handling**: Comprehensive error messages and user guidance

## Installation

### Prerequisites

1. **Install Core Dependencies**:
   ```bash
   pip install -r requirements.txt
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

### Troubleshooting

- **CUDA Issues**: If you encounter CUDA-related errors, try installing the CPU-only version:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  ```

- **Memory Issues**: Use the light mode for systems with limited RAM
- **Port Conflicts**: Change the port using `--port` argument if 7860 is occupied

## Usage

### Quick Start

1. **Full Mode** (recommended):
   ```bash
   python examples/gradio_app.py --mode full
   ```

2. **Light Mode** (for limited resources):
   ```bash
   python examples/gradio_app.py --mode light
   ```

3. **Demo Mode** (for quick testing):
   ```bash
   python examples/gradio_app.py --mode demo
   ```

### Command Line Options

- `--mode`: Interface mode (`full`, `light`, `demo`)
- `--port`: Server port (default: 7860)
- `--host`: Server host (default: 127.0.0.1)

### Interface Modes

#### Full Mode
- All features enabled
- Maximum parameter ranges
- Best for production use

#### Light Mode
- Reduced parameter ranges
- Optimized for limited resources
- Suitable for testing

#### Demo Mode
- Limited text length (200 characters)
- Restricted parameter ranges
- Auto-generation on text change
- Best for quick demonstrations

## Voice Prompts

The interface automatically detects voice prompts from the `examples/voice_prompts/` directory. To add your own voices:

1. Place `.wav` files in `examples/voice_prompts/`
2. Restart the Gradio app
3. Your voices will appear in the dropdown menu

## Parameters

- **Temperature** (0.1-1.0): Controls randomness in generation
  - Lower values (0.1-0.3): More deterministic, consistent output
  - Higher values (0.7-1.0): More creative, varied output

- **Top-p** (0.1-1.0): Controls diversity in token selection
  - Lower values: More focused, conservative choices
  - Higher values: More diverse, creative choices

- **Seed**: Random seed for reproducible results
  - Same seed + same text = same output
  - Useful for consistent results

## Tips for Best Results

1. **Text Quality**: Use clear, well-formatted text
2. **Voice Selection**: Try different voices for variety
3. **Parameter Tuning**: Start with default values, adjust based on needs
4. **Seed Management**: Use specific seeds for reproducible results
5. **Error Handling**: Check the status box for error messages

## Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - Ensure you have internet connection for model download
   - Check if the model path is correct

2. **CUDA out of memory**:
   - Use light mode
   - Reduce batch size
   - Close other GPU applications

3. **Slow generation**:
   - Use CPU mode if GPU is not available
   - Check system resources
   - Try shorter text inputs

4. **Audio quality issues**:
   - Adjust temperature and top-p parameters
   - Try different voices
   - Ensure input text is clear and well-formatted

### Performance Optimization

- **GPU Usage**: Ensure CUDA is properly installed for faster generation
- **Memory**: Close unnecessary applications to free up RAM
- **Network**: Stable internet connection for model downloads

## Contributing

To contribute to the Gradio interface:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](../LICENSE) file for details. 