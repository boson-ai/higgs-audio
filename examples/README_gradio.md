# 🎵 Higgs Audio V2 - Gradio Web Interface

A user-friendly web interface for Higgs Audio V2 that makes it easy to generate speech from text without using the command line.

![Higgs Audio V2 Gradio Interface](docs/gradio_ui_screenshot.png)

*The Gradio interface provides an intuitive web-based UI for text-to-speech generation*

## 🚀 Quick Start

### 1. Install Gradio
```bash
pip install gradio
```

### 2. Run the Web Interface
```bash
python examples/gradio_app.py
```

### 3. Open Your Browser
Navigate to `http://localhost:7860` to access the interface.

## 🎯 Features

- **📝 Text Input**: Enter any text to convert to speech
- **🎭 Voice Selection**: Choose from available voices or use automatic selection
- **🌡️ Temperature Control**: Adjust generation randomness
- **🎨 Scene Description**: Customize the audio context
- **📱 Responsive Design**: Works on desktop and mobile
- **⚡ Fast Generation**: Optimized for both CPU and GPU

## 🎮 How to Use

1. **Enter Text**: Type or paste the text you want to convert to speech
2. **Choose Voice**: Select a specific voice or use "auto" for automatic selection
3. **Adjust Settings**: Modify temperature and scene description if needed
4. **Generate**: Click the "Generate Audio" button
5. **Download**: Play the generated audio and download if desired

## 🎵 Available Voices

The interface automatically detects available voices from the `examples/voice_prompts/` directory:

- `belinda` - Female voice
- `broom_salesman` - Male voice
- `chadwick` - Male voice
- `en_man` - English male voice
- `en_woman` - English female voice
- And many more...

## ⚙️ Configuration

### Port and Network Settings
The interface runs on port 7860 by default. You can modify these settings in the `main()` function:

```python
app.launch(
    server_name="0.0.0.0",  # Allow external connections
    server_port=7860,       # Change port here
    share=False,            # Set to True for public link
    show_error=True,        # Show detailed errors
    quiet=False             # Show console output
)
```

### GPU vs CPU
- **GPU**: Automatically detected and used for faster generation
- **CPU**: Fallback option (slower but functional)

## 🔧 Troubleshooting

### Common Issues

1. **"No module named 'gradio'"**
   ```bash
   pip install gradio
   ```

2. **Port already in use**
   - Change the port in the code or kill the process using port 7860

3. **Model loading errors**
   - Ensure you have enough disk space for model downloads
   - Check your internet connection
   - Verify all dependencies are installed

4. **Memory issues**
   - Close other applications to free up RAM
   - Use CPU mode if GPU memory is insufficient

### Performance Tips

- **GPU Recommended**: Use a GPU with 24GB+ memory for optimal performance
- **Batch Processing**: Generate multiple audio files in sequence
- **Text Length**: Keep text under 500 words for faster generation

## 🌐 Advanced Usage

### Custom Voice Integration
Add your own voice samples to `examples/voice_prompts/` and they'll automatically appear in the interface.

### API Integration
The interface can be extended to support API calls for integration with other applications.

### Multi-language Support
The interface supports multiple languages - just enter text in your preferred language.

## 📊 System Requirements

- **Python**: 3.8+
- **RAM**: 16GB+ (32GB+ recommended)
- **GPU**: Optional but recommended (24GB+ VRAM)
- **Storage**: 10GB+ free space for models
- **Network**: Internet connection for initial model download

## 🤝 Contributing

To improve the Gradio interface:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This interface is part of the Higgs Audio V2 project and follows the same Apache 2.0 license. 