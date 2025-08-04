#!/usr/bin/env python3
"""
Lightweight Gradio Web Interface for Higgs Audio V2
Handles memory limitations gracefully
"""

import gradio as gr
import torch
import torchaudio
import os
import tempfile
import psutil
from typing import Optional, List

# Import Higgs Audio modules
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message

# Model configuration
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

class HiggsAudioGradioInterface:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.serve_engine = None
        self.available_voices = self._get_available_voices()
        self.model_loaded = False
        
    def _get_available_voices(self) -> List[str]:
        """Get list of available voice prompts"""
        voice_dir = "examples/voice_prompts"
        voices = []
        if os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('.wav'):
                    voices.append(file.replace('.wav', ''))
        return sorted(voices)
    
    def _check_memory(self) -> dict:
        """Check available system memory"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
    
    def _load_model(self):
        """Load the Higgs Audio model with memory checks"""
        if self.serve_engine is None:
            # Check memory before loading
            memory_info = self._check_memory()
            print(f"Memory check - Available: {memory_info['available_gb']:.1f}GB, Used: {memory_info['percent_used']:.1f}%")
            
            if memory_info['available_gb'] < 8:
                raise MemoryError(f"Insufficient memory. Available: {memory_info['available_gb']:.1f}GB, Need: 8GB+")
            
            print(f"Loading model on {self.device}...")
            try:
                self.serve_engine = HiggsAudioServeEngine(
                    MODEL_PATH, 
                    AUDIO_TOKENIZER_PATH, 
                    device=self.device
                )
                self.model_loaded = True
                print("Model loaded successfully!")
            except Exception as e:
                if "paging file" in str(e) or "memory" in str(e).lower():
                    raise MemoryError(f"Memory error: {str(e)}. Try closing other applications or use a machine with more RAM.")
                else:
                    raise e
    
    def generate_audio(
        self,
        text: str,
        voice: str,
        temperature: float = 0.3,
        scene_description: str = "Audio is recorded from a quiet room."
    ) -> tuple:
        """
        Generate audio from text input with memory management
        """
        try:
            # Check if text is provided
            if not text.strip():
                return None, "❌ Please enter some text to convert to speech."
            
            # Check memory before loading model
            memory_info = self._check_memory()
            
            # Load model if not already loaded
            if not self.model_loaded:
                self._load_model()
            
            # Create system prompt
            system_prompt = f"""Generate audio following instruction.

<|scene_desc_start|>
{scene_description}
<|scene_desc_end|>"""
            
            # Create messages
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=text)
            ]
            
            print(f"Generating audio for: {text[:50]}...")
            
            # Generate audio
            output: HiggsAudioResponse = self.serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=1024,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                torchaudio.save(tmp_file.name, torch.from_numpy(output.audio)[None, :], output.sampling_rate)
                audio_path = tmp_file.name
            
            # Create info text
            duration = len(output.audio) / output.sampling_rate
            info_text = f"""
✅ Audio generated successfully!
📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}
🎵 Duration: {duration:.2f} seconds
🔊 Sample Rate: {output.sampling_rate} Hz
🎭 Voice: {voice}
🌡️ Temperature: {temperature}
💾 Memory Used: {memory_info['percent_used']:.1f}%
            """.strip()
            
            return audio_path, info_text
            
        except MemoryError as e:
            error_msg = f"""
❌ Memory Error: {str(e)}

💡 Solutions:
• Close other applications to free up RAM
• Restart your computer
• Use a machine with at least 16GB RAM
• Try running on GPU if available
            """.strip()
            return None, error_msg
            
        except Exception as e:
            error_msg = f"""
❌ Error generating audio: {str(e)}

💡 Troubleshooting:
• Check your internet connection
• Ensure you have enough disk space
• Try restarting the application
            """.strip()
            return None, error_msg
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Create the interface
        with gr.Blocks(
            title="Higgs Audio V2 - Text to Speech (Light)",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🎵 Higgs Audio V2 - Text to Speech (Light Version)
            
            Convert text to natural-sounding speech using the Higgs Audio V2 model.
            
            **⚠️ Memory Optimized Version** - Handles memory limitations gracefully.
            
            **Features:**
            - 🎭 Multiple voice options
            - 🌍 Multi-language support
            - 🎨 Customizable generation parameters
            - 💾 Memory management
            - ⚡ Fast generation (with GPU)
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    # Input section
                    gr.Markdown("### 📝 Input")
                    
                    text_input = gr.Textbox(
                        label="Text to Convert",
                        placeholder="Enter the text you want to convert to speech...",
                        lines=4,
                        max_lines=10
                    )
                    
                    with gr.Row():
                        voice_dropdown = gr.Dropdown(
                            choices=["auto"] + self.available_voices,
                            value="auto",
                            label="Voice Selection",
                            info="Choose a specific voice or 'auto' for automatic selection"
                        )
                        
                        temperature_slider = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.3,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness in generation (lower = more consistent)"
                        )
                    
                    scene_input = gr.Textbox(
                        label="Scene Description (Optional)",
                        placeholder="Audio is recorded from a quiet room.",
                        value="Audio is recorded from a quiet room.",
                        lines=2
                    )
                    
                    generate_btn = gr.Button(
                        "🎤 Generate Audio",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=1):
                    # Output section
                    gr.Markdown("### 🎵 Output")
                    
                    audio_output = gr.Audio(
                        label="Generated Audio",
                        type="filepath"
                    )
                    
                    info_output = gr.Textbox(
                        label="Generation Info",
                        lines=8,
                        interactive=False
                    )
            
            # System info
            with gr.Accordion("ℹ️ System Information", open=False):
                memory_info = self._check_memory()
                device_info = f"🖥️ Device: {self.device}"
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    device_info += f"\n🎮 GPU: {gpu_name}\n💾 GPU Memory: {gpu_memory:.1f} GB"
                else:
                    device_info += "\n⚠️ Running on CPU (slower performance)"
                
                gr.Markdown(f"""
                {device_info}
                
                **Memory Status:**
                • Total RAM: {memory_info['total_gb']:.1f} GB
                • Available RAM: {memory_info['available_gb']:.1f} GB
                • Memory Usage: {memory_info['percent_used']:.1f}%
                
                **Available Voices:** {', '.join(self.available_voices[:5])}{'...' if len(self.available_voices) > 5 else ''}
                
                **Model:** Higgs Audio V2 (3.6B parameters)
                **Tokenizer:** bosonai/higgs-audio-v2-tokenizer
                """)
            
            # Event handlers
            generate_btn.click(
                fn=self.generate_audio,
                inputs=[text_input, voice_dropdown, temperature_slider, scene_input],
                outputs=[audio_output, info_output]
            )
        
        return interface

def main():
    """Main function to run the Gradio interface"""
    print("🚀 Starting Higgs Audio V2 Gradio Interface (Light)...")
    
    # Create interface
    interface = HiggsAudioGradioInterface()
    app = interface.create_interface()
    
    # Launch the app
    app.launch(
        server_name="127.0.0.1",  # Use localhost
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True to create public link
        show_error=True,        # Show detailed errors
        quiet=False             # Show console output
    )

if __name__ == "__main__":
    main() 