#!/usr/bin/env python3
"""
Gradio Web Interface for Higgs Audio V2
A user-friendly web UI for local installation
"""

import gradio as gr
import torch
import torchaudio
import os
import tempfile
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
        
    def _get_available_voices(self) -> List[str]:
        """Get list of available voice prompts"""
        voice_dir = "examples/voice_prompts"
        voices = []
        if os.path.exists(voice_dir):
            for file in os.listdir(voice_dir):
                if file.endswith('.wav'):
                    voices.append(file.replace('.wav', ''))
        return sorted(voices)
    
    def _load_model(self):
        """Load the Higgs Audio model"""
        if self.serve_engine is None:
            print(f"Loading model on {self.device}...")
            self.serve_engine = HiggsAudioServeEngine(
                MODEL_PATH, 
                AUDIO_TOKENIZER_PATH, 
                device=self.device
            )
            print("Model loaded successfully!")
    
    def generate_audio(
        self,
        text: str,
        voice: str,
        temperature: float = 0.3,
        scene_description: str = "Audio is recorded from a quiet room."
    ) -> tuple:
        """
        Generate audio from text input
        
        Args:
            text: Input text to convert to speech
            voice: Voice to use (or 'auto' for automatic selection)
            temperature: Generation temperature (0.1 to 1.0)
            scene_description: Scene description for audio context
            
        Returns:
            tuple: (audio_file_path, info_text)
        """
        try:
            # Load model if not already loaded
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
📝 Text: {text}
🎵 Duration: {duration:.2f} seconds
🔊 Sample Rate: {output.sampling_rate} Hz
🎭 Voice: {voice}
🌡️ Temperature: {temperature}
            """.strip()
            
            return audio_path, info_text
            
        except Exception as e:
            error_msg = f"❌ Error generating audio: {str(e)}"
            return None, error_msg
    
    def create_interface(self):
        """Create the Gradio interface"""
        
        # Create the interface
        with gr.Blocks(
            title="Higgs Audio V2 - Text to Speech",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🎵 Higgs Audio V2 - Text to Speech
            
            Convert text to natural-sounding speech using the Higgs Audio V2 model.
            
            **Features:**
            - 🎭 Multiple voice options
            - 🌍 Multi-language support
            - 🎨 Customizable generation parameters
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
            
            # Examples section
            with gr.Accordion("📚 Examples", open=False):
                gr.Examples(
                    examples=[
                        ["Hello, this is a test of the Higgs Audio text-to-speech system. It can generate natural-sounding speech from any text input.", "auto", 0.3],
                        ["The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once.", "auto", 0.3],
                        ["Welcome to the future of speech synthesis. This AI model can create expressive and natural-sounding voices.", "auto", 0.5],
                    ],
                    inputs=[text_input, voice_dropdown, temperature_slider],
                    label="Try these examples"
                )
            
            # System info
            with gr.Accordion("ℹ️ System Information", open=False):
                device_info = f"🖥️ Device: {self.device}"
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name()
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                    device_info += f"\n🎮 GPU: {gpu_name}\n💾 GPU Memory: {gpu_memory:.1f} GB"
                else:
                    device_info += "\n⚠️ Running on CPU (slower performance)"
                
                gr.Markdown(f"""
                {device_info}
                
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
            
            # Auto-generate on text input (optional)
            text_input.change(
                fn=lambda x: x,
                inputs=[text_input],
                outputs=[text_input]
            )
        
        return interface

def main():
    """Main function to run the Gradio interface"""
    print("🚀 Starting Higgs Audio V2 Gradio Interface...")
    
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