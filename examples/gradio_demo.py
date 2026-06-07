#!/usr/bin/env python3
"""
Demo Gradio Interface for Higgs Audio V2
Shows the UI without loading the heavy model
"""

import gradio as gr
import torch
import os
import tempfile
import psutil
from typing import List

class HiggsAudioDemoInterface:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
    def _check_memory(self) -> dict:
        """Check available system memory"""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'percent_used': memory.percent
        }
    
    def demo_generate_audio(
        self,
        text: str,
        voice: str,
        temperature: float = 0.3,
        scene_description: str = "Audio is recorded from a quiet room."
    ) -> tuple:
        """
        Demo function that shows the interface without loading the model
        """
        try:
            # Check if text is provided
            if not text.strip():
                return None, "❌ Please enter some text to convert to speech."
            
            # Check memory
            memory_info = self._check_memory()
            
            # Simulate processing time
            import time
            time.sleep(2)  # Simulate processing
            
            # Create demo info
            info_text = f"""
🎵 Demo Mode - Higgs Audio V2 Interface

📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}
🎭 Voice: {voice}
🌡️ Temperature: {temperature}
🎨 Scene: {scene_description}

💾 Memory Status:
• Total RAM: {memory_info['total_gb']:.1f} GB
• Available RAM: {memory_info['available_gb']:.1f} GB
• Memory Usage: {memory_info['percent_used']:.1f}%

⚠️ This is a demo interface. To use the full model:
1. Ensure you have at least 16GB RAM
2. Close other applications
3. Use the full interface: python examples/gradio_app.py

🖥️ Device: {self.device}
            """.strip()
            
            # Return a placeholder audio file or None
            return None, info_text
            
        except Exception as e:
            error_msg = f"❌ Demo error: {str(e)}"
            return None, error_msg
    
    def create_interface(self):
        """Create the demo Gradio interface"""
        
        # Create the interface
        with gr.Blocks(
            title="Higgs Audio V2 - Demo Interface",
            theme=gr.themes.Soft(),
            css="""
            .gradio-container {
                max-width: 1200px !important;
                margin: auto !important;
            }
            """
        ) as interface:
            
            gr.Markdown("""
            # 🎵 Higgs Audio V2 - Demo Interface
            
            **⚠️ This is a DEMO version** - Shows the interface without loading the heavy model.
            
            Convert text to natural-sounding speech using the Higgs Audio V2 model.
            
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
                        "🎤 Generate Audio (Demo)",
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
                        lines=12,
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
                
                **⚠️ Demo Mode:** This interface shows the UI without loading the heavy model.
                """)
            
            # Event handlers
            generate_btn.click(
                fn=self.demo_generate_audio,
                inputs=[text_input, voice_dropdown, temperature_slider, scene_input],
                outputs=[audio_output, info_output]
            )
        
        return interface

def main():
    """Main function to run the demo Gradio interface"""
    print("🚀 Starting Higgs Audio V2 Demo Interface...")
    
    # Create interface
    interface = HiggsAudioDemoInterface()
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