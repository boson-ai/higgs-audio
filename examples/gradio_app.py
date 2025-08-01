#!/usr/bin/env python3
"""
Higgs Audio Gradio UI
A web-based interface for text-to-speech generation using Higgs Audio v2.

Usage:
    python gradio_app.py --mode full    # Full interface with all features
    python gradio_app.py --mode light   # Lightweight interface
    python gradio_app.py --mode demo    # Demo mode with limited features
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import torch
from boson_multimodal import BosonMultimodal

# Add the parent directory to the path to import the generation module
sys.path.append(str(Path(__file__).parent.parent))
from examples.generation import generate_audio

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
VOICES_DIR = SCRIPT_DIR / "voice_prompts"
MODEL_NAME = "boson-ai/higgs-audio-v2-base"

def get_available_voices() -> List[str]:
    """Get list of available voice prompts."""
    voices = []
    if VOICES_DIR.exists():
        for p in VOICES_DIR.glob("*.wav"):
            voices.append(p.stem)
    voices.sort()
    return voices

def get_voice_path(voice_name: str) -> Optional[str]:
    """Get the full path to a voice file."""
    voice_file = VOICES_DIR / f"{voice_name}.wav"
    return str(voice_file) if voice_file.exists() else None

def generate_speech(
    text: str,
    voice: str,
    temperature: float,
    top_p: float,
    seed: int,
    mode: str = "full"
) -> Tuple[str, str]:
    """
    Generate speech from text using Higgs Audio.
    
    Args:
        text: Input text to convert to speech
        voice: Voice to use for generation
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        seed: Random seed for reproducibility
        mode: Interface mode (full, light, demo)
    
    Returns:
        Tuple of (audio_path, error_message)
    """
    try:
        if not text.strip():
            return None, "Please enter some text to generate speech."
        
        # Validate parameters based on mode
        if mode == "demo":
            if len(text) > 200:
                return None, "Demo mode: Text must be under 200 characters."
            temperature = min(temperature, 0.7)
            top_p = min(top_p, 0.9)
        
        # Get voice path
        voice_path = get_voice_path(voice) if voice else None
        
        # Create temporary file for output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            output_path = tmp_file.name
        
        # Generate audio
        generate_audio(
            transcript=text,
            out_path=output_path,
            ref_audio=voice_path,
            temperature=temperature,
            top_p=top_p,
            seed=seed if seed > 0 else None
        )
        
        return output_path, ""
        
    except Exception as e:
        return None, f"Error generating speech: {str(e)}"

def create_interface(mode: str = "full") -> gr.Interface:
    """Create the Gradio interface based on mode."""
    
    # Get available voices
    voices = get_available_voices()
    default_voice = voices[0] if voices else "smart_voice"
    
    # Interface title and description
    title = f"Higgs Audio v2 - {mode.title()} Mode"
    description = f"""
    Generate high-quality speech from text using Higgs Audio v2.
    
    **{mode.title()} Mode Features:**
    - Text-to-speech generation
    - Voice selection
    - Parameter tuning
    - Real-time audio preview
    """
    
    if mode == "demo":
        description += "\n**Demo Mode Limitations:**\n- Text limited to 200 characters\n- Reduced parameter ranges"
    
    # Create interface
    with gr.Blocks(title=title, theme=gr.themes.Soft()) as interface:
        gr.Markdown(f"# {title}")
        gr.Markdown(description)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                text_input = gr.Textbox(
                    label="Text to Speech",
                    placeholder="Enter the text you want to convert to speech...",
                    lines=4,
                    max_lines=8 if mode == "full" else 4
                )
                
                with gr.Row():
                    voice_dropdown = gr.Dropdown(
                        choices=voices,
                        value=default_voice,
                        label="Voice",
                        info="Select a voice for generation"
                    )
                    
                    seed_input = gr.Number(
                        value=42,
                        label="Seed",
                        info="Random seed for reproducible results"
                    )
                
                with gr.Row():
                    temp_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0 if mode == "full" else 0.7,
                        value=0.3,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in generation"
                    )
                    
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0 if mode == "full" else 0.9,
                        value=0.9,
                        step=0.1,
                        label="Top-p",
                        info="Controls diversity in generation"
                    )
                
                generate_btn = gr.Button("Generate Speech", variant="primary")
                
            with gr.Column(scale=1):
                # Output section
                audio_output = gr.Audio(
                    label="Generated Audio",
                    type="filepath"
                )
                
                error_output = gr.Textbox(
                    label="Status",
                    interactive=False
                )
                
                # Info section
                gr.Markdown("### Tips:")
                gr.Markdown("""
                - Use clear, well-formatted text for best results
                - Adjust temperature for more/less creative output
                - Try different voices for variety
                - Use seeds for reproducible results
                """)
        
        # Event handlers
        generate_btn.click(
            fn=lambda t, v, temp, tp, s: generate_speech(t, v, temp, tp, s, mode),
            inputs=[text_input, voice_dropdown, temp_slider, top_p_slider, seed_input],
            outputs=[audio_output, error_output]
        )
        
        # Auto-generate on text change (only in demo mode)
        if mode == "demo":
            text_input.change(
                fn=lambda t, v, temp, tp, s: generate_speech(t, v, temp, tp, s, mode) if t.strip() else (None, ""),
                inputs=[text_input, voice_dropdown, temp_slider, top_p_slider, seed_input],
                outputs=[audio_output, error_output]
            )
    
    return interface

def main():
    """Main function to run the Gradio app."""
    parser = argparse.ArgumentParser(description="Higgs Audio Gradio UI")
    parser.add_argument(
        "--mode",
        choices=["full", "light", "demo"],
        default="full",
        help="Interface mode: full (all features), light (basic), demo (limited)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to run the server on"
    )
    
    args = parser.parse_args()
    
    # Create and launch interface
    interface = create_interface(args.mode)
    
    print(f"Starting Higgs Audio Gradio UI in {args.mode} mode...")
    print(f"Server will be available at: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the server")
    
    interface.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        show_error=True
    )

if __name__ == "__main__":
    main() 