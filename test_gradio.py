#!/usr/bin/env python3
"""
Simple test to verify Gradio interface works
"""

import gradio as gr

def test_gradio():
    """Test if Gradio works"""
    print("Testing Gradio interface...")
    
    # Create a simple interface
    def greet(name):
        return f"Hello {name}!"
    
    # Create interface
    iface = gr.Interface(
        fn=greet,
        inputs=gr.Textbox(label="Name"),
        outputs=gr.Textbox(label="Greeting")
    )
    
    print("✅ Gradio interface created successfully!")
    print("🎉 Gradio is working!")
    
    return iface

if __name__ == "__main__":
    test_gradio() 