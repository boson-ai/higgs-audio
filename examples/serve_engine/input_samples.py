import base64
import os

from boson_multimodal.data_types import AudioContent, ChatMLSample, Message


def encode_base64_content_from_file(file_path: str) -> str:
    """Encode a content from a local file to base64 format."""
    # Read the audio file as binary and encode it directly to Base64
    with open(file_path, "rb") as audio_file:
        audio_base64 = base64.b64encode(audio_file.read()).decode("utf-8")
    return audio_base64


def get_interleaved_dialogue_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n"
        "<|scene_desc_start|>\n"
        "SPEAKER0: vocal fry;moderate pitch;monotone;masculine;young adult;slightly fast\n"
        "SPEAKER1: masculine;moderate;moderate pitch;monotone;mature\n\n"
        "In this scene, a group of adventurers is debating whether to investigate a potentially dangerous situation.\n"
        "<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="<|generation_instruction_start|>\nGenerate interleaved transcript and audio that lasts for around 20 seconds.\n<|generation_instruction_end|>",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


def get_zero_shot_input_sample():
    system_prompt = (
        "Generate audio following instruction.\n\n<|scene_desc_start|>\nSPEAKER0: british accent\n<|scene_desc_end|>"
    )

    messages = [
        Message(
            role="system",
            content=system_prompt,
        ),
        Message(
            role="user",
            content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
        ),
    ]
    chat_ml_sample = ChatMLSample(messages=messages)
    return chat_ml_sample


def get_voice_clone_input_sample():
    reference_text = "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic."
    reference_audio = encode_base64_content_from_file(
        os.path.join(os.path.dirname(__file__), "voice_examples/old_man.wav")
    )
    messages = [
        Message(
            role="user",
            content=reference_text,
        ),
        Message(
            role="assistant",
            content=AudioContent(raw_audio=reference_audio, audio_url="placeholder"),
        ),
        Message(
            role="user",
            content="Hey, everyone! Welcome back to Tech Talk Tuesdays.\n"
            "It's your host, Alex, and today, we're diving into a topic that's become absolutely crucial in the tech world — deep learning.\n"
            "And let's be honest, if you've been even remotely connected to tech, AI, or machine learning lately, you know that deep learning is everywhere.",
        ),
    ]
    return ChatMLSample(messages=messages)


INPUT_SAMPLES = {
    "interleaved_dialogue": get_interleaved_dialogue_input_sample,
    "zero_shot": get_zero_shot_input_sample,
    "voice_clone": get_voice_clone_input_sample,
}
