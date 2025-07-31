# Prediction interface for Cog ⚙️
# https://cog.run/python
import os
import subprocess
import time

import torch
import torchaudio
from cog import BasePredictor, Input, Path

from boson_multimodal.data_types import ChatMLSample, Message
from boson_multimodal.serve.serve_engine import HiggsAudioResponse, HiggsAudioServeEngine


MODEL_PATH = "higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "higgs-audio-v2-tokenizer"
MODEL_URL = "https://weights.replicate.delivery/default/bosonai/higgs-audio-v2-generation-3B-base/model.tar"
TOKENIZER_URL = "https://weights.replicate.delivery/default/bosonai/higgs-audio-v2-tokenizer/model.tar"


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-xf", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download weights
        if not os.path.exists(MODEL_PATH):
            download_weights(MODEL_URL, MODEL_PATH)
        if not os.path.exists(AUDIO_TOKENIZER_PATH):
            download_weights(TOKENIZER_URL, AUDIO_TOKENIZER_PATH)

        # Set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Initialize the serve engine
        self.serve_engine = HiggsAudioServeEngine(
            MODEL_PATH,
            AUDIO_TOKENIZER_PATH,
            device=self.device)
        print("Higgs Audio V2 model loaded successfully")

    def predict(
        self,
        text: str = Input(
            description="Text to convert to speech",
            default="The sun rises in the east and sets in the west",
        ),
        temperature: float = Input(
            description="Controls randomness in generation. Lower values are more deterministic.",
            ge=0.1,
            le=1.0,
            default=0.3,
        ),
        top_p: float = Input(
            description="Nucleus sampling parameter. Controls diversity of generated audio.",
            ge=0.1,
            le=1.0,
            default=0.95,
        ),
        top_k: int = Input(
            description="Top-k sampling parameter. Limits vocabulary to top k tokens.", ge=1, le=100, default=50
        ),
        max_new_tokens: int = Input(
            description="Maximum number of audio tokens to generate", ge=256, le=2048, default=1024
        ),
        scene_description: str = Input(
            description="Scene description for audio context", default="Audio is recorded from a quiet room."
        ),
        system_message: str = Input(description="Custom system message (optional)", default=""),
    ) -> Path:
        """Run a single prediction on the model"""
        try:
            # Construct system prompt
            if system_message:
                system_prompt = system_message
            else:
                system_prompt = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_description}\n<|scene_desc_end|>"

            # Prepare messages
            messages = [
                Message(
                    role="system",
                    content=system_prompt,
                ),
                Message(
                    role="user",
                    content=text,
                ),
            ]

            # Generate audio
            output: HiggsAudioResponse = self.serve_engine.generate(
                chat_ml_sample=ChatMLSample(messages=messages),
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
            )
            # Save output audio to a temporary file with a clear filename
            output_path = "/tmp/audio_output.wav"
            # Convert output audio to tensor and save
            audio_tensor = torch.from_numpy(output.audio)[None, :]
            torchaudio.save(output_path, audio_tensor, output.sampling_rate, format="wav")
            return Path(output_path)

        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {str(e)}")
