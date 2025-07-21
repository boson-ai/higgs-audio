"""Example for using HiggsAudio for generating both the transcript and audio in an interleaved manner."""

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
import torch
import torchaudio
import time
from loguru import logger
import click

from input_samples import INPUT_SAMPLES

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"


@click.command()
@click.argument("example", type=click.Choice(list(INPUT_SAMPLES.keys())))
def main(example: str):
    input_sample = INPUT_SAMPLES[example]()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    serve_engine = HiggsAudioServeEngine(
        MODEL_PATH,
        AUDIO_TOKENIZER_PATH,
        device=device,
    )

    logger.info("Starting generation...")
    start_time = time.time()
    output: HiggsAudioResponse = serve_engine.generate(
        chat_ml_sample=input_sample,
        max_new_tokens=1024,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Generation time: {elapsed_time:.2f} seconds")

    torchaudio.save(f"output_{example}.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
    logger.info(f"Generated text:\n{output.generated_text}")
    logger.info(f"Saved audio to output_{example}.wav")


if __name__ == "__main__":
    main()
