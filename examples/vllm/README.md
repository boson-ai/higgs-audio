# Serve Higgs Audio with vLLM

We provided both OpenAI compatible chat completion and audio speech server backed by vLLM engine. To start the server, you can use the following command

```bash
docker run --gpus all --ipc=host --shm-size=20gb --network=host \
bosonai/higgs-audio-vllm:latest \
--served-model-name "higgs-audio-v2-generation-3B-base" \
--model "bosonai/higgs-audio-v2-generation-3B-base"  \
--audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
--limit-mm-per-prompt audio=50 \
--max-model-len 8192 \
--port 8000 \
--gpu-memory-utilization 0.8 \
--disable-mm-preprocessor-cache
```

In audio speech API, we provided the same voices as the [voice_prompts](../voice_prompts) folder. In addition, if you want to use your custom voices, you can add the voice presets in the docker run command 

```bash
--voice-presets-dir YOUR_VOICE_PRESETS_PATH
```

And in the voice presets directory, you need to add `config.json` file for each voice in the following format:
```json
{
    "belinda": {
        "transcript": "Twas the night before my birthday. Hooray! It's almost here! It may not be a holiday, but it's the best day of the year.",
        "audio_file": "belinda.wav"
    },
    "broom_salesman": {
        "transcript": "I would imagine so. A wand with a dragon heartstring core is capable of dazzling magic. And the bond between you and your wand should only grow stronger. Do not be surprised at your new wand's ability to perceive your intentions - particularly in a moment of need.",
        "audio_file": "broom_salesman.wav"
    }
}
```

We tested on A100 GPU with 40GB memory, which can achieve about 1500 tokens/s throughput for audio generation, which translate to 60 seconds audio generation per second with higgs-audio-tokenizer.
We also tested on RTX 4090 GPU with 24GB memory, which can achieve about 600 tokens/s throughput for audio generation, which translate to 24 seconds audio generation per second.

### cURL Example
To quickly test the server with curl, you can use the following command to generate audio with the audio speech API.

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v2-generation-3B-base",
    "voice": "en_woman",
    "input": "Today is a wonderful day to build something people love!",
    "response_format": "pcm"
  }' \
  --output - | ffmpeg -f s16le -ar 24000 -ac 1 -i - speech.wav
```


### Python example
You can also use the python client code to achieve more complex use cases with the chat completion API.

Voice clone
```bash
python run_chat_completion.py --api-base http://localhost:8000/v1 --task voice_clone
```

Smart voice
```bash
python run_chat_completion.py --api-base http://localhost:8000/v1 --task smart_voice
```

Multispeaker
```bash
python run_chat_completion.py --api-base http://localhost:8000/v1 --task multispeaker
```
