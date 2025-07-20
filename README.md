<h1 align="center">HiggsAudio-V2: Redefining Expressiveness in Audio Generation</h1>

We are open-sourcing HiggsAudio-V2, a powerful audio foundation model pretrained on over 10 million hours of audio data and a diverse set of text data. Despite having no post-training or fine-tuning, HiggsAudio-V2 excels in expressive audio generation, thanks to its deep language and acoustic understanding.

On [EmergentTTS-Eval](https://github.com/boson-ai/emergenttts-eval-public), it achieves win rates of **75.7%** and **55.7%** over "gpt-4o-mini-tts" on the "Emotions" and "Questions" categories, respectively. It also obtains state-of-the-art performance on traditional TTS benchmarks like Seed-TTS Eval and Emotional Speech Dataset (ESD). Moreover, the model demonstrates capabilities rarely seen in previous systems, including generating natural multi-speaker dialogues in multiple languages, automatic prosody adaptation during narration, melodic humming with the cloned voice, and simultaneous generation of speech and background music.

<p align="center">
    <img src="figures/emergent-tts-emotions-win-rate.png" width=900>
</p>

Here's the demo video that shows some of the emergent capabilities:

[![Watch the demo video](./demo_video/thumbnail_demo_video.png)](https://www.dropbox.com/scl/fi/56dxm38o9s5ajw7yf59rz/open_source_repo_demo.mov?rlkey=19o9zr91c7hmjixq0fdjylhtr&st=azknsm78&dl=1)



## Installation

### Option 1: Using venv

```bash
python3 -m venv higgs_audio_env
source higgs_audio_env/bin/activate
pip install -r requirements.txt
pip install -e .
```


### Option 2: Using conda
```bash
conda create -n higgs-audio python=3.10
conda activate higgs-audio
pip install -r requirements.txt
pip install -e .
```

### Option 3: Using uv
```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install -e .
```

## Usage

### Get Started

Here's a basic python snippet to help you get started.

```python
from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
from boson_multimodal.data_types import ChatMLSample, Message, AudioContent

import torch
import torchaudio
import time
import click

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-staging"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer-staging"

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
        content="The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    ),
]
device = "cuda" if torch.cuda.is_available() else "cpu"

serve_engine = HiggsAudioServeEngine(MODEL_PATH, AUDIO_TOKENIZER_PATH, device=device)

output: HiggsAudioResponse = serve_engine.generate(
    chat_ml_sample=ChatMLSample(messages=messages),
    max_new_tokens=1024,
    temperature=1.0,
    top_p=0.95,
    top_k=50,
    stop_strings=["<|end_of_text|>", "<|eot_id|>"],
    seed=12345,
)
torchaudio.save(f"output.wav", torch.from_numpy(output.audio)[None, :], output.sampling_rate)
```

We also provide a list of examples under [examples](./examples). In the following we highlight a few examples to help you use HiggsAudio-V2.

### Zero-Shot Voice Cloning
Generate audio with specific voice characteristics (e.g., accents).

```bash
python3 examples/generation.py \
--transcript examples/transcript/single_speaker/en_basic.txt \
--ref_audio belinda \
--out_path generation.wav \
--seed 12345
```

### Single-speaker Generation with Smart Voice
If you do not specify reference voice, the model will decide the voice based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/single_speaker/en_basic.txt \
--out_path generation.wav \
--seed 12345
```


### Multi-speaker Dialog with Smart Voice
Generate multi-speaker dialog. The model will decide the voices based on the transcript it sees.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--out_path generation.wav \
--seed 1234
```

### Multi-speaker Dialog with Voice Clone

Generate multi-speaker dialog with the voices you picked.

```bash
python3 examples/generation.py \
--transcript examples/transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--out_path generation.wav \
--seed 1234
```


## Technical Details
<img src="figures/higgs_audio_v2_architecture_combined.png" width=900>


HiggsAudio-V2 adopts the "generation variant" depicted in the architecture figure above. Its strong performance is driven by three key technical innovations:
- We developed an automated annotation pipeline that leverages multiple ASR models, sound event classification models, and our in-house audio understanding model. Using this pipeline, we cleaned and annotated 10 million hours audio data, which we refer to as **AudioVerse**. The in-house understanding model is finetuned on top of [Higgs Audio V1 Understanding](https://www.boson.ai/blog/higgs-audio), which adopts the "understanding variant" shown in the architecture figure.
- We trained a unified audio tokenizer from scratch that captures both semantic and acoustic features. Learn more in the [tokenizer blog](./tech_blogs/TOKENIZER_BLOG.md).
- We proposed the DualFFN architecture, which enhances the LLM’s ability to model acoustics tokens with minimal computational overhead. See the [architecture blog](./tech_blogs/ARCHITECTURE_BLOG.md).

## Evaluation

Here's the performance of HiggsAudio-V2 on four benchmarks,  [Seed-TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval), [Emotional Speech Dataset (ESD)](https://paperswithcode.com/dataset/esd), [EmergentTTS-Eval](https://arxiv.org/abs/2505.23009), and Multi-speaker Eval:

#### Seed-TTS Eval & ESD

We prompt HiggsAudio-V2 with `<ref_text, ref_audio, text>` for zero-shot TTS. We adopt the standard evaluation metric in Seed-TTS Eval and ESD.

|                              | SeedTTS-Eval| | ESD   |                 |
|------------------------------|--------|--------|---------|-------------------|
|                              | WER ↓ | SIM ↑ | WER ↓ | SIM (emo2vec) ↑ |
| Cosyvoice2                   | 2.28   | 65.49  | 2.71    | 80.48             |
| Qwen2.5-omni†                | 2.33   | 64.10  | -       | -                 |
| ElevenLabs Multilingual V2   | **1.43**   | 50.00  | 1.66    | 65.87             |
| HiggsAudio V1                | 2.18   | 66.27  | **1.49**    | 82.84             |
| HiggsAudio V2 (base)         | 2.44   | **67.70**  | 1.78    | **86.13**         |


#### EmergentTTS-Eval ("Emotions" and "Questions")

Following the [EmergentTTS-Eval Paper](https://arxiv.org/abs/2505.23009), we report the win-rate over "gpt-4o-mini-tts" with the "alloy" voice.

| Model                              | Emotions (%) ↑ | Questions (%) ↑ |
|------------------------------------|--------------|----------------|
| HiggsAudio-V2 (base)               | **75.71%**   | **55.71%**         |
| [gpt-4o-audio-preview†](https://platform.openai.com/docs/models/gpt-4o-audio-preview)       | 61.64%       | 47.85%         |
| [Hume.AI](https://www.hume.ai/research)                            | 61.60%       | 43.21%         |
| **BASELINE:** [gpt-4o-mini-tts](https://platform.openai.com/docs/models/gpt-4o-mini-tts)  | 50.00%       | 50.00%         |
| [Qwen 2.5 Omni†](https://github.com/QwenLM/Qwen2.5-Omni)      | 41.60%       | 51.78%         |
| [minimax/speech-02-hd](https://replicate.com/minimax/speech-02-hd)               | 40.86%        | 47.32%         |
| [ElevenLabs Multilingual v2](https://elevenlabs.io/blog/eleven-multilingual-v2)         | 30.35%       | 39.46%         |
| [DeepGram Aura-2](https://deepgram.com/learn/introducing-aura-2-enterprise-text-to-speech)                    | 29.28%       | 48.21%         |
| [Sesame csm-1B](https://github.com/SesameAILabs/csm)                      | 15.96%       | 31.78%         |

<sup><sub>'†' means using the strong-prompting method described in the paper.</sub></sup>


#### Multi-speaker Eval

We also designed a multi-speaker evaluation benchmark to evaluate the capability of HiggsAudio-V2 for multi-speaker dialog generation. The benchmark contains three subsets

- `two-speaker-conversation`: 1000 synthetic dialogues involving two speakers. It contains two reference audio clips to evaluate the model’s ability in double voice cloning.
- `small talk`: 250 synthetic dialogues characterized by short utterances and a limited number of turns (4–6). It also contains two reference audio clips to test double voice cloning, though the dialogues are shorter and simpler than those in two-speaker-conversation.
- `small talk (no ref)`: 250 synthetic dialogues, also with short utterances and 4–6 turns. Unlike the other subsets, it does not include reference audio and is designed to evaluate the model’s ability to automatically assign appropriate voices to speakers.


We evaluate the word-error-rate (WER) and the geometric mean between intra-speaker similarity and inter-speaker dis-similarity on these three subsets. Other than HiggsAudio-V2, we also evaluated [MoonCast](https://github.com/jzq2000/MoonCast) and [nari-labs/dia](https://github.com/nari-labs/dia). Results are summarized in the following table. We are not able to run [nari-labs/dia](https://github.com/nari-labs/dia) on our "two-speaker-conversation" subset due to its strict limitation on the length of the utterances.

|                                                | two-speaker-conversation |                |small talk |                | small talk (no ref) |                |
| ---------------------------------------------- | -------------- | ------------------ | ---------- | -------------- | ------------------- | -------------- |
|                                                | WER ↓                      | Mean Sim & Dis-sim ↑ | WER ↓       |  Mean Sim & Dis-sim ↑ | WER ↓               | Mean Sim & Dis-sim ↑ |
| [MoonCast](https://github.com/jzq2000/MoonCast) | 38.77                    | 46.02         | **8.33**       | 63.68          | 24.65               | 53.94 |
| [nari-labs/dia](https://github.com/nari-labs/dia)         | \-                       | \-             | 17.62      | 63.15          | 19.46               | **61.14**          |
| HiggsAudio-V2 (base)     | **18.88**                    | **51.95**          | 11.89      | **67.92**              | **14.65**               | 55.28              |



## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). Please see the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for complete attribution and licensing information.
