<div align="center">

# 🎉 Higgs Audio v3 is here — you no longer need this repo!

### **👉 Don't clone this repository to use the latest model.**

**Higgs Audio v3** is a standalone release and does **not** depend on the code here.
Just grab the weights or call the hosted API:

### 🤗 **[Hugging Face — bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)**
### 📖 **[Boson AI API — docs.boson.ai/models/higgs-audio-tts](https://docs.boson.ai/models/higgs-audio-tts/overview)**

_Conversational TTS across 100+ languages · zero-shot voice cloning · inline emotion / style / prosody control._

</div>

---

## Use Higgs Audio v3

### Option 1 — Boson AI API (no setup, no GPU)

Free, rate-limited public preview. Get a key at [boson.ai/workspace](https://boson.ai/workspace).

```bash
export BOSON_API_KEY=bai-xxxx

curl https://api.boson.ai/v1/audio/speech \
  -H "Authorization: Bearer $BOSON_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "higgs-audio-v3-tts", "input": "Hello, this is a test."}' \
  --output out.mp3
```

OpenAI-compatible; supports preset voices, zero-shot cloning, and streaming. Full reference: **[API docs](https://docs.boson.ai/models/higgs-audio-tts/overview)**.

### Option 2 — Self-host the open weights

Weights: **[bosonai/higgs-audio-v3-tts-4b](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b)**. We recommend serving with **[SGLang-Omni](https://github.com/sgl-project/sglang-omni)**:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxxxxxx
hf download bosonai/higgs-audio-v3-tts-4b

sgl-omni serve --model-path bosonai/higgs-audio-v3-tts-4b --port 8000
```

Serving, voice-cloning, and streaming recipes are in the [model card](https://huggingface.co/bosonai/higgs-audio-v3-tts-4b) and the [SGLang-Omni cookbook](https://sgl-project.github.io/sglang-omni/cookbook/higgs_tts.html).

You can also serve the open weights with **[vLLM-Omni](https://github.com/vllm-project/vllm-omni)**:

```bash
vllm-omni serve bosonai/higgs-audio-v3-tts-4b --host 0.0.0.0 --port 8095 --trust-remote-code --omni
```

This exposes the same OpenAI-compatible `/v1/audio/speech` API with zero-shot voice cloning. Plain-text TTS, voice-clone, and benchmark recipes are in the [vLLM-Omni Higgs Audio v3 recipe](https://github.com/vllm-project/vllm-omni/blob/main/recipes/BosonAI/Higgs-Audio-V3-TTS.md).

> [!NOTE]
> Higgs Audio v3 is released under the **Boson Higgs Audio v3 Research and Non-Commercial License**. Production / hosted / revenue-generating use requires a separate commercial license.

---

## Looking for Higgs Audio v2 / v2.5?

The full v2 / v2.5 documentation — installation, examples, technical details, and benchmarks — has moved to **[README_V2.md](./README_V2.md)**. Those models remain available on Hugging Face: [v2 (3B base)](https://huggingface.co/bosonai/higgs-audio-v2-generation-3B-base) and the [v2.5 blog](https://www.boson.ai/blog/higgs-audio-v2.5).

## Contribution and Support

For contribution and support guidelines, please see [SUPPORT_GUIDELINES.md](SUPPORT_GUIDELINES.md).

## We Are Hiring!

If you are passionate about multimodal AI, speech/audio models, or large-scale systems,
check out our open positions at [Boson AI Careers](https://jobs.lever.co/bosonai).

## Citation

```bibtex
@misc{bosonai_higgs_audio_tts_v3_2026,
  title  = {Higgs Audio v3 TTS: Conversational Speech for Voice AI from Boson AI},
  author = {Boson AI},
  year   = {2026},
  howpublished = {https://huggingface.co/bosonai/higgs-audio-v3-tts-4b},
}
```

## Third-Party Licenses

The `boson_multimodal/audio_processing/` directory contains code derived from third-party repositories, primarily from [xcodec](https://github.com/zhenye234/xcodec). See the [`LICENSE`](boson_multimodal/audio_processing/LICENSE) in that directory for attribution and licensing.
