# Examples to use HiggsAudioServeEngine

The `run_hf_example.py` script provides three different examples for using the `HiggsAudioServeEngine`. 
Each example will generate an audio file (`output_{example}.wav`) in the current directory.

### Zero-Shot Voice Generation
Generate audio with specific voice characteristics (e.g., accents).

```bash
python run_hf_example.py zero_shot
```

### Voice Cloning
Clone a voice from a reference audio sample.

```bash
python run_hf_example.py voice_clone
```

### (Experimental) Interleaved Dialogue Generation
Higgs Audio v2 is also able to generate text. Here's an example that shows it is able to generate multi-speaker conversations with interleaved transcript and audio from scene descriptions.

```bash
python run_hf_example.py interleaved_dialogue
```
