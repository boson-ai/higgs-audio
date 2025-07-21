# Examples

> [!NOTE]  
> If you do not like the audio you get, you can generate multiple times with different seeds. In addition, you may need to apply text normalization to get the best performance, e.g. converting 70 °F to "seventy degrees Fahrenheit", and converting "1 2 3 4" to "one two three four". The model also performs better in longer sentences. Right now, the model has not been post-trained, we will release the post-trained model in the future.

## Single-speaker Audio Generation

### Voice clone

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio broom_salesman \
--seed 12345 \
--out_path generation.wav
```

The model will read the transcript with the same voice as in the [reference audio](./voice_prompts/broom_salesman.wav). The technique is also called shallow voice clone.

We have some example audio prompts stored in [voice_prompts](./voice_prompts/). Feel free to pick one in the folder and try out the model. Here's another example that uses the voice of `belinda`. You can also add new own favorite voice in the folder and clone the voice.

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio belinda \
--seed 12345 \
--out_path generation.wav
```

#### (Experimental) Cross-lingual voice clone

This example demonstrates voice cloning with a Chinese prompt, where the synthesized speech is in English.

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--scene_prompt empty \
--ref_audio zh_man_sichuan \
--temperature 0.3 \
--seed 12345 \
--out_path generation.wav
```

### Smart voice

The model supports reading the transcript with a random voice.

```bash
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--seed 12345 \
--out_path generation.wav
```

It also works for other languages like Chinese.

```bash
python3 generation.py \
--transcript transcript/single_speaker/zh_ai.txt \
--seed 12345 \
--out_path generation.wav
```

### Describe speaker characteristics with text

The model allows you to describe the speaker via text. See [voice_prompts/profile.yaml](voice_prompts/profile.yaml) for examples. You can run the following two examples that try to specify male / female British accent for the speakers. Also, try to remove the `--seed 12345` flag to see how the model is generating different voices.

```bash
# Male British Accent
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio profile:male_en_british \
--seed 12345 \
--out_path generation.wav

# Female British Accent
python3 generation.py \
--transcript transcript/single_speaker/en_dl.txt \
--ref_audio profile:female_en_british \
--seed 12345 \
--out_path generation.wav
```

### Chunking for long-form audio generation

To generate long-form audios, you can chunk the text and render each chunk one by one while putting the previous generated audio and the reference audio in the prompt. Here's an example that generates the first five paragraphs of Higgs Audio v1 release blog. See [text](./transcript/single_speaker/en_higgs_audio_blog.md).

```bash
python3 generation.py \
--scene_prompt scene_prompts/reading_blog.txt \
--transcript transcript/single_speaker/en_higgs_audio_blog.md \
--ref_audio en_man \
--chunk_method word \
--temperature 0.3 \
--generation_chunk_buffer_size 2 \
--seed 12345 \
--out_path generation.wav
```

### Experimental and Emergent Capabilities

As shown in our demo, the pretrained model is demonstrating emergent features. We prepared some samples to help you explore these experimental prompts. We will enhance the stability of these experimental prompts in the future version of HiggsAudio.

#### (Experimental) Hum a tune with the cloned voice
The model is able to hum a tune with the cloned voice.

```bash
python3 generation.py \
--transcript transcript/single_speaker/experimental/en_humming.txt \
--ref_audio en_woman \
--ras_win_len 0 \
--seed 12345 \
--out_path generation.wav
```

#### (Experimental) Read the sentence while adding background music (BGM)

```bash
python3 generation.py \
--transcript transcript/single_speaker/experimental/en_bgm.txt \
--ref_audio en_woman \
--ras_win_len 0 \
--ref_audio_in_system_message \
--seed 123456 \
--out_path generation.wav
```

## Multi-speaker Audio Generation


### Smart voice

To get started to explore HiggsAudio's capability in generating multi-speaker audios. Let's try to generate a multi-speaker dialog from transcript in the zero-shot fashion. See the transcript in [transcript/multi_speaker/en_argument.txt](transcript/multi_speaker/en_argument.txt). The speakers are annotated with `[SPEAKER0]` and `[SPEAKER1]`.

```bash
python3 generation.py \
--transcript transcript/multi_speaker/en_argument.txt \
--seed 12345 \
--out_path generation.wav
```

### Multi-voice clone
You can also try to clone the voices from multiple people simultaneously and generate audio about the transcript. Here's an example that puts reference audios in the system message and prompt the model iteratively. You can hear "Belinda" arguing with "Broom Salesman".

```bash
python3 generation.py \
--transcript transcript/multi_speaker/en_argument.txt \
--ref_audio belinda,broom_salesman \
--ref_audio_in_system_message \
--chunk_method speaker \
--seed 12345 \
--out_path generation.wav
```

You can also let "Broom Salesman" talking to "Belinda", who recently trained HiggsAudio.

```bash
python3 generation.py \
--transcript transcript/multi_speaker/en_higgs.txt \
--ref_audio broom_salesman,belinda \
--ref_audio_in_system_message \
--chunk_method speaker \
--chunk_max_num_turns 2 \
--seed 12345 \
--out_path generation.wav
```
