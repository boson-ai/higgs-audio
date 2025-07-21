import asyncio
import base64
import torch
import numpy as np
from io import BytesIO
from dataclasses import dataclass
from typing import List, Optional, Union
from copy import deepcopy
from transformers import AutoTokenizer, AutoProcessor
from transformers.cache_utils import StaticCache
from transformers.generation.streamers import BaseStreamer
from transformers.generation.stopping_criteria import StoppingCriteria
from dataclasses import asdict
from loguru import logger
import threading
import librosa


from ..dataset.chatml_dataset import ChatMLSample, ChatMLDatasetSample, prepare_chatml_sample
from ..model.higgs_audio import HiggsAudioModel
from ..model.higgs_audio.utils import revert_delay_pattern
from ..data_collator.higgs_audio_collator import HiggsAudioSampleCollator
from ..audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer


@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    finish_reason: Optional[str] = None


class AsyncHiggsAudioStreamer(BaseStreamer):
    """
    Async streamer that handles both text and audio token generation from Higgs-Audio model.
    Stores chunks in a queue to be consumed by downstream applications.

    Parameters:
        tokenizer (`AutoTokenizer`):
            The tokenizer used to decode text tokens.
        skip_prompt (`bool`, *optional*, defaults to `False`):
            Whether to skip the prompt tokens in generation.
        timeout (`float`, *optional*):
            The timeout for the queue. If `None`, the queue will block indefinitely.
        decode_kwargs (`dict`, *optional*):
            Additional keyword arguments to pass to the tokenizer's `decode` method.

    Examples:
        ```python
        >>> from transformers import AutoTokenizer
        >>> from threading import Thread
        >>> import asyncio

        >>> tokenizer = AutoTokenizer.from_pretrained("path/to/higgs/tokenizer")
        >>> model = HiggsAudioModel.from_pretrained("path/to/higgs/model")
        >>> inputs = tokenizer(["Generate some text and audio:"], return_tensors="pt")

        >>> async def main():
        ...     streamer = AsyncHiggsAudioStreamer(tokenizer)
        ...     generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=20)
        ...     thread = Thread(target=model.generate, kwargs=generation_kwargs)
        ...     thread.start()
        ...
        ...     async for delta in streamer:
        ...         if delta.text is not None:
        ...             print("Text:", delta.text)
        ...         if delta.audio_tokens is not None:
        ...             print("Audio tokens shape:", delta.audio_tokens.shape)
        >>> asyncio.run(main())
        ```
    """

    def __init__(
        self,
        tokenizer: "AutoTokenizer",
        skip_prompt: bool = False,
        timeout: Optional[float] = None,
        audio_num_codebooks: int = 1,
        **decode_kwargs,
    ):
        self.tokenizer = tokenizer
        self.skip_prompt = skip_prompt
        self.timeout = timeout
        self.decode_kwargs = decode_kwargs
        self.audio_num_codebooks = audio_num_codebooks
        # Queue to store generated chunks
        self.queue = asyncio.Queue()
        self.stop_signal = None

        # Get running event loop
        self.loop = asyncio.get_running_loop()
        self.has_asyncio_timeout = hasattr(asyncio, "timeout")

        # State tracking
        self.next_tokens_are_prompt = True

    def put(self, value: torch.Tensor):
        """
        Receives tokens and processes them as either text or audio tokens.
        For text tokens, decodes and caches them until complete words are formed.
        For audio tokens, directly queues them.
        """
        if value.shape[0] > 1 and not self.next_tokens_are_prompt:
            # This is likely audio tokens (shape: [audio_num_codebooks])
            assert value.shape[0] == self.audio_num_codebooks, "Number of codebooks mismatch"
            delta = HiggsAudioStreamerDelta(audio_tokens=value)
            self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)
            return

        # Skip prompt tokens if configured
        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        # Process as text tokens
        if len(value.shape) > 1:
            value = value[0]

        text = self.tokenizer.decode(value, **self.decode_kwargs)
        delta = HiggsAudioStreamerDelta(text=text, text_tokens=value)
        self.loop.call_soon_threadsafe(self.queue.put_nowait, delta)

    def end(self):
        """Flushes any remaining text tokens and signals the end of generation."""
        self.next_tokens_are_prompt = True
        self.loop.call_soon_threadsafe(self.queue.put_nowait, self.stop_signal)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            if self.has_asyncio_timeout:
                async with asyncio.timeout(self.timeout):
                    value = await self.queue.get()
            else:
                value = await asyncio.wait_for(self.queue.get(), timeout=self.timeout)
        except asyncio.TimeoutError:
            raise TimeoutError()
        else:
            if value == self.stop_signal:
                raise StopAsyncIteration()
            else:
                return value


class AsyncStoppingCriteria(StoppingCriteria):
    """
    Stopping criteria that checks for stop signal from a threading event.

    Args:
        stop_signal (threading.Event): Event that will receive stop signals
    """

    def __init__(self, stop_signal: threading.Event):
        self.stop_signal = stop_signal

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.stop_signal.is_set():
            logger.info(f"Stop signal received. Can be caused by client disconnection.")
            return True
        return False


@dataclass
class HiggsAudioResponse:
    audio: Optional[np.ndarray] = None
    generated_audio_tokens: Optional[np.ndarray] = None
    sampling_rate: Optional[int] = None
    generated_text: str = ""
    generated_text_tokens: Optional[np.ndarray] = None
    usage: Optional[dict] = None


class HiggsAudioServeEngine:
    def __init__(
        self,
        model_name_or_path: str,
        audio_tokenizer_name_or_path: str,
        tokenizer_name_or_path: Optional[str] = None,
        device: str = "cuda",
        torch_dtype: Union[torch.dtype, str] = "auto",
        kv_cache_lengths: List[int] = [1024, 4096, 8192],  # Multiple KV cache sizes
    ):
        """
        Initialize the HiggsAudioServeEngine, a serving wrapper for the HiggsAudioModel.
        The model, tokenizer, and audio tokenizer will be downloaded from the Hugging Face Hub if they are not local.

        Args:
            model_name_or_path (str):
                The name or path of the model to load.
            audio_tokenizer_name_or_path (str):
                The name or path of the audio tokenizer to load.
            tokenizer_name_or_path (str):
                The name or path of the tokenizer to load.
            device (str):
                The device to use for the model.
            kv_cache_lengths (List[int]):
                The lengths of the KV caches to use for the model. Used for cuda graph capture when device is cuda.
            torch_dtype (Union[torch.dtype, str]):
                The dtype to use for the model.
        """
        self.device = device
        self.model_name_or_path = model_name_or_path
        self.torch_dtype = torch_dtype

        # Initialize model and tokenizer
        self.model = HiggsAudioModel.from_pretrained(model_name_or_path, torch_dtype=torch_dtype).to(device)
        logger.info(f"Loaded model from {model_name_or_path}, dtype: {self.model.dtype}")

        if tokenizer_name_or_path is None:
            tokenizer_name_or_path = model_name_or_path
        logger.info(f"Loading tokenizer from {tokenizer_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)

        logger.info(f"Initializing Higgs Audio Tokenizer")
        self.audio_tokenizer = load_higgs_audio_tokenizer(audio_tokenizer_name_or_path, device=device)

        self.audio_num_codebooks = self.model.config.audio_num_codebooks
        self.audio_codebook_size = self.model.config.audio_codebook_size
        self.audio_tokenizer_tps = self.audio_tokenizer.tps
        self.samples_per_token = int(self.audio_tokenizer.sampling_rate // self.audio_tokenizer_tps)
        self.hamming_window_len = 2 * self.audio_num_codebooks * self.samples_per_token
        # Set the audio special tokens
        self.model.set_audio_special_tokens(self.tokenizer)

        # Prepare KV caches for different lengths
        cache_config = deepcopy(self.model.config.text_config)
        cache_config.num_hidden_layers = self.model.config.text_config.num_hidden_layers
        if self.model.config.audio_dual_ffn_layers:
            cache_config.num_hidden_layers += len(self.model.config.audio_dual_ffn_layers)
        # A list of KV caches for different lengths
        self.kv_caches = {
            length: StaticCache(
                config=cache_config,
                max_batch_size=1,
                max_cache_len=length,
                device=self.model.device,
                dtype=self.model.dtype,
            )
            for length in sorted(kv_cache_lengths)
        }

        if self.model.config.encode_whisper_embed:
            logger.info(f"Loading whisper processor")
            whisper_processor = AutoProcessor.from_pretrained(
                "openai/whisper-large-v3-turbo",
                trust_remote=True,
                device=self.device,
            )
        else:
            whisper_processor = None

        # Reuse collator to prepare inference samples
        self.collator = HiggsAudioSampleCollator(
            whisper_processor=whisper_processor,
            encode_whisper_embed=self.model.config.encode_whisper_embed,
            audio_in_token_id=self.model.config.audio_in_token_idx,
            audio_out_token_id=self.model.config.audio_out_token_idx,
            audio_stream_bos_id=self.model.config.audio_stream_bos_id,
            audio_stream_eos_id=self.model.config.audio_stream_eos_id,
            pad_token_id=self.model.config.pad_token_id,
            return_audio_in_tokens=False,
            use_delay_pattern=self.model.config.use_delay_pattern,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            round_to=1,
        )

        # Capture CUDA graphs for each KV cache length
        if device == "cuda":
            logger.info(f"Capturing CUDA graphs for each KV cache length")
            self.model.capture_model(self.kv_caches.values())

    def _prepare_inputs(self, chat_ml_sample: ChatMLSample, force_audio_gen: bool = False):
        input_tokens, _, audio_contents, _ = prepare_chatml_sample(
            chat_ml_sample,
            self.tokenizer,
        )

        postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        if force_audio_gen:
            postfix += "<|audio_out_bos|>"
        postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
        input_tokens.extend(postfix)

        # Configure the audio inputs
        audio_ids_l = []
        for audio_content in audio_contents:
            if audio_content.audio_url not in ["placeholder", ""]:
                raw_audio, _ = librosa.load(audio_content.audio_url, sr=self.audio_tokenizer.sampling_rate)
            elif audio_content.raw_audio is not None:
                raw_audio, _ = librosa.load(
                    BytesIO(base64.b64decode(audio_content.raw_audio)), sr=self.audio_tokenizer.sampling_rate
                )
            else:
                raw_audio = None

            if raw_audio is not None:
                audio_ids = self.audio_tokenizer.encode(raw_audio, self.audio_tokenizer.sampling_rate)
                audio_ids_l.append(audio_ids.squeeze(0).cpu())

        if len(audio_ids_l) > 0:
            audio_ids_start = torch.tensor(
                np.cumsum(np.array([0] + [audio_ids.shape[1] for audio_ids in audio_ids_l])),
                dtype=torch.long,
                device=self.device,
            )[0:-1]
            audio_ids_concat = torch.cat(audio_ids_l, dim=1)
        else:
            audio_ids_start = None
            audio_ids_concat = None

        sample = ChatMLDatasetSample(
            input_ids=torch.LongTensor(input_tokens),
            label_ids=None,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=None,
            audio_waveforms_start=None,
            audio_sample_rate=None,
            audio_speaker_indices=None,
        )
        data = self.collator([sample])
        inputs = asdict(data)
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)

        return inputs

    def _prepare_kv_caches(self):
        for kv_cache in self.kv_caches.values():
            kv_cache.reset()

    def generate(
        self,
        chat_ml_sample: ChatMLSample,
        max_new_tokens: int,
        temperature: float = 0.7,
        top_k: Optional[int] = None,
        top_p: float = 0.95,
        stop_strings: Optional[List[str]] = None,
        force_audio_gen: bool = False,
        ras_win_len: Optional[int] = 7,
        ras_win_max_num_repeat: int = 2,
        seed: Optional[int] = None,
    ):
        """
        Generate audio from a chatml sample.
        Args:
            chat_ml_sample: A chatml sample.
            max_new_tokens: The maximum number of new tokens to generate.
            temperature: The temperature to use for the generation.
            top_p: The top p to use for the generation.
            stop_strings: A list of strings to stop the generation.
            force_audio_gen: Whether to force audio generation. This ensures the model generates audio tokens rather than text tokens.
            ras_win_len: The length of the RAS window. We use 7 by default. You can disable it by setting it to None or <=0.
            ras_win_max_num_repeat: The maximum number of times to repeat the RAS window.
        Returns:
            A dictionary with the following keys:
                audio: The generated audio.
                sampling_rate: The sampling rate of the generated audio.
        """
        # Default stop strings
        if stop_strings is None:
            stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
        if ras_win_len is not None and ras_win_len <= 0:
            ras_win_len = None

        with torch.no_grad():
            inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)
            prompt_token_ids = inputs["input_ids"][0].cpu().numpy()

            self._prepare_kv_caches()

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                stop_strings=stop_strings,
                tokenizer=self.tokenizer,
                do_sample=False if temperature == 0.0 else True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                past_key_values_buckets=self.kv_caches,
                ras_win_len=ras_win_len,
                ras_win_max_num_repeat=ras_win_max_num_repeat,
                seed=seed,
            )

            if len(outputs[1]) > 0:
                wv_list = []
                for output_audio in outputs[1]:
                    vq_code = revert_delay_pattern(output_audio).clip(0, self.audio_codebook_size - 1)[:, 1:-1]
                    wv_numpy = self.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]
                    wv_list.append(wv_numpy)
                wv_numpy = np.concatenate(wv_list)
            else:
                wv_numpy = None

            # We only support one request at a time now
            generated_text_tokens = outputs[0][0].cpu().numpy()[len(prompt_token_ids) :]
            generated_text = self.tokenizer.decode(generated_text_tokens)
            generated_audio_tokens = outputs[1][0].cpu().numpy()
            return HiggsAudioResponse(
                audio=wv_numpy,
                generated_audio_tokens=generated_audio_tokens,
                sampling_rate=self.audio_tokenizer.sampling_rate,
                generated_text=generated_text,
                generated_text_tokens=generated_text_tokens,
                usage={
                    "prompt_tokens": prompt_token_ids.shape[0],
                    "completion_tokens": generated_text_tokens.shape[0] + generated_audio_tokens.shape[1],
                    "total_tokens": (
                        prompt_token_ids.shape[0] + generated_text_tokens.shape[0] + generated_audio_tokens.shape[1]
                    ),
                    "cached_tokens": 0,
                },
            )
