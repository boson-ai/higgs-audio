import dacite
import pandas as pd
import torch
import json

import numpy as np
import multiprocessing as mp

from dataclasses import dataclass, fields
from abc import ABC, abstractmethod
from typing import Union, List, Dict, Optional

from ..data_types import ChatMLSample, TextContent, AudioContent
from ..constants import AUDIO_IN_TOKEN, AUDIO_OUT_TOKEN

from loguru import logger

# add lru_cache import
from functools import lru_cache

# Whisper processor, 30 sec -> 3000 features
# Then we divide 4 in the audio towker, we decrease 3000 features to 750, which gives 25 Hz
WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC = 25


@dataclass
class ChatMLDatasetSample:
    input_ids: torch.LongTensor  # Shape (seq_len,): The input text tokens.
    label_ids: torch.LongTensor  # Shape (seq_len,): The label ids.
    audio_ids_concat: torch.LongTensor  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    audio_ids_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio token in the concatenated audio tokens.
    audio_waveforms_concat: (
        torch.Tensor
    )  # Shape (total_wv_length,): The concatenated audio waveforms for audio-in features.
    audio_waveforms_start: (
        torch.LongTensor
    )  # Shape (num_audios,): The start index of each audio waveform in the concatenated audio waveforms.
    audio_sample_rate: torch.Tensor  # Shape (num_audios,): The sampling rate of the audio waveforms.
    audio_speaker_indices: (
        torch.LongTensor
    )  # Shape (num_audios,) -1 means unknown speaker: The speaker indices for each audio.
    audio_label_ids_concat: Optional[torch.LongTensor] = (
        None  # Shape (num_codebooks, audio_seq_len): The audio tokens that are concatenated.
    )
    # Here `audio_seq_len` is the length of the concatenated audio tokens.`
    reward: Optional[float] = None

    def num_audios(self):
        return max(len(self.audio_waveforms_start), len(self.audio_ids_start))

    def get_audio_codes(self, idx):
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_ids_concat[:, code_start:code_end]

    def get_audio_codes_labels(self, idx):
        if self.audio_label_ids_concat is None:
            return None
        code_start = self.audio_ids_start[idx]
        if idx < len(self.audio_ids_start) - 1:
            code_end = self.audio_ids_start[idx + 1]
        else:
            code_end = self.audio_ids_concat.shape[-1]

        return self.audio_label_ids_concat[:, code_start:code_end]

    def get_wv(self, idx):
        wv_start = self.audio_waveforms_start[idx]
        sr = self.audio_sample_rate[idx]
        if idx < len(self.audio_waveforms_start) - 1:
            wv_end = self.audio_waveforms_start[idx + 1]
        else:
            wv_end = self.audio_waveforms_concat.shape[-1]
        return self.audio_waveforms_concat[wv_start:wv_end], sr

    def cal_num_tokens(
        self,
        encode_whisper_embed: bool = True,
        encode_audio_in_tokens: bool = False,
        encode_audio_out_tokens: bool = True,
        audio_in_token_id: int = 128015,
        audio_out_token_id: int = 128016,
    ) -> int:
        # we firstly exclude <|AUDIO|> and <|AUDIO_OUT|> because we do late merging and replace those position with actual audio features and audio token ids
        # It's assumed that we always have audio_ids when audio_waveforms are there (but not vice-versa)
        num_tokens = len(self.input_ids) - len(self.audio_ids_start)

        if encode_whisper_embed and len(self.audio_waveforms_concat) > 0:
            audio_lengths = torch.diff(self.audio_waveforms_start)
            if len(audio_lengths):
                # Sum before calling .item()
                num_tokens += (
                    (
                        np.ceil(WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC * audio_lengths / self.audio_sample_rate[:-1])
                    ).sum()
                ).item()
            # add the last audio's token estimation
            num_tokens += (
                np.ceil(
                    WHISPER_EMBED_NUM_HIDDEN_STATE_PER_SEC
                    * (self.audio_waveforms_concat.shape[0] - self.audio_waveforms_start[-1])
                    / self.audio_sample_rate[-1]
                )
            ).item()

        if self.audio_ids_concat.size(1) > 0:
            audio_io_ids = self.input_ids[
                (self.input_ids == audio_in_token_id) | (self.input_ids == audio_out_token_id)
            ]
            audio_io_id_lengths = torch.concat(
                [
                    torch.diff(self.audio_ids_start),
                    torch.tensor([self.audio_ids_concat.shape[-1] - self.audio_ids_start[-1]]),
                ]
            )
            if encode_audio_in_tokens:
                num_tokens += torch.sum(audio_io_id_lengths[audio_io_ids == audio_in_token_id]).item()

            if encode_audio_out_tokens:
                num_tokens += torch.sum(audio_io_id_lengths[audio_io_ids == audio_out_token_id]).item()

        return int(num_tokens)

    @classmethod
    def merge(
        cls,
        samples: List["ChatMLDatasetSample"],
        eos_token_id: int,
        ignore_index: int,
        padding_size: Optional[int] = None,
    ) -> "ChatMLDatasetSample":
        """Merges a list of ChatMLDatasetSample instances, inserting eos_token_id and ignore_index between them, and adjusting offsets for audio_ids_start and audio_waveforms_start.

        Args:
            samples (List[ChatMLDatasetSample]): List of samples to merge.
            eos_token_id (int): Tokens to be inserted into input_ids between samples.
            ignore_index (int): Default label for padding.
            padding_size (Optional[int]): If provided, pad the sequence to with this length.

        Returns:
            ChatMLDatasetSample: Merged and potentially padded sample.
        """
        if not samples:
            logger.fatal("The samples list is empty and cannot be merged.")
            raise ValueError("The samples list is empty and cannot be merged.")

        # Initialize empty lists for concatenation
        input_ids_list = []
        label_ids_list = []
        audio_ids_concat_list = []
        audio_ids_start_list = []
        audio_waveforms_concat_list = []
        audio_waveforms_start_list = []
        audio_sample_rate_list = []
        audio_speaker_indices_list = []

        # Track offsets
        audio_ids_offset = 0
        audio_waveforms_offset = 0

        for sample in samples:
            # Add input_ids and label_ids with padding
            if input_ids_list:
                input_ids_list.append(torch.tensor([eos_token_id], dtype=torch.long))
                label_ids_list.append(torch.tensor([ignore_index], dtype=torch.long))
            input_ids_list.append(sample.input_ids)
            label_ids_list.append(sample.label_ids)

            # Add audio_ids_concat and handle empty audio ids
            if sample.audio_ids_concat.size(1) > 0:
                audio_ids_concat_list.append(sample.audio_ids_concat)

                # Offset and add audio_ids_start
                audio_ids_start_list.append(sample.audio_ids_start + audio_ids_offset)
                audio_ids_offset += sample.audio_ids_concat.size(
                    1
                )  # (num_codebooks, seq_len): Update offset by audio_seq_len

            # Add audio_waveforms_concat
            if sample.audio_waveforms_concat.size(0) > 0:
                # Check dimensions of the audio waveform to ensure consistency
                if (
                    audio_waveforms_concat_list
                    and sample.audio_waveforms_concat.dim() != audio_waveforms_concat_list[0].dim()
                ):
                    logger.warning(
                        f"Skipping audio waveform with inconsistent dimensions: expected {audio_waveforms_concat_list[0].dim()}D, got {sample.audio_waveforms_concat.dim()}D"
                    )
                    continue

                audio_waveforms_concat_list.append(sample.audio_waveforms_concat)
                audio_waveforms_start_list.append(sample.audio_waveforms_start + audio_waveforms_offset)
                audio_waveforms_offset += sample.audio_waveforms_concat.size(0)

                # Add audio_sample_rate and audio_speaker_indices
                audio_sample_rate_list.append(sample.audio_sample_rate)

            audio_speaker_indices_list.append(sample.audio_speaker_indices)

        # Concatenate all tensors
        input_ids = torch.cat(input_ids_list, dim=0)
        label_ids = torch.cat(label_ids_list, dim=0)

        # Apply padding if padding_size is specified
        if padding_size is not None and padding_size > 0:
            input_ids = torch.cat([input_ids, torch.full((padding_size,), eos_token_id, dtype=torch.long)], dim=0)
            label_ids = torch.cat([label_ids, torch.full((padding_size,), ignore_index, dtype=torch.long)], dim=0)

        # Safely concatenate audio tensors with proper error handling
        try:
            audio_ids_concat = torch.cat(audio_ids_concat_list, dim=1) if audio_ids_concat_list else torch.tensor([[]])
            audio_ids_start = torch.cat(audio_ids_start_list, dim=0) if audio_ids_start_list else torch.tensor([])

            # Check for dimensional consistency in audio waveforms
            if audio_waveforms_concat_list:
                dims = [t.dim() for t in audio_waveforms_concat_list]
                if not all(d == dims[0] for d in dims):
                    # If dimensions don't match, log warning and filter out the problematic tensors
                    logger.warning(
                        f"Inconsistent dimensions in audio waveforms: {dims}. Filtering to keep only consistent ones."
                    )
                    expected_dim = max(set(dims), key=dims.count)  # Most common dimension
                    audio_waveforms_concat_list = [t for t in audio_waveforms_concat_list if t.dim() == expected_dim]

                    # Recalculate audio_waveforms_start with the filtered list
                    if audio_waveforms_concat_list:
                        audio_waveforms_offset = 0
                        audio_waveforms_start_list = []
                        for waveform in audio_waveforms_concat_list:
                            audio_waveforms_start_list.append(torch.tensor([audio_waveforms_offset]))
                            audio_waveforms_offset += waveform.size(0)

            audio_waveforms_concat = (
                torch.cat(audio_waveforms_concat_list, dim=0) if audio_waveforms_concat_list else torch.tensor([])
            )
            audio_waveforms_start = (
                torch.cat(audio_waveforms_start_list, dim=0) if audio_waveforms_start_list else torch.tensor([])
            )
            audio_sample_rate = (
                torch.cat(audio_sample_rate_list, dim=0) if audio_sample_rate_list else torch.tensor([])
            )
            audio_speaker_indices = (
                torch.cat(audio_speaker_indices_list, dim=0) if audio_speaker_indices_list else torch.tensor([])
            )

        except RuntimeError as e:
            logger.error(f"Error during tensor concatenation: {str(e)}")
            logger.warning("Falling back to empty audio tensors")
            # Fall back to empty tensors
            audio_ids_concat = torch.tensor([[]])
            audio_ids_start = torch.tensor([])
            audio_waveforms_concat = torch.tensor([])
            audio_waveforms_start = torch.tensor([])
            audio_sample_rate = torch.tensor([])
            audio_speaker_indices = torch.tensor([])

        # Create the merged sample
        merged_sample = cls(
            input_ids=input_ids,
            label_ids=label_ids,
            audio_ids_concat=audio_ids_concat,
            audio_ids_start=audio_ids_start,
            audio_waveforms_concat=audio_waveforms_concat,
            audio_waveforms_start=audio_waveforms_start,
            audio_sample_rate=audio_sample_rate,
            audio_speaker_indices=audio_speaker_indices,
        )

        return merged_sample


@dataclass
class RankedChatMLDatasetSampleTuple:
    samples: List[ChatMLDatasetSample]
    scores: List[float]

    def max_score_sample(self) -> ChatMLDatasetSample:
        idx = self.scores.index(max(self.scores))
        self.samples[idx].reward = self.scores[idx]
        return self.samples[idx]

    def min_score_sample(self) -> ChatMLDatasetSample:
        idx = self.scores.index(min(self.scores))
        self.samples[idx].reward = self.scores[idx]
        return self.samples[idx]


@dataclass
class ChatMLDatasetStorageSample:
    input_tokens: torch.LongTensor
    label_tokens: torch.LongTensor
    audio_bytes_cache_dir_index: int
    audio_codes_cache_dir_index: int
    audio_bytes_indices: torch.LongTensor
    audio_codes_indices: torch.LongTensor
    speaker_indices: torch.LongTensor
    file_index: int
    original_sample_index: int


# TODO(sxjscience): We need to revist the logic about parsing speaker ids.
# Currently, we assume that the speaker id is stored at the "misc" field in ChatMLSample.
def prepare_chatml_sample(sample: Union[ChatMLSample, Dict], tokenizer):
    """Preprocess the ChatML sample to get the tokens for the text part.

    Args:
        sample (ChatMLSample): The ChatML sample to preprocess.
        tokenizer: The tokenizer to use for encoding the text.

    """

    try:
        if not isinstance(sample, ChatMLSample):
            # Handle all fields that could be NaN
            if "speaker" in sample and pd.isna(sample["speaker"]):
                sample["speaker"] = None
            if "start_index" in sample and pd.isna(sample["start_index"]):
                sample["start_index"] = None
            if "content" in sample and pd.isna(sample["content"]):
                sample["content"] = ""

            # Convert any other potential NaN values in nested structures
            def convert_nan_to_none(obj):
                import numpy as np

                if isinstance(obj, (pd.Series, np.ndarray)):
                    return obj.tolist()
                elif pd.api.types.is_scalar(obj) and pd.isna(obj):
                    return None
                elif isinstance(obj, dict):
                    return {k: convert_nan_to_none(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):  # Fixed: Handle both list and tuple
                    return [convert_nan_to_none(item) for item in obj]
                return obj

            # Clean the sample data
            clean_sample = convert_nan_to_none(sample)

            val_keys = []
            for field in fields(ChatMLSample):
                if field.name in clean_sample:
                    val_keys.append(field.name)
            clean_sample = {k: clean_sample[k] for k in val_keys}

            try:
                sample = dacite.from_dict(
                    data_class=ChatMLSample, data=clean_sample, config=dacite.Config(strict=True, check_types=True)
                )
            except Exception as e:
                print(f"Failed to convert to ChatMLSample: {e}")
                print(f"Clean sample: {json.dumps(clean_sample, indent=2)}")
                return None, None, None, None

        input_tokens = []
        label_tokens = []
        audio_contents = []
        speaker_id = None
        if sample.speaker is not None:
            speaker_id = sample.speaker
        elif sample.misc is not None:
            if "speaker" in sample.misc:
                speaker_id = sample.misc["speaker"]

        total_m = len(sample.messages)
        for turn_id, message in enumerate(sample.messages):
            role = message.role
            recipient = message.recipient
            content = message.content
            content_l = []

            if isinstance(content, str):
                content_l.append(TextContent(text=content))
            elif isinstance(content, TextContent):
                content_l.append(content)
            elif isinstance(content, AudioContent):
                content_l.append(content)
            elif isinstance(content, list):
                for ele in content:
                    if isinstance(ele, str):
                        content_l.append(TextContent(text=ele))
                    else:
                        content_l.append(ele)
            if turn_id == 0:
                prefix = f"<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n"
            else:
                prefix = f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            eot_postfix = "<|eot_id|>"
            eom_postfix = "<|eom_id|>"

            # Add token caching mechanism
            @lru_cache(maxsize=128)
            def cached_encode(text):
                return tokenizer.encode(text, add_special_tokens=False)

            # Use cached_encode instead of direct tokenizer.encode calls
            prefix_tokens = cached_encode(prefix)
            input_tokens.extend(prefix_tokens)
            label_tokens.extend([-100 for _ in prefix_tokens])

            if recipient:
                assert role == "assistant", "Recipient is only available for assistant role."
                recipient_tokens = tokenizer.encode(f"{recipient}<|recipient|>", add_special_tokens=False)
                input_tokens.extend(recipient_tokens)
                label_tokens.extend(recipient_tokens)

            for content in content_l:
                if content.type == "text":
                    text_tokens = tokenizer.encode(content.text, add_special_tokens=False)
                    input_tokens.extend(text_tokens)
                    if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                        label_tokens.extend(text_tokens)
                    else:
                        label_tokens.extend([-100 for _ in text_tokens])

                elif content.type == "audio":
                    # Generate the text-part of the audio tokens
                    audio_contents.append(content)
                    if role == "user" or role == "system":
                        # Add the text tokens
                        text_tokens = tokenizer.encode(
                            f"<|audio_bos|><|AUDIO|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                        label_tokens.extend([-100 for _ in text_tokens])
                    elif role == "assistant":
                        # Add the text tokens for audio-out part.
                        text_tokens = tokenizer.encode(
                            f"<|audio_out_bos|><|AUDIO_OUT|><|audio_eos|>",
                            add_special_tokens=False,
                        )
                        input_tokens.extend(text_tokens)
                        if sample.start_index is None or turn_id >= sample.start_index:
                            label_tokens.extend(text_tokens)
                        else:
                            label_tokens.extend([-100 for _ in text_tokens])
            next_id = turn_id + 1
            if role == "assistant" and next_id != total_m and sample.messages[next_id].role == "assistant":
                postfix_tokens = tokenizer.encode(eom_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            else:
                postfix_tokens = tokenizer.encode(eot_postfix, add_special_tokens=False)
                input_tokens.extend(postfix_tokens)
            if role == "assistant" and (sample.start_index is None or turn_id >= sample.start_index):
                label_tokens.extend(postfix_tokens)
            else:
                label_tokens.extend([-100 for _ in postfix_tokens])

        return input_tokens, label_tokens, audio_contents, speaker_id

    except Exception as e:
        print(f"Error in prepare_chatml_sample: {str(e)}")
        print(f"Sample data: {json.dumps(sample, indent=2)}")
        return None, None, None, None


def extract_generation_prompt_from_input_tokens(input_tokens, tokenizer):
    """Extract the generation prompt and reference answer from the input tokens.

    For example:

    Input Text = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    What words do you hear from the provided audio? Write it down for me.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\nAt first they went by quick, too quick to even get.<|eot_id|>'

    -->

    Prompt = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n
    What words do you hear from the provided audio? Write it down for me.<|audio_bos|><|AUDIO|><|audio_eos|><|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>\n\n',
    Reference = 'At first they went by quick, too quick to even get.'

    Args:
        input_tokens: The input tokens.
        audio_contents: The audio contents.
        tokenizer: The tokenizer to use for decoding the text.

    Returns:
        prompt_tokens: The tokens for the prompt.
        reference_answer: The reference answer.
        num_audios_in_reference: The number of audios in the reference answer.

    """
    input_text = tokenizer.decode(input_tokens)
    generation_prefix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    postfix = "<|eot_id|>"
    assert generation_prefix in input_text
    generation_prompt_end_loc = input_text.rfind(generation_prefix) + len(generation_prefix)
    generation_prompt = input_text[:generation_prompt_end_loc]
    reference_answer = input_text[generation_prompt_end_loc : input_text.find(postfix, generation_prompt_end_loc)]
    num_audios_in_reference = reference_answer.count(AUDIO_IN_TOKEN) + reference_answer.count(AUDIO_OUT_TOKEN)
    return tokenizer.encode(generation_prompt, add_special_tokens=False), reference_answer, num_audios_in_reference


def prepare_chatml_dataframe_single_process(df, tokenizer):
    """Prepare the ChatML DataFrame."""
    ret = []
    for _, row in df.iterrows():
        input_tokens, label_tokens, audio_contents, speaker_id = prepare_chatml_sample(row.to_dict(), tokenizer)
        ret.append((input_tokens, label_tokens, audio_contents, speaker_id))
    return ret


def prepare_chatml_dataframe(df, tokenizer, num_process=16):
    if num_process is None:
        return prepare_chatml_dataframe_single_process(df, tokenizer)
    else:
        num_process = max(min(len(df) // 1000, num_process), 1)
        workloads = np.array_split(df, num_process)
        with mp.Pool(num_process) as pool:
            ret = pool.starmap(
                prepare_chatml_dataframe_single_process, [(workload, tokenizer) for workload in workloads]
            )
    return sum(ret, [])


class DatasetInterface(ABC):
    @abstractmethod
    def __getitem__(self, idx) -> Union["ChatMLDatasetSample", "RankedChatMLDatasetSampleTuple"]:
        """Retrieve a dataset sample by index."""
        raise NotImplementedError


class IterableDatasetInterface(ABC):
    @abstractmethod
    def __iter__(self) -> Union["ChatMLDatasetSample", "RankedChatMLDatasetSampleTuple"]:
        """Retrieve a sample by iterating through the dataset."""
        raise NotImplementedError


@dataclass
class DatasetInfo:
    dataset_type: str
    group_type: Optional[str] = None
    mask_text: Optional[bool] = None  # Whether to mask the text tokens for pretraining samples.
