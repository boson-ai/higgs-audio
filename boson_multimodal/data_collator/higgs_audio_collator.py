import librosa
import torch
import torch.nn.functional as F
import math
from typing import List, Tuple

from dataclasses import dataclass
from typing import Optional
from transformers.models.whisper.processing_whisper import WhisperProcessor

from ..dataset.chatml_dataset import ChatMLDatasetSample
from ..model.higgs_audio.utils import build_delay_pattern_mask


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def _ceil_to_next_power_of_two(self, x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


@dataclass
class HiggsAudioBatchInput:
    input_ids: torch.LongTensor  # shape (bsz, seq_len).
    attention_mask: torch.Tensor  # shape (bsz, seq_len).
    audio_features: Optional[torch.Tensor]  # shape (num_audio_in, feature_dim, max_mel_seq_len).
    audio_feature_attention_mask: Optional[torch.Tensor]  # shape (num_audio_in, max_mel_seq_len).
    audio_out_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    audio_out_ids_start: Optional[torch.LongTensor]  # shape (num_audio_out,)
    # The audio_out_ids_start_group_loc has the same length as audio_out_ids_start. It is used to recover group location in a batch for an audio segment
    # Currently, we concatenante audio segments along dim 0 to handle variadic audio segment length. However, in the alignment stage, we need the location information
    # For example,
    #  audio_out_ids_start = [0, 2, 4, 8]; and the first two audio segments come from the same sample in a batch, and other two come from different samples.
    #  This is a batch of 3 samples, then we will have the group location as:
    #  audio_out_ids_start_group_loc = [0, 0, 1, 2]
    audio_out_ids_start_group_loc: Optional[
        torch.LongTensor
    ]  # shape (num_audio_out,), specify which a sample's group location in the batch
    audio_in_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_in_total_length)
    audio_in_ids_start: Optional[torch.LongTensor]  # shape (num_audio_in,)
    label_ids: Optional[torch.LongTensor]  # shape (bsz, seq_len)
    label_audio_ids: Optional[torch.LongTensor]  # shape (num_codebooks, audio_out_total_length)
    reward: Optional[float] = None


class HiggsAudioSampleCollator:
    """Sample collator for Higgs-Audio model.

    Args:
        whisper_processor (WhisperProcessor): The whisper processor.
        audio_in_token_id (int): The token id for audio-in.
        audio_out_token_id (int): The token id for audio-out.
        pad_token_id (int): The token id for padding.
        audio_stream_bos_id (int): The token id for audio-stream beginning of sentence.
        audio_stream_eos_id (int): The token id for audio-stream end of sentence.
        round_to (int): The round-to value.
        pad_left (bool): Whether to pad left.
        return_audio_in_tokens (bool): Whether to return audio-in tokens.
        use_delay_pattern (bool): Whether to use delay pattern.
        disable_audio_codes_transform (bool): Whether to add bos and eos tokens to audio codes.
        chunk_size_seconds (int): The chunk size in seconds.
        add_new_bos_eos_for_long_chunk (bool): Whether to add new bos and eos tokens for long chunks.
        mask_audio_out_token_label (bool): Whether to always mask the label associated with <|AUDIO_OUT|> token. Since we will always have `<|AUDIO_OUT|>` after `<|audio_bos|>`, we can safely mask <|AUDIO_OUT|>.

    """

    def __init__(
        self,
        whisper_processor: WhisperProcessor,
        audio_in_token_id,
        audio_out_token_id,
        pad_token_id,
        audio_stream_bos_id,
        audio_stream_eos_id,
        round_to=8,
        pad_left=False,
        encode_whisper_embed=True,
        return_audio_in_tokens=True,
        audio_num_codebooks=None,
        use_delay_pattern=False,
        disable_audio_codes_transform=False,
        chunk_size_seconds=30,  # Maximum duration for each chunk
        add_new_bos_eos_for_long_chunk=True,
        mask_audio_out_token_label=True,
    ):
        self.whisper_processor = whisper_processor
        self.round_to = round_to
        self.pad_left = pad_left
        self.audio_in_token_id = audio_in_token_id
        self.audio_out_token_id = audio_out_token_id
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.pad_token_id = pad_token_id
        self.encode_whisper_embed = encode_whisper_embed
        self.return_audio_in_tokens = return_audio_in_tokens
        self.audio_num_codebooks = audio_num_codebooks
        self.use_delay_pattern = use_delay_pattern
        if encode_whisper_embed:
            self.chunk_size_seconds = chunk_size_seconds
            self.chunk_size_samples = int(chunk_size_seconds * whisper_processor.feature_extractor.sampling_rate)
        else:
            self.chunk_size_seconds = None
            self.chunk_size_samples = None
        self.disable_audio_codes_transform = disable_audio_codes_transform
        self.add_new_bos_eos_for_long_chunk = add_new_bos_eos_for_long_chunk
        self.mask_audio_out_token_label = mask_audio_out_token_label

    def _process_and_duplicate_audio_tokens(
        self, input_ids: torch.Tensor, audio_idx: int, wv: torch.Tensor, sr: int, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Process long audio and duplicate corresponding audio tokens.

        Args:
            input_ids: Input token ids
            audio_idx: Index of the audio token in the sequence
            wv: Audio waveform
            sr: Sample rate
            labels: Optional label ids to be duplicated alongside input ids

        Returns:
            Tuple of:
                - New input ids with duplicated audio tokens
                - New label ids (if labels were provided) or None
                - Number of chunks created
        """
        # Calculate number of chunks needed
        total_samples = len(wv)
        num_chunks = math.ceil(total_samples / self.chunk_size_samples)

        if num_chunks <= 1:
            return input_ids, labels, 1

        # Get the three tokens: <|audio_bos|><|AUDIO|><|audio_eos|>
        audio_token_seq = input_ids[audio_idx - 1 : audio_idx + 2]
        # Duplicate sequence for each chunk
        duplicated_sequence = audio_token_seq.repeat(num_chunks)

        # Create new input_ids with duplicated tokens
        new_input_ids = torch.cat([input_ids[: audio_idx - 1], duplicated_sequence, input_ids[audio_idx + 2 :]])

        # If labels are provided, duplicate them as well
        new_labels = None
        if labels is not None:
            label_seq = labels[audio_idx - 1 : audio_idx + 2]
            duplicated_labels = label_seq.repeat(num_chunks)
            new_labels = torch.cat([labels[: audio_idx - 1], duplicated_labels, labels[audio_idx + 2 :]])

        return new_input_ids, new_labels, num_chunks

    def __call__(self, batch: List[ChatMLDatasetSample]):
        """Collate the input data with support for long audio processing."""

        label_ids = None
        label_audio_ids = None
        if all([ele.label_ids is None for ele in batch]):
            return_labels = False
        else:
            return_labels = True

        if self.encode_whisper_embed:
            # Process each sample in the batch to handle long audio
            # TODO(?) The implementation here can be optimized.
            processed_batch = []
            for i in range(len(batch)):
                sample = batch[i]
                audio_in_mask = sample.input_ids == self.audio_in_token_id
                audio_in_indices = torch.where(audio_in_mask)[0]
                audio_out_mask = sample.input_ids == self.audio_out_token_id

                # Process each audio token and duplicate if needed
                modified_input_ids = sample.input_ids
                modified_labels = sample.label_ids if return_labels else None
                modified_waveforms_concat = []
                modified_waveforms_start = []
                modified_sample_rate = []
                offset = 0  # Track position changes from duplicating tokens
                curr_wv_offset = 0

                # Process input audio tokens
                for idx, audio_idx in enumerate(audio_in_indices):
                    # Get the audio for this token
                    wv, sr = sample.get_wv(idx)  # Use idx since we want the original audio index
                    if sr != self.whisper_processor.feature_extractor.sampling_rate:
                        resampled_wv = librosa.resample(
                            wv.cpu().numpy(),
                            orig_sr=sr,
                            target_sr=self.whisper_processor.feature_extractor.sampling_rate,
                        )
                    else:
                        resampled_wv = wv.cpu().numpy()
                    wv = torch.tensor(resampled_wv, device=wv.device)
                    sr = self.whisper_processor.feature_extractor.sampling_rate

                    # Process and duplicate tokens if necessary
                    token_pos = audio_idx + offset
                    modified_input_ids, modified_labels, num_chunks = self._process_and_duplicate_audio_tokens(
                        modified_input_ids, token_pos, wv, sr, modified_labels
                    )

                    # Update audio data
                    for chunk_idx in range(num_chunks):
                        chunk_start = chunk_idx * self.chunk_size_samples
                        chunk_end = min((chunk_idx + 1) * self.chunk_size_samples, len(wv))
                        chunk_wv = wv[chunk_start:chunk_end]
                        modified_waveforms_concat.append(chunk_wv)
                        modified_waveforms_start.append(curr_wv_offset)
                        curr_wv_offset += len(chunk_wv)
                        modified_sample_rate.append(sr)

                    # Update offset for next iteration
                    offset += (num_chunks - 1) * 3  # Each new chunk adds 3 more tokens

                # Create new sample with modified tokens and audio data
                processed_sample = ChatMLDatasetSample(
                    input_ids=modified_input_ids,
                    label_ids=modified_labels if return_labels else sample.label_ids,
                    audio_ids_concat=sample.audio_ids_concat,
                    audio_ids_start=sample.audio_ids_start,
                    audio_waveforms_concat=torch.cat(modified_waveforms_concat)
                    if modified_waveforms_concat
                    else sample.audio_waveforms_concat,
                    audio_waveforms_start=torch.tensor(modified_waveforms_start, dtype=torch.long)
                    if modified_waveforms_start
                    else sample.audio_waveforms_start,
                    audio_sample_rate=torch.tensor(modified_sample_rate)
                    if modified_sample_rate
                    else sample.audio_sample_rate,
                    audio_speaker_indices=torch.tensor([]),
                    # FIXME(sxjscience): The logic here is not correct for audio_label_ids_concat.
                    audio_label_ids_concat=sample.audio_label_ids_concat,
                )
                # audio_in_chunk_len = len(torch.where(modified_input_ids == self.audio_in_token_id)[0])
                # assert audio_in_chunk_len == processed_sample.num_audios(), f"Mismatch: audio_in_chunk_len={audio_in_chunk_len}, processed_sample.num_audios()={processed_sample.num_audios()}"
                processed_batch.append(processed_sample)
        else:
            processed_batch = batch

        # Get the max sequence length based on processed batch
        max_seq_length = _ceil_to_nearest(max([len(sample.input_ids) for sample in processed_batch]), self.round_to)

        # Get the ids for audio-in and audio-out for each batch
        audio_in_wv_l = []
        audio_in_ids_l = []
        audio_out_ids_l = []
        audio_out_ids_group_loc_l = []
        audio_in_label_ids_l = None
        audio_out_label_ids_l = None
        reward_l = []

        if return_labels:
            audio_out_no_train_flag = []  # Whether the audio-out data should be trained on or not.

        # Process the audio inputs and outputs
        for i in range(len(processed_batch)):
            audio_in_mask = processed_batch[i].input_ids == self.audio_in_token_id
            audio_out_mask = processed_batch[i].input_ids == self.audio_out_token_id
            audio_ids = torch.ones_like(processed_batch[i].input_ids)
            audio_ids[audio_in_mask ^ audio_out_mask] = torch.cumsum(audio_ids[audio_in_mask ^ audio_out_mask], 0) - 1
            audio_in_ids = audio_ids[audio_in_mask]
            audio_out_ids = audio_ids[audio_out_mask]

            if return_labels:
                audio_out_no_train_flag.append(processed_batch[i].label_ids[audio_out_mask] < 0)
                if self.mask_audio_out_token_label:
                    processed_batch[i].label_ids[audio_out_mask] = -100

            # Process audio inputs
            if self.return_audio_in_tokens:
                audio_in_ids_l.extend(
                    [processed_batch[i].get_audio_codes(idx)[: self.audio_num_codebooks, :] for idx in audio_in_ids]
                )
                if processed_batch[i].audio_label_ids_concat is not None:
                    if audio_in_label_ids_l is None:
                        audio_in_label_ids_l = []
                    audio_in_label_ids_l.extend(
                        [
                            processed_batch[i].get_audio_codes_labels(idx)[: self.audio_num_codebooks, :]
                            for idx in audio_in_ids
                        ]
                    )

            audio_out_ids_l.extend(
                [processed_batch[i].get_audio_codes(idx)[: self.audio_num_codebooks, :] for idx in audio_out_ids]
            )
            audio_out_ids_group_loc_l.append(i)
            if processed_batch[i].reward is not None:
                reward_l.append(processed_batch[i].reward)

            if processed_batch[i].audio_label_ids_concat is not None:
                if audio_out_label_ids_l is None:
                    audio_out_label_ids_l = []
                audio_out_label_ids_l.extend(
                    [
                        processed_batch[i].get_audio_codes_labels(idx)[: self.audio_num_codebooks, :]
                        for idx in audio_out_ids
                    ]
                )

            if self.encode_whisper_embed:
                for idx in audio_in_ids:
                    wv, sr = processed_batch[i].get_wv(idx)
                    resampled_wv = wv.cpu().numpy()
                    # Split long audio into chunks
                    total_samples = len(resampled_wv)
                    for chunk_start in range(0, total_samples, self.chunk_size_samples):
                        chunk_end = min(chunk_start + self.chunk_size_samples, total_samples)
                        chunk = resampled_wv[chunk_start:chunk_end]
                        audio_in_wv_l.append(chunk)
            # assert len(audio_in_wv_l) == processed_batch[i].num_audios(), \
            #     f"Assertion failed: Mismatch in number of audios. " \
            #     f"Expected {processed_batch[i].num_audios()}, but got {len(audio_in_wv_l)} at index {i}."

        if return_labels:
            audio_out_no_train_flag = torch.cat(audio_out_no_train_flag, dim=0)

        # Process all audio features
        if len(audio_in_wv_l) > 0:
            feature_ret = self.whisper_processor.feature_extractor(
                audio_in_wv_l,
                sampling_rate=self.whisper_processor.feature_extractor.sampling_rate,
                return_attention_mask=True,
                padding="max_length",
            )
            audio_features = torch.from_numpy(feature_ret["input_features"])
            audio_feature_attention_mask = torch.from_numpy(feature_ret["attention_mask"])
        else:
            if self.encode_whisper_embed:
                audio_features = torch.zeros(
                    (
                        0,
                        self.whisper_processor.feature_extractor.feature_size,
                        self.whisper_processor.feature_extractor.nb_max_frames,
                    ),
                    dtype=torch.float32,
                )
                audio_feature_attention_mask = torch.zeros(
                    (0, self.whisper_processor.feature_extractor.nb_max_frames), dtype=torch.int32
                )
            else:
                audio_features = None
                audio_feature_attention_mask = None

        # Process audio input tokens
        if len(audio_in_ids_l) > 0:
            # Append audio-stream-bos and eos tokens
            new_audio_in_ids_l = []
            for ele in audio_in_ids_l:
                if self.disable_audio_codes_transform:
                    # Do not add audio-stream-bos or eos tokens.
                    # This may indicate that the sample comes from ConstantLengthDatasetWithBuffer.
                    audio_codes = ele
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long),
                            ele,
                            torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    if self.use_delay_pattern:
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                new_audio_in_ids_l.append(audio_codes)
            audio_in_ids = torch.cat(new_audio_in_ids_l, dim=1).long()
            audio_in_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_in_ids_l[:-1]]), dim=0
            )
        else:
            audio_in_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_in_ids_start = torch.zeros(0, dtype=torch.long)

        # Process audio output tokens
        audio_out_ids_start_group_loc = None
        if len(audio_out_ids_l) > 0:
            new_audio_out_ids_l = []
            label_audio_ids_l = []
            for idx, ele in enumerate(audio_out_ids_l):
                if self.disable_audio_codes_transform:
                    # Do not add audio-stream-bos or eos tokens.
                    # This may indicate that the sample comes from ConstantLengthDatasetWithBuffer.
                    audio_codes = ele
                    if return_labels:
                        label_audio_ids = audio_out_label_ids_l[idx]
                else:
                    audio_codes = torch.cat(
                        [
                            torch.full((ele.shape[0], 1), self.audio_stream_bos_id, dtype=torch.long),
                            ele,
                            torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                        ],
                        dim=1,
                    )
                    if return_labels:
                        label_audio_ids = torch.cat(
                            [
                                torch.full((ele.shape[0], 1), -100, dtype=torch.long),
                                ele,
                                torch.full((ele.shape[0], 1), self.audio_stream_eos_id, dtype=torch.long),
                            ],
                            dim=1,
                        )
                    if self.use_delay_pattern:
                        audio_codes = build_delay_pattern_mask(
                            audio_codes.unsqueeze(0),
                            bos_token_id=self.audio_stream_bos_id,
                            pad_token_id=self.audio_stream_eos_id,
                        )[0].squeeze(0)
                        if return_labels:
                            label_audio_ids = build_delay_pattern_mask(
                                label_audio_ids.unsqueeze(0),
                                bos_token_id=-100,
                                pad_token_id=-100,
                            )[0].squeeze(0)
                new_audio_out_ids_l.append(audio_codes)

                if return_labels:
                    if audio_out_no_train_flag[idx]:
                        label_audio_ids[:] = -100
                    label_audio_ids_l.append(label_audio_ids)

            audio_out_ids = torch.cat(new_audio_out_ids_l, dim=1).long()
            if return_labels:
                label_audio_ids = torch.cat(label_audio_ids_l, dim=1).long()
            audio_out_ids_start = torch.cumsum(
                torch.tensor([0] + [audio_codes.shape[1] for audio_codes in new_audio_out_ids_l[:-1]]), dim=0
            )
            audio_out_ids_start_group_loc = torch.tensor(audio_out_ids_group_loc_l, dtype=torch.long)
        else:
            audio_out_ids = torch.zeros((0, 0), dtype=torch.long)
            audio_out_ids_start = torch.zeros(0, dtype=torch.long)
            if return_labels:
                label_audio_ids = torch.zeros((0, 0), dtype=torch.long)

        reward = torch.tensor(reward_l, dtype=torch.float32)

        # Handle padding for input ids and attention mask
        if self.pad_left:
            input_ids = torch.stack(
                [
                    F.pad(ele.input_ids, (max_seq_length - len(ele.input_ids), 0), value=self.pad_token_id)
                    for ele in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(ele.label_ids, (max_seq_length - len(ele.label_ids), 0), value=-100)
                        for ele in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(torch.ones_like(ele.input_ids), (max_seq_length - len(ele.input_ids), 0), value=0)
                    for ele in processed_batch
                ]
            )
        else:
            input_ids = torch.stack(
                [
                    F.pad(ele.input_ids, (0, max_seq_length - len(ele.input_ids)), value=self.pad_token_id)
                    for ele in processed_batch
                ]
            )
            if return_labels:
                label_ids = torch.stack(
                    [
                        F.pad(ele.label_ids, (0, max_seq_length - len(ele.label_ids)), value=-100)
                        for ele in processed_batch
                    ]
                )
            attention_mask = torch.stack(
                [
                    F.pad(torch.ones_like(ele.input_ids), (0, max_seq_length - len(ele.input_ids)), value=0)
                    for ele in processed_batch
                ]
            )

        if not self.return_audio_in_tokens:
            audio_in_ids = None
            audio_in_ids_start = None

        # Apply audio_num_codebooks limit if specified
        if self.audio_num_codebooks is not None:
            if audio_in_ids is not None:
                audio_in_ids = audio_in_ids[: self.audio_num_codebooks]
            if audio_out_ids is not None:
                audio_out_ids = audio_out_ids[: self.audio_num_codebooks]
            if label_audio_ids is not None:
                label_audio_ids = label_audio_ids[: self.audio_num_codebooks]

        return HiggsAudioBatchInput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_features=audio_features,
            audio_feature_attention_mask=audio_feature_attention_mask,
            audio_out_ids=audio_out_ids,
            audio_out_ids_start=audio_out_ids_start,
            audio_out_ids_start_group_loc=audio_out_ids_start_group_loc,
            audio_in_ids=audio_in_ids,
            audio_in_ids_start=audio_in_ids_start,
            label_ids=label_ids,
            label_audio_ids=label_audio_ids,
            reward=reward,
        )
