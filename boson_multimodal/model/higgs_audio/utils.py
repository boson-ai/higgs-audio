import contextlib
from contextlib import contextmanager
from functools import wraps
import torch
from transformers.integrations import is_deepspeed_available

if is_deepspeed_available():
    from deepspeed.utils import groups as deepspeed_groups
    from deepspeed.sequence.layer import _SeqAllToAll
else:
    deepspeed_groups = None
    _SeqAllToAll = None


def _ceil_to_nearest(n, round_to):
    return (n + round_to - 1) // round_to * round_to


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def build_delay_pattern_mask(
    input_ids: torch.LongTensor,
    bos_token_id: int,
    pad_token_id: int,
):
    """Implement the delay pattern proposed in "Simple and Controllable Music Generation", https://arxiv.org/pdf/2306.05284

    In the delay pattern, each codebook is offset by the previous codebook by
    one. We insert a special delay token at the start of the sequence if its delayed, and append pad token once the sequence finishes.

    Take the example where there are 4 codebooks and audio sequence length=5. After shifting, the output should have length seq_len + num_codebooks - 1

    - [ *,  *,  *,  *,  *,  P,  P,  P]
    - [ B,  *,  *,  *,  *,  *,  P,  P]
    - [ B,  B,  *,  *,  *,  *,  *,  P]
    - [ B,  B,  B,  *,  *,  *,  *,  *]

    where B indicates the delay token id, P is the special padding token id and `*` indicates that the original audio token.

    Now let's consider the case where we have a sequence of audio tokens to condition on.
    The audio tokens were originally in the following non-delayed form:

    - [a, b]
    - [c, d]
    - [e, f]
    - [g, h]

    After conversion, we get the following delayed form:
    - [a, b, -1, -1, -1]
    - [B, c,  d, -1, -1]
    - [B, B,  e,  f, -1]
    - [B, B,  B,  g,  h]

    Note that we have a special token `-1` that indicates it should be replaced by a new token we see in the generation phase.
    In that case, we should override the `-1` tokens in auto-regressive generation.

    Args:
        input_ids (:obj:`torch.LongTensor`):
            The input ids of the prompt. It will have shape (bsz, num_codebooks, seq_len).
        bos_token_id (:obj:`int`):
            The id of the special delay token
        pad_token_id (:obj:`int`):
            The id of the padding token. Should be the same as eos_token_id.

    Returns:
        input_ids (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. It will have shape (bsz, num_codebooks, seq_len + num_codebooks - 1).
        input_ids_with_gen_mask (:obj:`torch.LongTensor`):
            The transformed input ids with delay pattern applied. The -1 in the output indicates new tokens that should be generated.

    """
    bsz, num_codebooks, seq_len = input_ids.shape

    new_seq_len = seq_len + num_codebooks - 1
    input_ids_with_gen_mask = torch.ones((bsz, num_codebooks, new_seq_len), dtype=torch.long, device=input_ids.device)
    bos_mask = torch.tril(input_ids_with_gen_mask, -1) > 0
    eos_mask = torch.triu(input_ids_with_gen_mask, seq_len) > 0
    input_ids_with_gen_mask[bos_mask] = bos_token_id
    input_ids_with_gen_mask[(~bos_mask) & (~eos_mask)] = input_ids.reshape(-1)
    input_ids = input_ids_with_gen_mask.clone()
    input_ids[eos_mask] = pad_token_id
    input_ids_with_gen_mask[eos_mask] = -1
    return input_ids, input_ids_with_gen_mask


def revert_delay_pattern(data):
    """Convert samples encoded with delay pattern back to the original form.

    Args:
        data (:obj:`torch.Tensor`):
            The data with delay pattern applied. It will have shape (num_codebooks, seq_len + num_codebooks - 1).

    Returns:
        ret (:obj:`torch.Tensor`):
            Recovered data with delay pattern removed. It will have shape (num_codebooks, seq_len).
    """
    assert len(data.shape) == 2
    out_l = []
    num_codebooks = data.shape[0]
    for i in range(num_codebooks):
        out_l.append(data[i : (i + 1), i : (data.shape[1] - num_codebooks + 1 + i)])
    return torch.cat(out_l, dim=0)


def merge_input_ids_with_audio_features(
    audio_features_embed,
    audio_features_length,
    audio_in_embed,
    audio_in_ids_start,
    audio_out_embed,
    audio_out_ids_start,
    audio_in_token_idx,
    audio_out_token_idx,
    inputs_embeds,
    input_ids,
    attention_mask,
    label_ids,
    pad_token_id,
    ignore_index=-100,
    round_to=8,
    left_padding=True,
):
    """
    Merge input_ids with audio features into final embeddings.

    Args:
        audio_features_embed (`torch.Tensor` of shape `(num_audios, max_audio_tokens, embed_dim)`):
            Encoded vectors of all audios in the batch (obtained from the semantic encoder)
        audio_features_length (`torch.LongTensor` of shape `(num_audios,)`):
            The length of audio embeddings of each audio as stacked in `audio_features_embed`
        audio_in_embed (`torch.Tensor` of shape `(total_num_audio_in_tokens, embed_dim)`):
            The embeddings of audio-in tokens
        audio_in_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-in tokens for each audio
        audio_out_embed (`torch.Tensor` of shape `(total_num_audio_out_tokens, embed_dim)`):
            The embeddings of audio-out tokens
        audio_out_ids_start (`torch.LongTensor` of shape `(num_audios,)`):
            The start index of the audio-out tokens for each audio
        audio_in_token_idx
            The index of the audio-in token in the vocabulary
        audio_out_token_idx
            The index of the audio-out token in the vocabulary
        inputs_embeds (`torch.Tensor` of shape `(batch_size, sequence_length, embed_dim)`):
            Token embeddings before merging with audio embeddings
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Input_ids of tokens, possibly filled with audio token
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices.
        label_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*)
            labels need to be recalculated to support training (if provided)
        pad_token_id (`int`):
            The index of the pad token in the vocabulary
        ignore_index
            The index to ignore in the loss calculation
        round_to
            The number to round to for padding
        left_padding
            Whether to apply left padding

    Returns:
        final_embedding
            The final embeddings after merging audio embeddings with text embeddings.
        final_attention_mask
            The final attention mask after merging audio embeddings with text embeddings.
        final_labels
            The labels for the text stream
        position_ids
            Positional ids for the merged data
        final_input_ids
            The final input_ids after merging audio embeddings with text embeddings.
        final_audio_in_mask
            Mask for audio-in embeddings
        final_audio_in_discrete_codes_mask
            Mask for audio-in discrete tokens
        final_audio_out_mask
            Mask for audio-out embeddings

    Explanation:
        each audio has variable length embeddings, with length specified by
        - audio_features_length
        - audio_in_ids_start
        - audio_out_ids_start

        Task:
        - fill each <|AUDIO|> with audio embeddings (it can be the combination of embeddings extracted by WhisperEncoder and embeddings from audio codebooks)
        - fill each <|AUDIO_OUT|> with the audio-out embeddings

        Example:
            <|AUDIO_OUT|>: X (5 tokens), Y (3 tokens)
            <|AUDIO|>: Z (8 tokens)

            X, Y are in the same sequence (in-context voice-clone). Z is in a different sequence (audio understanding).
        if right padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                o p q r Z s t u v _ _ _ _ _ _
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                o p q r Z Z Z Z Z Z Z Z s t u v _ _ _ _ _
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                o p q r _ _ _ _ _ _ _ _ s t u v _ _ _ _ _
            ]
        elif left padding
            input_ids: [
                a b c d e f X g h i j k Y l m
                _ _ _ _ _ _ o p q r Z s t u v
            ]
            input_ids should be: [
                a b c d e f X X X X X g h i j k Y Y Y l m
                _ _ _ _ _ o p q r Z Z Z Z Z Z Z Z s t u v
            ]
            labels should be: [
                a b c d e f _ _ _ _ _ g h i j k _ _ _ l m
                _ _ _ _ _ o p q r _ _ _ _ _ _ _ _ s t u v
            ]

    """
    if label_ids is None:
        skip_labels = True
    else:
        skip_labels = False
    if audio_features_embed is not None and audio_features_embed.shape[0] == 0:
        audio_features_embed = None
    if audio_in_embed is not None and audio_in_embed.shape[0] == 0:
        audio_in_embed = None
    if audio_out_embed is not None and audio_out_embed.shape[0] == 0:
        audio_out_embed = None

    batch_size, sequence_length, embed_dim = inputs_embeds.shape

    target_device = inputs_embeds.device
    if left_padding is None:
        left_padding = torch.any(attention_mask[:, 0] == 0)

    audio_in_token_mask = input_ids == audio_in_token_idx
    audio_out_token_mask = input_ids == audio_out_token_idx
    text_token_mask = (input_ids != audio_in_token_idx) & (input_ids != audio_out_token_idx)

    # 1. Calculate the number of tokens for each placeholder (like [<|AUDIO|>, <|AUDIO_OUT|>]).
    token_placeholder_num = torch.ones_like(input_ids)

    if audio_features_embed is not None:
        num_audios, max_audio_tokens, _ = audio_features_embed.shape
        audio_in_features_mask = torch.arange(max_audio_tokens).expand(num_audios, max_audio_tokens).to(
            audio_features_length.device
        ) < audio_features_length.unsqueeze(1)
        masked_audio_in_features = audio_features_embed[audio_in_features_mask].view(-1, embed_dim)
        token_placeholder_num[audio_in_token_mask] = audio_features_length.long()

    if audio_in_embed is not None:
        audio_in_codes_length = torch.concat(
            [
                audio_in_ids_start[1:] - audio_in_ids_start[:-1],
                torch.tensor(
                    [audio_in_embed.shape[0] - audio_in_ids_start[-1]],
                    device=audio_in_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        if audio_features_embed is not None:
            token_placeholder_num[audio_in_token_mask] += audio_in_codes_length.long()
        else:
            token_placeholder_num[audio_in_token_mask] = audio_in_codes_length.long()

    if audio_out_embed is not None:
        audio_out_codes_length = torch.concat(
            [
                audio_out_ids_start[1:] - audio_out_ids_start[:-1],
                torch.tensor(
                    [audio_out_embed.shape[0] - audio_out_ids_start[-1]],
                    device=audio_out_ids_start.device,
                    dtype=torch.long,
                ),
            ],
            dim=0,
        )
        token_placeholder_num[audio_out_token_mask] = audio_out_codes_length.long()

    new_token_positions = torch.cumsum(token_placeholder_num, -1) - 1
    max_token_num = _ceil_to_nearest(token_placeholder_num.sum(-1).max(), round_to)
    nb_audio_pad = max_token_num - 1 - new_token_positions[:, -1]

    if left_padding:
        new_token_positions += nb_audio_pad[:, None]  # offset for left padding

    # 2. Create the full embedding, already padded to the maximum position
    final_embedding = torch.zeros(
        (batch_size, max_token_num, embed_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
    )
    final_attention_mask = torch.zeros(
        (batch_size, max_token_num), dtype=attention_mask.dtype, device=inputs_embeds.device
    )
    final_input_ids = torch.full(
        (batch_size, max_token_num), pad_token_id, dtype=input_ids.dtype, device=inputs_embeds.device
    )
    if skip_labels:
        final_labels = None
    else:
        final_labels = torch.full(
            (batch_size, max_token_num), ignore_index, dtype=label_ids.dtype, device=inputs_embeds.device
        )

    final_audio_in_mask = torch.full((batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device)
    final_audio_in_discrete_codes_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    final_audio_out_mask = torch.full(
        (batch_size, max_token_num), False, dtype=torch.bool, device=inputs_embeds.device
    )
    # 3. Get the audio-in token positions and audio-out token positions
    batch_id = torch.arange(batch_size, device=target_device).unsqueeze(1).expand(batch_size, sequence_length)
    audio_in_batch_id = batch_id[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_batch_id = batch_id[audio_out_token_mask]  # Shape (num_audio_out,)
    audio_features_token_ends = new_token_positions[audio_in_token_mask]  # Shape (num_audio_in,)
    audio_out_embed_ends = new_token_positions[audio_out_token_mask]  # Shape (num_audio_out,)

    if audio_in_embed is not None:
        # Fill in the audio-in embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_in_ids_start.shape[0], max_token_num)
        )
        audio_in_embed_token_starts = audio_features_token_ends - audio_in_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_in_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_features_token_ends.unsqueeze(1))
        )
        batch_indices = audio_in_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_in_embed
        final_input_ids[batch_indices, col_indices] = audio_in_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_in_mask[batch_indices, col_indices] = True
        final_audio_in_discrete_codes_mask[batch_indices, col_indices] = True
        audio_features_token_ends = audio_features_token_ends - audio_in_codes_length

    if audio_features_embed is not None:
        # Fill in the audio features
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_features_embed.shape[0], max_token_num)
        )
        audio_features_token_starts = audio_features_token_ends - audio_features_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_features_token_starts.unsqueeze(1))
            & (seq_indices <= audio_features_token_ends.unsqueeze(1))
        )
        batch_indices = audio_in_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = masked_audio_in_features
        final_input_ids[batch_indices, col_indices] = audio_in_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_in_mask[batch_indices, col_indices] = True

    if audio_out_embed is not None:
        # Fill in the audio-out embeddings
        seq_indices = (
            torch.arange(max_token_num, device=target_device)
            .unsqueeze(0)
            .expand(audio_out_ids_start.shape[0], max_token_num)
        )
        audio_out_embed_token_starts = audio_out_embed_ends - audio_out_codes_length + 1
        batch_indices, col_indices = torch.where(
            (seq_indices >= audio_out_embed_token_starts.unsqueeze(1))
            & (seq_indices <= audio_out_embed_ends.unsqueeze(1))
        )
        batch_indices = audio_out_batch_id[batch_indices]
        final_embedding[batch_indices, col_indices] = audio_out_embed
        final_input_ids[batch_indices, col_indices] = audio_out_token_idx
        if not skip_labels:
            final_labels[batch_indices, col_indices] = ignore_index
        final_audio_out_mask[batch_indices, col_indices] = True

    # Fill in the original text embeddings and labels
    batch_indices, non_audio_indices = torch.where(text_token_mask)
    text_to_overwrite = new_token_positions[batch_indices, non_audio_indices]
    final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_audio_indices]
    if not skip_labels:
        final_labels[batch_indices, text_to_overwrite] = label_ids[batch_indices, non_audio_indices]
    final_input_ids[batch_indices, text_to_overwrite] = input_ids[batch_indices, non_audio_indices]
    final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_audio_indices]
    final_attention_mask = final_attention_mask | final_audio_in_mask | final_audio_out_mask

    # Trim the tensor if there are redundant padding tokens
    if left_padding:
        first_non_zero_loc = final_attention_mask.sum(0).nonzero()[0]
        first_non_zero_loc = (first_non_zero_loc // round_to) * round_to
        if first_non_zero_loc > 0:
            final_attention_mask = final_attention_mask[:, first_non_zero_loc:]
            final_embedding = final_embedding[:, first_non_zero_loc:]
            if not skip_labels:
                final_labels = final_labels[:, first_non_zero_loc:]
            final_input_ids = final_input_ids[:, first_non_zero_loc:]
            final_audio_in_mask = final_audio_in_mask[:, first_non_zero_loc:]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, first_non_zero_loc:]
            final_audio_out_mask = final_audio_out_mask[:, first_non_zero_loc:]
    else:
        # We have done right padding, so we need to trim the mask
        last_non_zero_loc = final_attention_mask.sum(0).nonzero()[-1] + 1
        last_non_zero_loc = ((last_non_zero_loc + round_to - 1) // round_to) * round_to
        if last_non_zero_loc < max_token_num:
            final_attention_mask = final_attention_mask[:, :last_non_zero_loc]
            final_embedding = final_embedding[:, :last_non_zero_loc]
            if not skip_labels:
                final_labels = final_labels[:, :last_non_zero_loc]
            final_input_ids = final_input_ids[:, :last_non_zero_loc]
            final_audio_in_mask = final_audio_in_mask[:, :last_non_zero_loc]
            final_audio_in_discrete_codes_mask = final_audio_in_discrete_codes_mask[:, :last_non_zero_loc]
            final_audio_out_mask = final_audio_out_mask[:, :last_non_zero_loc]

    position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
    return (
        final_embedding,
        final_attention_mask,
        final_labels,
        position_ids,
        final_input_ids,
        final_audio_in_mask,
        final_audio_in_discrete_codes_mask,
        final_audio_out_mask,
    )


def is_deepspeed_ulysses_enabled():
    if deepspeed_groups is None:
        return False

    """Check if sequence parallelism is enabled."""
    return deepspeed_groups._get_sequence_parallel_world_size() > 1


def support_deepspeed_ulysses(module):
    """A decorator around Pytorch module. It is needed for the module that needs access to sequence parallel info."""
    module._sp_size = None
    module._sp_rank = None
    module._sp_group = None

    @property
    def sp_size(self):
        if self._sp_size is None:
            self._sp_size = 1
            if is_deepspeed_ulysses_enabled():
                self._sp_size = deepspeed_groups._get_sequence_parallel_group().size()
        return self._sp_size

    @property
    def sp_rank(self):
        if self._sp_rank is None:
            self._sp_rank = 0
            if is_deepspeed_ulysses_enabled():
                self._sp_rank = deepspeed_groups._get_sequence_parallel_rank()
        return self._sp_rank

    @property
    def sp_group(self):
        if self._sp_group is None and is_deepspeed_ulysses_enabled():
            self._sp_group = deepspeed_groups._get_sequence_parallel_group()
        return self._sp_group

    module.sp_size = sp_size
    module.sp_rank = sp_rank
    module.sp_group = sp_group

    return module


def deepspeed_ulysses_attention(seq_dim=1, head_dim=2):
    """Perform all-to-all before and after the attention function."""

    def attention_decorator(attn_func=None):
        def wrapped(*args, **kwargs):
            if is_deepspeed_ulysses_enabled():
                sp_group = deepspeed_groups._get_sequence_parallel_group()
                scatter_idx = head_dim  # Scatter on num_heads dimension
                gather_idx = seq_dim  # Gather on seq_len dimension
                batch_dim_idx = 0
                args = list(args)
                args[0] = _SeqAllToAll.apply(sp_group, args[0], scatter_idx, gather_idx, batch_dim_idx)
                args[1] = _SeqAllToAll.apply(sp_group, args[1], scatter_idx, gather_idx, batch_dim_idx)
                args[2] = _SeqAllToAll.apply(sp_group, args[2], scatter_idx, gather_idx, batch_dim_idx)
                args = tuple(args)

            attn_output = attn_func(*args, **kwargs)

            if is_deepspeed_ulysses_enabled():
                scatter_idx = seq_dim  # Scatter back on seq_len dimension
                gather_idx = head_dim  # Gather on num_heads dimension
                batch_dim_idx = 0
                attn_output = _SeqAllToAll.apply(sp_group, attn_output, scatter_idx, gather_idx, batch_dim_idx)

            return attn_output

        return wrapped

    return attention_decorator


def deepspeed_ulysses_rope(state_seq_dim=2, trig_seq_dim=1):
    """Slice the corresponding cos and sin chunks for rope."""

    def rope_decorator(rope_func=None):
        def wrapped(*args, **kwargs):
            if is_deepspeed_ulysses_enabled():
                sp_rank = deepspeed_groups._get_sequence_parallel_rank()
                args = list(args)
                seq_chunk_size = args[0].size(state_seq_dim)
                args[2] = torch.narrow(args[2], trig_seq_dim, sp_rank * seq_chunk_size, seq_chunk_size)
                args[3] = torch.narrow(args[3], trig_seq_dim, sp_rank * seq_chunk_size, seq_chunk_size)
                args = tuple(args)

            return rope_func(*args, **kwargs)

        return wrapped

    return rope_decorator


def _gather_tensors(input_, group=None):
    """Gather tensors and concatenate them along a dimension."""
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_
    tensor_shapes = [
        torch.empty(len(input_.size()), dtype=torch.int64, device=input_.device) for _ in range(world_size)
    ]
    input_size = torch.tensor(input_.size(), dtype=torch.int64, device=input_.device)
    torch.distributed.all_gather(tensor_shapes, input_size, group=group)
    gathered_buffers = [
        torch.empty(tensor_shapes[i].tolist(), dtype=input_.dtype, device=input_.device) for i in range(world_size)
    ]
    torch.distributed.all_gather(gathered_buffers, input_, group=group)
    return gathered_buffers


def _scatter_tensors(input_, group=None):
    """Scatter tensors."""
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_
    rank = torch.distributed.get_rank(group)
    return input_[rank]


class _GatherTensors(torch.autograd.Function):
    """All gather tensors among the ranks."""

    @staticmethod
    def symbolic(graph, input_, group):
        return _gather_tensors(input_, group)

    @staticmethod
    def forward(ctx, input_, group):
        ctx.group = group
        return torch.nested.as_nested_tensor(_gather_tensors(input_, group), layout=torch.jagged)

    @staticmethod
    def backward(ctx, grad_output):
        return _scatter_tensors(grad_output, ctx.group), None


def all_gather_tensors(input_, size=None, dim=0, group=None):
    if torch.distributed.get_world_size(group) == 1:
        # no sequence parallelism
        return input_
    gathered_tensors = _GatherTensors.apply(input_, group)

    if size:
        split_gathered_tensors = []
        for s, gathered_tensor in zip(size, gathered_tensors):
            split_gathered_tensor = torch.split(gathered_tensor, s.tolist())
            split_gathered_tensors.append(split_gathered_tensor)

        gathered_tensors = [y for x in zip(*split_gathered_tensors) for y in x]

    return torch.cat(gathered_tensors, dim).contiguous()


def get_sequence_data_parallel_world_size():
    return torch.distributed.get_world_size()


def get_sequence_data_parallel_rank():
    return torch.distributed.get_rank()


def get_sequence_data_parallel_group():
    return torch.distributed.group.WORLD


if is_deepspeed_available():
    deepspeed_groups._get_sequence_data_parallel_world_size = get_sequence_data_parallel_world_size
    deepspeed_groups._get_sequence_data_parallel_rank = get_sequence_data_parallel_rank
    deepspeed_groups._get_sequence_data_parallel_group = get_sequence_data_parallel_group


def _gather_tokens(input_, dim=0, group=None):
    """Gather tensors and concatenate them along a dimension"""
    input_ = input_.contiguous()
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_

    gather_buffer = torch.empty(world_size * input_.numel(), dtype=input_.dtype, device=input_.device)
    torch.distributed.all_gather_into_tensor(gather_buffer, input_, group=group)
    if dim == 0:
        shape = list(input_.size())
        shape[0] = shape[0] * world_size
        output = gather_buffer.view(shape)
    else:
        tensor_list = [
            gather_buffer.narrow(0, input_.numel() * i, input_.numel()).view_as(input_) for i in range(world_size)
        ]
        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


def _drop_tokens(input_, dim=0, group=None):
    """Divide a tensor among the sequence parallel ranks"""
    world_size = torch.distributed.get_world_size(group)
    if world_size == 1:
        return input_
    this_rank = torch.distributed.get_rank(group)
    assert input_.shape[dim] % world_size == 0, (
        f"input dimension {dim} ({input_.shape[dim]}) is not divisible by sequence parallel world size ({world_size})"
    )
    chunk_size = input_.shape[dim] // world_size

    return torch.narrow(input_, dim, this_rank * chunk_size, chunk_size)


class _DropTokens(torch.autograd.Function):
    "Divide tokens equally among the sequence parallel ranks"

    @staticmethod
    def symbolic(graph, input_, dim, group, grad_scale):
        return _drop_tokens(input_, dim, group)

    @staticmethod
    def forward(ctx, input_, dim, group, grad_scale):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _drop_tokens(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _gather_tokens(grad_output, ctx.dim, ctx.group)
        if ctx.grad_scale != 1:
            grad_input /= ctx.grad_scale
        return grad_input, None, None, None


class _GatherTokens(torch.autograd.Function):
    "Gather tokens among the sequence parallel ranks"

    @staticmethod
    def symbolic(graph, input_, dim, group, grad_scale):
        return _gather_tokens(input_, dim, group)

    @staticmethod
    def forward(ctx, input_, dim, group, grad_scale):
        ctx.dim = dim
        ctx.group = group
        ctx.grad_scale = grad_scale
        return _gather_tokens(input_, dim, group)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = _drop_tokens(grad_output, ctx.dim, ctx.group)
        if ctx.grad_scale != 1:
            grad_input *= ctx.grad_scale
        return grad_input, None, None, None


def drop_tokens(input_, dim=0, group=None, grad_scale=1):
    if torch.distributed.get_world_size(group) == 1:
        # no sequence parallelism
        return input_
    return _DropTokens.apply(input_, dim, group, grad_scale)


def gather_tokens(input_, dim=0, group=None, grad_scale=1):
    if torch.distributed.get_world_size(group) == 1:
        # no sequence parallelism
        return input_
    return _GatherTokens.apply(input_, dim, group, grad_scale)


def sequence_chunking_per_rank(sp_size, sp_rank, *args, dim=1):
    """
    Slice the inputs to create chuncks per the sequence parallel rank. This is used for the context parallel training.

    Args:
        sp_size (`int`):
            Sequence parallel size.
        sp_rank (`int`):
            Sequence parallel rank for the current process.
        dim (`int`):
           The dimension to slice
    """
    if sp_size == 1:
        return args[0] if len(args) == 1 else args

    seq_length = args[0].size(dim)
    for arg in args[1:]:
        assert arg.size(dim) == seq_length, (
            f"arg={arg} ({arg.shape[dim]}) does not have the same size as args[0] ({seq_length}) in dimension {dim}"
        )
    assert seq_length % sp_size == 0, (
        f"dimension {dim} ({args[0].shape[dim]}) is not divisible by sequence parallel world size ({sp_size})"
    )

    sub_seq_length = seq_length // sp_size
    sub_seq_start = sp_rank * sub_seq_length

    output = []
    for ind in args:
        ind = torch.narrow(ind, dim, sub_seq_start, sub_seq_length)
        output.append(ind)

    return tuple(output) if len(output) > 1 else output[0]


@contextmanager
def disable_deepspeed_ulysses():
    """Disable deepspeed ulysses (sequence parallelism) if it is enabled"""
    if is_deepspeed_ulysses_enabled():
        _old_get_sequence_parallel_world_size = deepspeed_groups._get_sequence_parallel_world_size

        def _get_sequence_parallel_world_size():
            return 1

        deepspeed_groups._get_sequence_parallel_world_size = _get_sequence_parallel_world_size
        try:
            yield
        finally:
            deepspeed_groups._get_sequence_parallel_world_size = _old_get_sequence_parallel_world_size
    else:
        context = contextlib.nullcontext
        with context():
            yield
