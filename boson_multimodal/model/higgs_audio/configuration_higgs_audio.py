from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto import CONFIG_MAPPING


class HiggsAudioEncoderConfig(PretrainedConfig):
    """Configuration of the Audio encoder in Higgs-Audio."""

    model_type = "higgs_audio_encoder"

    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        encoder_layerdrop=0.0,
        d_model=1280,
        dropout=0.0,
        attention_dropout=0.0,
        activation_function="gelu",
        activation_dropout=0.0,
        scale_embedding=False,
        init_std=0.02,
        max_source_positions=1500,
        pad_token_id=128001,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.num_hidden_layers = encoder_layers
        self.init_std = init_std
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        self.max_source_positions = max_source_positions
        self.pad_token_id = pad_token_id


class HiggsAudioConfig(PretrainedConfig):
    r"""
    This is the configuration class for the HiggsAudioModel.

    Args:
        text_config (`Union[AutoConfig, dict]`):
            The config object or dictionary of the text backbone.
        audio_encoder_config (`Union[AutoConfig, dict]`):
            The config object or dictionary of the whisper encoder.
            The audio encoder will be bidirectional and will be only available for audio understanding.
        audio_tokenizer_config
            The config object or dictionary of the audio tokenizer.
        audio_adapter_type
            The type of audio adapter to use. We support two types of adapter:
            - stack:
                We stack additional Transformer layers after the main LLM backbone for audio generation.
            - dual_ffn:
                For selected part of the LLM backbone, we replace the text FFN with a dual FFN architecture
                that contains an additional audio FFN. The audio FFN will be triggered when the location is marked for audio tokens.
            - dual_ffn_fast_forward:
                We pick a few layers in the LLM backbone to plug-in the audio FFN. For the remaining layers,
                the audio hidden states will be directly fast-forward to the next layer.
                This reduces the computational cost for audio generation.
        audio_embed_avg (`bool`, *optional*, defaults to False):
            Whether to average the audio embeddings before sending them to the text attention layer.
        audio_ffn_hidden_size
            The hidden size of the audio feedforward network in dual-path FFN
        audio_ffn_intermediate_size
            The intermediate size of the audio feedforward network in dual-path FFN
        audio_dual_ffn_layers
            The layers in the LLM backbone to plug-in the dual FFN layer (mixture of audio FFN and text FFN).
        audio_decoder_proj_num_attention (`int`, *optional*, defaults to 0):
            The number of attention heads in the audio decoder projection layer.
        use_delay_pattern (`bool`, *optional*, defaults to False):
            Whether to use delay pattern in the audio decoder.
        skip_audio_tower (`bool`, *optional*, defaults to False):
            Whether to skip the audio tower in the audio encoder.
        use_audio_out_embed_projector (`bool`, *optional*, defaults to False):
            Whether to use an embedding projector to map audio out embeddings.
        use_audio_out_self_attention (`bool`, *optional*, defaults to False):
            Whether to use self-attention to aggregate information from audio-tokens before sending to the text attention layer.
        audio_num_codebooks (`int`, *optional*, defaults to 12):
            The number of codebooks in RVQGAN.
        audio_codebook_size (`int`, *optional*, defaults to 1024):
            The size of each codebook in RVQGAN.
        audio_stream_bos_id
            The id of the bos in the audio stream
        audio_stream_eos_id
            The id of the eos in the audio stream
        audio_bos_token (`str`, *optional*, defaults to "<|audio_bos|>"):
            The special `<|audio_bos|>` token. In Higgs-Audio, it is mapped to 128011,
            which is the index of `<|reserved_special_token_3|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_eos_token (`str`, *optional*, defaults to "<|audio_eos|>"):
            The special `<|audio_eos|>` token. We use 128012 as the default value,
            which is the index of `<|reserved_special_token_4|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_out_bos_token (`str`, *optional*, defaults to "<|audio_out_bos|>"):
            The special `<|audio_out_bos|>` token. We use 128013 as the default value,
            which is the index of `<|reserved_special_token_5|>` in Llama-3.1-8B-Instruct's tokenizer.
        audio_token (`str`, *optional*, defaults to "<|AUDIO|>"):
            The special `<|AUDIO|>` token. We use 128015 as the default value,
            which is the index of `<|reserved_special_token_7|>` in Llama-3.1-8B-Instruct's tokenizer.
            This token indicates that the location should be filled in with whisper features.
        audio_out_token (`str`, *optional*, defaults to "<|AUDIO_OUT|>"):
            The special `<|AUDIO_OUT|>` token. We use 128016 as the default value,
            which is the index of `<|reserved_special_token_8|>` in Llama-3.1-8B-Instruct's tokenizer.
            This token indicates that the location should be filled in with audio tokens extracted via audio tokenizer.
    """

    model_type = "higgs_audio"
    is_composition = True

    def __init__(
        self,
        text_config=None,
        audio_encoder_config=None,
        audio_tokenizer_config=None,
        audio_adapter_type="stack",
        audio_embed_avg=False,
        audio_ffn_hidden_size=4096,
        audio_ffn_intermediate_size=14336,
        audio_dual_ffn_layers=None,
        audio_decoder_proj_num_layers=0,
        encode_whisper_embed=True,
        encode_audio_in_tokens=False,
        use_delay_pattern=False,
        skip_audio_tower=False,
        use_audio_out_embed_projector=False,
        use_audio_out_self_attention=False,
        use_rq_transformer=False,
        rq_transformer_hidden_size=None,
        rq_transformer_intermediate_size=None,
        rq_transformer_num_attention_heads=None,
        rq_transformer_num_key_value_heads=None,
        rq_transformer_num_hidden_layers=3,
        audio_num_codebooks=12,
        audio_codebook_size=1024,
        audio_stream_bos_id=1024,
        audio_stream_eos_id=1025,
        audio_bos_token="<|audio_bos|>",
        audio_eos_token="<|audio_eos|>",
        audio_out_bos_token="<|audio_out_bos|>",
        audio_in_token="<|AUDIO|>",
        audio_out_token="<|AUDIO_OUT|>",
        audio_in_token_idx=128015,
        audio_out_token_idx=128016,
        pad_token_id=128001,
        audio_out_bos_token_id=128013,
        audio_eos_token_id=128012,
        **kwargs,
    ):
        if isinstance(audio_encoder_config, dict):
            audio_encoder_config["model_type"] = (
                audio_encoder_config["model_type"] if "model_type" in audio_encoder_config else "higgs_audio_encoder"
            )
            audio_encoder_config = CONFIG_MAPPING[audio_encoder_config["model_type"]](**audio_encoder_config)
        elif audio_encoder_config is None:
            audio_encoder_config = HiggsAudioEncoderConfig()

        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()
        
        # Ensure _attn_implementation is set to avoid KeyError
        if not hasattr(text_config, '_attn_implementation') or text_config._attn_implementation is None:
            text_config._attn_implementation = "eager"

        assert audio_adapter_type in [
            "stack",
            "dual_ffn",
            "dual_ffn_fast_forward",
        ], f"Invalid audio adapter type: {audio_adapter_type}"
        if audio_adapter_type.startswith("dual_ffn"):
            assert audio_dual_ffn_layers is not None, (
                "audio_dual_ffn_layers must be specified when using dual_ffn adapter."
            )
        self.text_config = text_config
        self.audio_encoder_config = audio_encoder_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.audio_adapter_type = audio_adapter_type
        self.audio_embed_avg = audio_embed_avg
        self.audio_ffn_hidden_size = audio_ffn_hidden_size
        self.audio_ffn_intermediate_size = audio_ffn_intermediate_size
        self.audio_dual_ffn_layers = audio_dual_ffn_layers
        self.audio_decoder_proj_num_layers = audio_decoder_proj_num_layers
        self.encode_whisper_embed = encode_whisper_embed
        self.encode_audio_in_tokens = encode_audio_in_tokens
        self.use_delay_pattern = use_delay_pattern
        self.skip_audio_tower = skip_audio_tower
        self.use_audio_out_embed_projector = use_audio_out_embed_projector
        self.use_audio_out_self_attention = use_audio_out_self_attention

        self.use_rq_transformer = use_rq_transformer

        if self.use_rq_transformer:
            assert not self.use_delay_pattern, "Delay pattern is not supported if you turned on RQ-Transformer!"
        self.rq_transformer_hidden_size = rq_transformer_hidden_size
        self.rq_transformer_intermediate_size = rq_transformer_intermediate_size
        self.rq_transformer_num_attention_heads = rq_transformer_num_attention_heads
        self.rq_transformer_num_key_value_heads = rq_transformer_num_key_value_heads
        self.rq_transformer_num_hidden_layers = rq_transformer_num_hidden_layers

        if use_rq_transformer:
            # For RQ-Transformer, we set the hidden_size to the same as the text model's hidden size if it is not specified.
            if self.rq_transformer_hidden_size is None:
                self.rq_transformer_hidden_size = text_config.hidden_size
            assert self.rq_transformer_hidden_size % 128 == 0
            if self.rq_transformer_intermediate_size is None:
                self.rq_transformer_intermediate_size = text_config.intermediate_size
            if self.rq_transformer_num_attention_heads is None:
                self.rq_transformer_num_attention_heads = self.rq_transformer_hidden_size // 128
            if self.rq_transformer_num_key_value_heads is None:
                self.rq_transformer_num_key_value_heads = self.rq_transformer_hidden_size // 128 // 4
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_attention_heads == 0
            assert self.rq_transformer_hidden_size % self.rq_transformer_num_key_value_heads == 0

        self.audio_num_codebooks = audio_num_codebooks
        self.audio_codebook_size = audio_codebook_size
        self.audio_bos_token = audio_bos_token
        self.audio_eos_token = audio_eos_token
        self.audio_out_bos_token = audio_out_bos_token
        self.audio_in_token = audio_in_token
        self.audio_out_token = audio_out_token
        self.audio_in_token_idx = audio_in_token_idx
        self.audio_out_token_idx = audio_out_token_idx
        self.audio_stream_bos_id = audio_stream_bos_id
        self.audio_stream_eos_id = audio_stream_eos_id
        self.audio_out_bos_token_id = audio_out_bos_token_id
        self.audio_eos_token_id = audio_eos_token_id

        super().__init__(**kwargs)
        self.pad_token_id = pad_token_id
