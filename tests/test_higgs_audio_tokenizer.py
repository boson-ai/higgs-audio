import os
import json
import inspect
import tempfile
import pytest
from unittest.mock import MagicMock, patch
import torch

from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
    HiggsAudioTokenizer,
    _normalize_tokenizer_config,
    load_higgs_audio_tokenizer,
    DEFAULT_TOKENIZER_REPO,
    DEFAULT_TOKENIZER_REVISION,
)


def test_tokenizer_accepts_kwargs():
    """Verify HiggsAudioTokenizer.__init__ accepts unexpected kwargs without raising TypeError."""
    sig = inspect.signature(HiggsAudioTokenizer.__init__)
    assert any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_normalize_legacy_config():
    """Verify legacy flat config passes through intact."""
    legacy_config = {
        "n_filters": 32,
        "D": 256,
        "codebook_dim": 64,
        "target_bandwidths": [0.5, 1, 1.5, 2, 4],
        "ratios": [8, 5, 4, 2, 3],
        "sample_rate": 24000,
        "bins": 1024,
        "n_q": 8,
        "semantic_techer": "hubert_base_general",
    }
    normalized = _normalize_tokenizer_config(legacy_config)
    assert normalized["D"] == 256
    assert normalized["ratios"] == [8, 5, 4, 2, 3]
    assert normalized["bins"] == 1024
    assert normalized["n_q"] == 8
    assert normalized["semantic_techer"] == "hubert_base_general"


def test_normalize_nested_v2_config():
    """Verify newer HF nested schema (acoustic_model_config, semantic_model_config) is mapped correctly."""
    nested_config = {
        "acoustic_model_config": {
            "hidden_size": 256,
            "downsampling_ratios": [8, 5, 4, 2, 3],
            "codebook_size": 1024,
            "n_codebooks": 8,
            "codebook_dim": 64,
        },
        "semantic_model_config": {
            "model_type": "hubert",
            "semantic_sample_rate": 16000,
        },
        "codebook_dim": 64,
        "sample_rate": 24000,
        "target_bandwidths": [0.5, 1, 1.5, 2],
        "architectures": ["HiggsAudioV2TokenizerModel"],
        "transformers_version": "5.3.0.dev0",
    }
    normalized = _normalize_tokenizer_config(nested_config)
    assert normalized["D"] == 256
    assert normalized["ratios"] == [8, 5, 4, 2, 3]
    assert normalized["bins"] == 1024
    assert normalized["n_q"] == 8
    assert normalized["semantic_techer"] == "hubert_base_general"
    assert normalized["semantic_sample_rate"] == 16000


def test_load_tokenizer_pins_default_revision():
    """Verify load_higgs_audio_tokenizer pins to DEFAULT_TOKENIZER_REVISION for the default repo."""
    with (
        patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.snapshot_download") as mock_download,
        patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.open"),
        patch("json.load", return_value={"bins": 1024}),
        patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.HiggsAudioTokenizer") as mock_model_cls,
        patch("os.path.exists") as mock_exists,
        patch("torch.load"),
    ):
        mock_exists.side_effect = lambda p: False if p == DEFAULT_TOKENIZER_REPO else True
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance

        load_higgs_audio_tokenizer(DEFAULT_TOKENIZER_REPO, device="cpu")

        mock_download.assert_called_once_with(DEFAULT_TOKENIZER_REPO, revision=DEFAULT_TOKENIZER_REVISION)


def test_load_tokenizer_custom_revision():
    """Verify load_higgs_audio_tokenizer respects explicit custom revision."""
    with (
        patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.snapshot_download") as mock_download,
        patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.open"),
        patch("json.load", return_value={"bins": 1024}),
        patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.HiggsAudioTokenizer") as mock_model_cls,
        patch("os.path.exists") as mock_exists,
        patch("torch.load"),
    ):
        mock_exists.side_effect = lambda p: False if p == DEFAULT_TOKENIZER_REPO else True
        mock_instance = MagicMock()
        mock_model_cls.return_value = mock_instance

        load_higgs_audio_tokenizer(DEFAULT_TOKENIZER_REPO, device="cpu", revision="custom_hash_123")

        mock_download.assert_called_once_with(DEFAULT_TOKENIZER_REPO, revision="custom_hash_123")


def test_load_tokenizer_safetensors_fallback():
    """Verify load_higgs_audio_tokenizer loads model.safetensors when model.pth is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump({"bins": 1024, "acoustic_model_config": {"hidden_size": 128}}, f)

        safetensors_path = os.path.join(tmpdir, "model.safetensors")
        with open(safetensors_path, "wb") as f:
            f.write(b"")

        with (
            patch("boson_multimodal.audio_processing.higgs_audio_tokenizer.HiggsAudioTokenizer") as mock_model_cls,
            patch("safetensors.torch.load_file", return_value={}) as mock_load_file,
        ):
            mock_instance = MagicMock()
            mock_model_cls.return_value = mock_instance

            model = load_higgs_audio_tokenizer(tmpdir, device="cpu")
            assert mock_load_file.called
            assert mock_instance.load_state_dict.called
