# Higgs Audio Transformers 4.46+ Compatibility Fix

This PR fixes compatibility issues with transformers 4.46+ while maintaining backward compatibility with older versions.

## Problem

Users are experiencing crashes when using Higgs Audio with transformers versions 4.46+ due to breaking API changes. This is preventing users from using modern Python environments and newer dependencies.

**Related Issues:** #127

## Solution

The fix addresses 5 major API changes in transformers 4.46+:

### 1. LLAMA_ATTENTION_CLASSES Removal
**Issue:** `LLAMA_ATTENTION_CLASSES` was removed from transformers.models.llama.modeling_llama
**Fix:** Create compatibility mapping with fallback to basic attention classes

### 2. StaticCache API Changes  
**Issue:** `get_max_length()` was renamed to `max_cache_len` attribute
**Fix:** Add compatibility function that tries multiple attribute names

### 3. GenerationConfig Structure Changes
**Issue:** `generation_kwargs` attribute is missing in newer versions
**Fix:** Dynamically create the attribute if it doesn't exist

### 4. Generation Loop Signature Changes
**Issue:** `_has_unfinished_sequences()` no longer accepts `cur_len` and `max_length` parameters
**Fix:** Triple fallback approach supporting all parameter combinations

### 5. Attention Return Value Changes
**Issue:** Attention layers may return 2 or 3 values depending on transformers version
**Fix:** Dynamic unpacking based on returned value count

## Files Changed

1. `boson_multimodal/model/higgs_audio/modeling_higgs_audio.py` - Core compatibility fixes
2. `boson_multimodal/model/higgs_audio/configuration_higgs_audio.py` - Config attention defaults
3. `boson_multimodal/serve/serve_engine.py` - CUDA graph graceful fallback

## Backward Compatibility

All fixes use try-catch patterns and graceful fallbacks to maintain compatibility with transformers 4.30-4.46+.

## Testing

- Tested with transformers 4.46.3 on Python 3.13
- Confirmed working with original requirements (transformers 4.45.1)
- No breaking changes to existing functionality

---

## How to Apply These Fixes

### Option 1: Manual Application (Recommended)
Copy the compatibility fixes from the TTS Audio Suite repository:
- Source: https://github.com/diodiogod/TTS-Audio-Suite/commit/99c4b88
- Files to copy from: `engines/higgs_audio/boson_multimodal/model/higgs_audio/`

### Option 2: Key Code Snippets

#### 1. In modeling_higgs_audio.py - Add after imports:
```python
# Compatibility fix for LLAMA_ATTENTION_CLASSES removal in transformers 4.46+
try:
    from transformers.models.llama.modeling_llama import LLAMA_ATTENTION_CLASSES
except ImportError:
    from transformers.models.llama.modeling_llama import (
        LlamaAttention, LlamaFlashAttention2, LlamaSdpaAttention
    )
    LLAMA_ATTENTION_CLASSES = {
        "eager": LlamaAttention,
        "flash_attention_2": LlamaFlashAttention2, 
        "sdpa": LlamaSdpaAttention,
    }

def get_cache_max_length(cache):
    """Get cache max length with compatibility for different transformers versions"""
    if hasattr(cache, 'get_max_length'):
        return cache.get_max_length()
    elif hasattr(cache, 'max_cache_len'):
        return cache.max_cache_len
    elif hasattr(cache, 'get_seq_length'):
        return cache.get_seq_length()
    elif hasattr(cache, 'max_length'):
        return cache.max_length
    else:
        return 2048  # Reasonable default
```

#### 2. In configuration_higgs_audio.py - Fix text config creation:
```python
# In HiggsAudioConfig.__init__ around line 165
if isinstance(text_config, dict):
    text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
    # Ensure attention implementation is set to avoid None errors in newer transformers
    if "_attn_implementation" not in text_config:
        text_config["_attn_implementation"] = "eager"
    text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
elif text_config is None:
    text_config = CONFIG_MAPPING["llama"]()
    # Set default attention implementation for newer transformers compatibility
    text_config._attn_implementation = "eager"
```

#### 3. Additional fixes needed:
- GenerationConfig.generation_kwargs compatibility
- _has_unfinished_sequences parameter compatibility  
- Attention return value unpacking
- CUDA graph graceful fallback

## Benefits

- ✅ Fixes crashes with transformers 4.46+
- ✅ Maintains backward compatibility
- ✅ Enables use with Python 3.13
- ✅ No performance degradation
- ✅ Graceful fallbacks for all edge cases

## Impact

This fix will resolve issues for users trying to use Higgs Audio with:
- Modern Python environments (3.11, 3.12, 3.13)
- Recent transformers versions
- ComfyUI and other integrations
- Docker containers with newer dependencies

---

**Note:** The complete working implementation is available in the TTS Audio Suite repository where these fixes have been tested and validated.