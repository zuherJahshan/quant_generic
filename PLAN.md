# Quantize Transformer Models

## Overview

Create a Python module that (1) downloads a pre-trained Vision Transformer (ViT-B/16 from torchvision, ImageNet-1K weights), and (2) provides a generic recursive quantization function inspired by SpinQuant's `add_actquant`.

## Key Design (from SpinQuant)

SpinQuant's `add_actquant` in `utils/quant_utils.py` recurses over the module tree using `dir(module)` + `getattr`/`setattr` to find layers matching a list of types, then wraps each with `ActQuantWrapper`. It also handles `nn.Sequential` and `nn.ModuleList` children. Our implementation will follow the same pattern but be **generic** -- the wrapper class is passed as an argument rather than hardcoded.

## File Structure

A single file `/workspace/quantize.py` containing:

### 1. Pre-trained model loading

- Use `torchvision.models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)` -- no extra dependencies needed.

### 2. Default quantization wrapper: `QuantizedLayerWrapper(nn.Module)`

A simple example wrapper (users can supply their own):
- Stores the original module as `self.module`
- Computes per-channel min/max symmetric quantization scales on the weight at init time
- In `forward`, fake-quantizes the weight (quantize then dequantize) and calls the original module's linear operation with the fake-quantized weight
- Configurable `bits` parameter (default 8)

### 3. Core function: `quantize_model(model, layer_types, wrapper_cls, **wrapper_kwargs)`

Recursive function that walks the module tree (inspired by SpinQuant's `add_actquant`):
- `layer_types`: list of `nn.Module` subclasses to match (e.g., `[nn.Linear]`)
- `wrapper_cls`: any `nn.Module` class whose `__init__` accepts `(module, **kwargs)`
- `**wrapper_kwargs`: forwarded to the wrapper constructor (e.g., `bits=4`)
- Skips modules that are already an instance of `wrapper_cls` to avoid double-wrapping
- Handles `nn.Sequential` and `nn.ModuleList` by replacing children in-place

### 4. `find_quantized_layers(model, wrapper_cls)` utility

Returns a dict of `{name: module}` for all wrapped layers (useful for inspection).

### 5. `__main__` block

- Loads ViT-B/16
- Prints the model architecture before quantization
- Calls `quantize_model(model, [nn.Linear], QuantizedLayerWrapper, bits=8)`
- Prints the model architecture after quantization
- Runs a dummy forward pass to verify correctness
- Prints count of quantized layers

## Dependencies

Only `torch` and `torchvision` (already installed). A `requirements.txt` will pin them.
