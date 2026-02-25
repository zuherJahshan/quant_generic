# Code Description

## `quantize.py` — Quantization Utilities

This module provides utilities for loading pre-trained Vision Transformers from timm and applying fake quantization to transformer models. It supports both **replacing** layers (e.g., `nn.Linear` → `QuantizedLinear`) and **wrapping** layers to quantize their inputs.

---

### `load_pretrained_vit(model_name, pretrained)`

Loads a pre-trained Vision Transformer from the timm library. Default model is ViT-B/16 (`vit_base_patch16_224`). Returns the model instance.

---

### `QuantizedLinear` (lines 19–70)

A **standalone replacement** for `nn.Linear` (not a wrapper). It owns its own weight and bias parameters, copied from the original layer at init. Performs per-output-channel symmetric fake quantization on weights and per-token symmetric fake quantization on activations with online scale computation.

**`__init__(self, original, bits=8)`**

- Accepts an `nn.Linear` and copies `in_features`, `out_features`, `weight`, and `bias` into its own parameters.
- Stores `bits` and registers a `maxq` buffer (`2^(bits-1) - 1`).
- Calls `_compute_scales()` to compute per-output-channel symmetric quantization scales from the weight.

**`_compute_scales()`**

- Computes per-output-channel symmetric scales from the weight.
- Reshapes weight to `(out_features, -1)`, takes per-channel max absolute value, divides by `maxq`, and registers as a buffer.

**`_fake_quant_weight()`**

- Applies fake quantization to the weight (quantize then dequantize).
- Returns `scale * clamp(round(w/scale), -maxq-1, maxq)`.

**`forward(x)`**

1. **Per-token activation scale**: Reshapes input to `(-1, D)`, computes per-row max absolute value, derives scale as `max/ maxq`.
2. **Fake-quantize activation**: `round(x/scale) * scale`.
3. **Fake-quantize weight** and compute `F.linear(x_q, w_q, bias)`.

---

### `InputQuantizedWrapper` (lines 73–99)

A **wrapper** that quantizes the input before passing it to the wrapped module. Uses the same per-token symmetric fake quantization scheme as `QuantizedLinear` activations.

**`__init__(self, module, bits=8)`**

- Stores the wrapped module and `bits`, registers `maxq` buffer.

**`_quantize_input(x)`**

- Per-token symmetric fake quantization of the input.
- Reshapes to `(-1, x.shape[-1])`, computes per-row scale, quantizes, and returns dequantized tensor.

**`forward(x)`**

- Quantizes input via `_quantize_input(x)` and passes to `self.module(x_q)`.

---

### `quantize_model(model, replacement_list, input_quantize_list=None, name="")` (lines 101–175)

Recursively walks the module tree and (1) **replaces** layers matching `replacement_list`, and (2) optionally **wraps** layers matching `input_quantize_list` with `InputQuantizedWrapper` so their inputs are quantized.

**Arguments**

- `model`: Root module to process.
- `replacement_list`: List of `(source_type, replacement_cls)` or `(source_type, replacement_cls, kwargs)`. Each tuple specifies which layer type to replace and with what class.
- `input_quantize_list`: Optional list of `(layer_type,)` or `(layer_type, kwargs)`. Layers matching these types get wrapped with `InputQuantizedWrapper`. Applied after replacement.
- `name`: Internal use for recursion.

**Logic**

1. Early-exit if the current module is already a replacement class or `InputQuantizedWrapper` (avoids double-wrapping).
2. Build lookup dicts from `replacement_list` and `input_quantize_list`.
3. Define `process_module(tmp)`: if `tmp` matches a replacement rule, replace it; if the result matches an input-quantize rule, wrap with `InputQuantizedWrapper`.
4. Iterate over attributes via `dir(model)`; for each attribute, if its type matches a rule, call `process_module` and `setattr`. Special handling for `nn.Sequential` and `nn.ModuleList` (process each child).
5. Recurse into children via `model.named_children()`.

---

### `find_quantized_layers(model, replacement_cls, name="")` (lines 178–192)

Recursively finds all modules that are instances of `replacement_cls` (e.g., `QuantizedLinear`). Returns a dict mapping full module name (e.g., `blocks.0.attn.qkv`) to the module.

---

### `__main__` block (lines 195–219)

When run as a script: loads ViT-B/16, prints architecture before/after quantization, applies `quantize_model` with `[(nn.Linear, QuantizedLinear, {"bits": 8})]`, prints count of quantized layers, and runs a dummy forward pass to verify.

---

## `main.py` — Evaluation Script

A CLI script that quantizes a ViT and evaluates it on the ImageNet validation set.

---

### Constants

- `BATCH_SIZE = 64`, `NUM_WORKERS = 4`.

---

### `evaluate(model, dataloader, device)`

Computes top-1 and top-5 accuracy. Iterates over the dataloader, runs the model in eval mode, and returns `(top1_acc, top5_acc)` as percentages.

---

### `main()`

1. **Argument parsing**: `imagenet_dir` (required), `--model` (timm model name, default `vit_base_patch16_224`), `--bits` (quantization bits, default 8), `--no-quantize` (skip quantization for baseline).

2. **Model loading**: Loads pre-trained model via `load_pretrained_vit`, moves to device.

3. **Quantization**: Unless `--no-quantize`, calls `quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits})])` and prints count of quantized layers.

4. **Data loading**: Uses timm's data config and transform, creates `ImageFolder` dataloader for the validation directory.

5. **Evaluation**: Runs `evaluate()` and prints top-1 and top-5 accuracy.
