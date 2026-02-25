Replace linear layers with QuantizedLinear; add InputQuantizedWrapper

quantize.py:
- QuantizedLinear: Standalone replacement for nn.Linear (not wrapper). Owns weight/bias
  copied from original; per-channel weight and per-token activation fake quantization
- quantize_model(model, replacement_list, input_quantize_list=None): New API. replacement_list
  specifies (source_type, replacement_cls) or (source_type, replacement_cls, kwargs).
  input_quantize_list specifies layers whose inputs get quantized via InputQuantizedWrapper
- InputQuantizedWrapper: Wraps any module, quantizes input before passing through.
  Per-token symmetric fake quantization; skip recursion to avoid double-wrapping
- find_quantized_layers(): Parameter renamed to replacement_cls; works with QuantizedLinear

main.py:
- quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits})])
- Import QuantizedLinear instead of QuantizedLayerWrapper

Other:
- description.md: Code description for quantize.py and main.py

Switch from torchvision to timm for ViT models

torchvision MultiheadAttention uses NonDynamicallyQuantizableLinear (not nn.Linear),
so attention layers were not quantized. timm ViT uses nn.Linear throughout:
- quantize.py: load_pretrained_vit() now uses timm.create_model(model_name, pretrained=True)
- main.py: timm for model loading; timm.data for transforms (resolve_data_config, create_transform)
- Added --model argparse option for timm model name (default: vit_base_patch16_224)
- requirements.txt: add timm>=0.9.0

Result: 49 quantized layers (vs 25) including attn.qkv, attn.proj, mlp.fc1/fc2, head.

Add activation quantization with online per-token scales

QuantizedLayerWrapper now quantizes both weights and activations:
- Activation scale computed online at runtime from current input (no calibration)
- Per-token scale: one scale per (batch, token) position over feature dimension
- Handles 2D (B, D) and 3D (B, N, D) inputs via reshape
- Symmetric fake quantization for activations, same scheme as weights

Add transformer quantization with ViT-B/16 and ImageNet evaluation

Quantization module (quantize.py):
- load_pretrained_vit(): Load ViT-B/16 with ImageNet-1K weights from torchvision
- QuantizedLayerWrapper: Default wrapper for nn.Linear with per-channel symmetric
  fake quantization; configurable bits (default 8); fixes device handling so
  buffers (scale, maxq) move correctly when model runs on GPU
- quantize_model(): Recursively search module tree for layers matching
  layer_types, wrap each with user-supplied wrapper_cls; handles nn.Sequential
  and nn.ModuleList; follows SpinQuant add_actquant pattern
- find_quantized_layers(): Return dict of wrapped modules by full name

Evaluation script (example_4bit.py):
- Quantize ViT-B/16 and evaluate on ImageNet validation set
- argparse --bits: User-defined quantization bits (default: 4)
- tqdm progress bar during evaluation loop
- Top-1 and top-5 accuracy on /data/shared_data/imagenet/val
- Uses ViT_B_16_Weights.IMAGENET1K_V1.transforms() for preprocessing

Other:
- PLAN.md: Plan document for quantization design
- requirements.txt: torch and torchvision dependencies
