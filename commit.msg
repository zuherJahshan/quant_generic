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
