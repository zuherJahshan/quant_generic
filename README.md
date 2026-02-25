# Transformer Quantization

Quantize Vision Transformer (ViT-B/16) and evaluate on ImageNet.

## Data

Download the ImageNet validation set from the [official ImageNet website](https://www.image-net.org/). The validation directory should follow the standard layout: `val/n01440764/`, `val/n01443537/`, etc., with one folder per class containing the validation images.

## Quick Start

```bash
pip install -r requirements.txt
```

Run evaluation (the ImageNet validation directory path is required):

```bash
# Baseline (no quantization)
python main.py /path/to/imagenet/val --no-quantize

# 4-bit quantization
python main.py /path/to/imagenet/val --bits 4

# 8-bit quantization
python main.py /path/to/imagenet/val --bits 8
```

## Usage

| Argument | Required | Description |
|----------|----------|-------------|
| `imagenet_dir` | Yes | Path to the ImageNet validation folder |
| `--model` | No | timm model name (default: vit_base_patch16_224) |
| `--bits` | No | Number of bits for quantization (default: 8) |
| `--no-quantize` | No | Skip quantization; evaluate full-precision baseline |

## Example

```bash
python main.py /data/shared_data/imagenet/val --bits 4
```
