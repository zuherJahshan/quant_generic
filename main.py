"""Dummy example: quantize ViT-B/16 and evaluate on ImageNet."""

import argparse

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from quantize import load_pretrained_vit, quantize_model, QuantizedLinear, find_quantized_layers

BATCH_SIZE = 64
NUM_WORKERS = 4


def evaluate(model, dataloader, device):
    """Compute top-1 and top-5 accuracy."""
    model.eval()
    correct1, correct5, total = 0, 0, 0
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            _, pred5 = outputs.topk(5, dim=1)
            _, pred1 = outputs.max(1)
            total += targets.size(0)
            correct1 += (pred1 == targets).sum().item()
            correct5 += (pred5 == targets.unsqueeze(1)).any(dim=1).sum().item()
            pbar.set_postfix(top1=f"{100.0 * correct1 / total:.2f}%", top5=f"{100.0 * correct5 / total:.2f}%")
    return 100.0 * correct1 / total, 100.0 * correct5 / total


def main():
    parser = argparse.ArgumentParser(description="Quantize ViT-B/16 and evaluate on ImageNet")
    parser.add_argument(
        "imagenet_dir",
        type=str,
        help="Path to ImageNet validation directory (e.g. /path/to/imagenet/val)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="vit_base_patch16_224",
        help="timm model name (default: vit_base_patch16_224)",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=8,
        help="Number of bits for quantization (default: 8)",
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Skip quantization; evaluate baseline model in full precision",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading pre-trained {args.model} from timm...")
    model = load_pretrained_vit(args.model, pretrained=True)
    model = model.to(device)

    if args.no_quantize:
        print("Skipping quantization (baseline full precision)")
    else:
        print(f"Quantizing nn.Linear layers to {args.bits}-bit...")
        quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": args.bits})])
        replaced = find_quantized_layers(model, QuantizedLinear)
        print(f"Quantized {len(replaced)} layers to {args.bits}-bit")
    print(model)

    print("Loading ImageNet validation set...")
    data_config = timm.data.resolve_data_config(model.pretrained_cfg)
    preprocess = timm.data.create_transform(**data_config)
    val_dataset = ImageFolder(args.imagenet_dir, transform=preprocess)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    print(f"Validation samples: {len(val_dataset)}")

    print("Running evaluation...")
    top1, top5 = evaluate(model, val_loader, device)
    print(f"Top-1 accuracy: {top1:.2f}%")
    print(f"Top-5 accuracy: {top5:.2f}%")


if __name__ == "__main__":
    main()
