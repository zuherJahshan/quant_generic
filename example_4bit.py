"""Dummy example: quantize ViT-B/16 to 4-bit."""

from quantize import load_pretrained_vit, quantize_model, QuantizedLayerWrapper, find_quantized_layers
import torch
import torch.nn as nn

print("Loading pre-trained ViT-B/16...")
model = load_pretrained_vit()
model.eval()

print("Quantizing nn.Linear layers to 4-bit...")
quantize_model(model, [nn.Linear], QuantizedLayerWrapper, bits=4)

wrapped = find_quantized_layers(model, QuantizedLayerWrapper)
print(f"Quantized {len(wrapped)} layers to 4-bit")

print("Running dummy forward pass...")
x = torch.randn(2, 3, 224, 224)
with torch.no_grad():
    out = model(x)
print(f"Input: {x.shape} -> Output: {out.shape}")
print("Done.")
