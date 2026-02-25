"""
Quantization utilities for transformer models.
Downloads pre-trained ViT (ImageNet) and provides a generic recursive
quantization function that wraps target layers with a user-supplied wrapper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights


def load_pretrained_vit(weights=ViT_B_16_Weights.IMAGENET1K_V1):
    """Load a pre-trained Vision Transformer (ViT-B/16) with ImageNet weights."""
    return vit_b_16(weights=weights)


class QuantizedLayerWrapper(nn.Module):
    """
    Default quantization wrapper for nn.Linear layers.
    Performs per-output-channel symmetric fake quantization of the weight.
    """

    def __init__(self, module: nn.Module, bits: int = 8, **kwargs):
        super().__init__()
        assert isinstance(module, nn.Linear), "QuantizedLayerWrapper expects nn.Linear"
        self.module = module
        self.bits = bits
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(maxq, dtype=torch.float32))
        self._compute_scales()

    def _compute_scales(self):
        """Compute per-channel symmetric quantization scales from the weight."""
        w = self.module.weight.data
        # Per-output-channel: shape (out_features,)
        w_flat = w.reshape(w.shape[0], -1)
        w_max = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = w_max / self.maxq
        self.register_buffer("scale", scale.reshape(-1, *([1] * (w.dim() - 1))))

    def _fake_quant_weight(self):
        """Quantize and dequantize the weight (fake quantization)."""
        w = self.module.weight
        scale = self.scale.to(w.device)
        q = torch.clamp(torch.round(w / scale), -(self.maxq + 1), self.maxq)
        return (scale * q).to(w.dtype)

    def forward(self, x):
        w_q = self._fake_quant_weight()
        return F.linear(x, w_q, self.module.bias)


def quantize_model(model, layer_types, wrapper_cls, name: str = "", **wrapper_kwargs):
    """
    Recursively search the module tree for layers matching layer_types and wrap
    each with wrapper_cls(module, **wrapper_kwargs).

    Args:
        model: The root module to process.
        layer_types: List of nn.Module subclasses to match (e.g., [nn.Linear]).
        wrapper_cls: Class to wrap matching layers; must accept (module, **kwargs).
        name: Internal use for recursion.
        **wrapper_kwargs: Passed to wrapper_cls constructor.
    """
    if isinstance(model, wrapper_cls):
        return

    for attr in dir(model):
        if attr.startswith("_"):
            continue
        try:
            tmp = getattr(model, attr)
        except (AttributeError, TypeError):
            continue
        if type(tmp) in layer_types:
            setattr(model, attr, wrapper_cls(tmp, **wrapper_kwargs))
        elif isinstance(tmp, nn.Sequential):
            replaced = []
            for child in tmp.children():
                if type(child) in layer_types:
                    replaced.append(wrapper_cls(child, **wrapper_kwargs))
                else:
                    replaced.append(child)
            setattr(model, attr, nn.Sequential(*replaced))
        elif isinstance(tmp, nn.ModuleList):
            replaced = []
            for child in tmp.children():
                if type(child) in layer_types:
                    replaced.append(wrapper_cls(child, **wrapper_kwargs))
                else:
                    replaced.append(child)
            setattr(model, attr, nn.ModuleList(replaced))

    for child_name, child in list(model.named_children()):
        quantize_model(
            child,
            layer_types,
            wrapper_cls,
            name=f"{name}.{child_name}" if name else child_name,
            **wrapper_kwargs,
        )


def find_quantized_layers(model, wrapper_cls, name: str = ""):
    """
    Recursively find all modules that are instances of wrapper_cls.

    Returns:
        Dict mapping full module name to the wrapped module.
    """
    result = {}
    for child_name, child in model.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, wrapper_cls):
            result[full_name] = child
        else:
            result.update(find_quantized_layers(child, wrapper_cls, full_name))
    return result


if __name__ == "__main__":
    print("Loading pre-trained ViT-B/16 (ImageNet-1K)...")
    model = load_pretrained_vit()
    model.eval()

    print("\n--- Model architecture (before quantization) ---")
    print(model)

    print("\n--- Applying quantization to nn.Linear layers ---")
    quantize_model(model, [nn.Linear], QuantizedLayerWrapper, bits=8)

    print("\n--- Model architecture (after quantization) ---")
    print(model)

    wrapped = find_quantized_layers(model, QuantizedLayerWrapper)
    print(f"\nQuantized {len(wrapped)} layers: {list(wrapped.keys())[:5]}...")

    print("\n--- Running dummy forward pass ---")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Forward pass succeeded.")
