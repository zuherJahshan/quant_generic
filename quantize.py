"""
Quantization utilities for transformer models.
Loads pre-trained ViT from timm and provides a generic recursive
quantization function that replaces target layers with user-supplied classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm


def load_pretrained_vit(model_name: str = "vit_base_patch16_224", pretrained: bool = True):
    """Load a pre-trained Vision Transformer from timm (ViT-B/16 by default)."""
    return timm.create_model(model_name, pretrained=pretrained)


class QuantizedLinear(nn.Module):
    """
    Standalone replacement for nn.Linear with per-output-channel symmetric
    fake quantization of the weight, and per-token symmetric fake quantization
    of activations with online scale.
    """

    def __init__(self, original: nn.Linear, bits: int = 8, **kwargs):
        super().__init__()
        assert isinstance(original, nn.Linear), "QuantizedLinear expects nn.Linear"
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.bits = bits
        self.weight = nn.Parameter(original.weight.data.clone())
        self.bias = nn.Parameter(original.bias.data.clone()) if original.bias is not None else None
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(maxq, dtype=torch.float32))
        self._compute_scales()

    def _compute_scales(self):
        """Compute per-channel symmetric quantization scales from the weight."""
        w = self.weight.data
        # Per-output-channel: shape (out_features,)
        w_flat = w.reshape(w.shape[0], -1)
        w_max = w_flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
        scale = w_max / self.maxq
        self.register_buffer("scale", scale.reshape(-1, *([1] * (w.dim() - 1))))

    def _fake_quant_weight(self):
        """Quantize and dequantize the weight (fake quantization)."""
        w = self.weight
        dev = w.device
        scale = self.scale.to(dev)
        maxq = self.maxq.to(dev)
        q = torch.clamp(torch.round(w / scale), -(maxq + 1), maxq)
        return (scale * q).to(w.dtype)

    def forward(self, x):
        # 1. Online per-token activation scale
        x_flat = x.reshape(-1, x.shape[-1])  # (B*N, D) or (B, D)
        x_max = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_act = x_max / self.maxq.to(x.device)
        scale_act = scale_act.reshape(*x.shape[:-1], 1)  # (B, N, 1) or (B, 1)

        # 2. Fake-quantize activation
        maxq = self.maxq.to(x.device)
        q_act = torch.clamp(torch.round(x / scale_act), -(maxq + 1), maxq)
        x_q = (scale_act * q_act).to(x.dtype)

        # 3. Fake-quantize weight and compute output
        w_q = self._fake_quant_weight()
        return F.linear(x_q, w_q, self.bias)


class InputQuantizedWrapper(nn.Module):
    """
    Wrapper that quantizes the input before passing it to the wrapped module.
    Uses per-token symmetric fake quantization with online scale.
    """

    def __init__(self, module: nn.Module, bits: int = 8, **kwargs):
        super().__init__()
        self.module = module
        self.bits = bits
        maxq = 2 ** (bits - 1) - 1
        self.register_buffer("maxq", torch.tensor(maxq, dtype=torch.float32))

    def _quantize_input(self, x):
        """Per-token symmetric fake quantization of the input."""
        x_flat = x.reshape(-1, x.shape[-1])
        x_max = x_flat.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale_act = x_max / self.maxq.to(x.device)
        scale_act = scale_act.reshape(*x.shape[:-1], 1)
        maxq = self.maxq.to(x.device)
        q_act = torch.clamp(torch.round(x / scale_act), -(maxq + 1), maxq)
        return (scale_act * q_act).to(x.dtype)

    def forward(self, x):
        x_q = self._quantize_input(x)
        return self.module(x_q)


def quantize_model(model, replacement_list, input_quantize_list=None, name: str = ""):
    """
    Recursively replace layers according to replacement_list, and optionally
    wrap layers in input_quantize_list with InputQuantizedWrapper.

    Args:
        model: Root module to process.
        replacement_list: List of (source_type, replacement_cls) or
                         (source_type, replacement_cls, replacement_kwargs).
                         Each tuple specifies: which layer type to replace,
                         and the class to replace it with.
        input_quantize_list: Optional list of (layer_type,) or
                            (layer_type, input_quantize_kwargs). Layers matching
                            these types will be wrapped with InputQuantizedWrapper
                            so their inputs are quantized. Applied after replacement.
        name: Internal use for recursion.
    """
    replacement_clses = [r[1] for r in replacement_list]
    if any(isinstance(model, cls) for cls in replacement_clses):
        return
    if isinstance(model, InputQuantizedWrapper):
        return  # Don't recurse into wrapper to avoid double-wrapping

    # Build lookup: source_type -> (replacement_cls, kwargs)
    rules = {}
    for item in replacement_list:
        source_type = item[0]
        replacement_cls = item[1]
        replacement_kwargs = item[2] if len(item) > 2 else {}
        rules[source_type] = (replacement_cls, replacement_kwargs)

    # Build input quantize lookup: layer_type -> kwargs
    input_quantize_rules = {}
    if input_quantize_list:
        for item in input_quantize_list:
            layer_type = item[0]
            input_quantize_kwargs = item[1] if len(item) > 1 else {}
            input_quantize_rules[layer_type] = input_quantize_kwargs

    def process_module(tmp):
        """Apply replacement and/or input quantization to a module."""
        result = tmp
        if type(tmp) in rules:
            replacement_cls, replacement_kwargs = rules[type(tmp)]
            result = replacement_cls(tmp, **replacement_kwargs)
        if type(result) in input_quantize_rules and not isinstance(result, InputQuantizedWrapper):
            kwargs = input_quantize_rules[type(result)]
            result = InputQuantizedWrapper(result, **kwargs)
        return result

    for attr in dir(model):
        if attr.startswith("_"):
            continue
        try:
            tmp = getattr(model, attr)
        except (AttributeError, TypeError):
            continue
        if type(tmp) in rules or type(tmp) in input_quantize_rules:
            setattr(model, attr, process_module(tmp))
        elif isinstance(tmp, nn.Sequential):
            processed = [process_module(child) for child in tmp.children()]
            setattr(model, attr, nn.Sequential(*processed))
        elif isinstance(tmp, nn.ModuleList):
            processed = [process_module(child) for child in tmp.children()]
            setattr(model, attr, nn.ModuleList(processed))

    for child_name, child in list(model.named_children()):
        quantize_model(
            child,
            replacement_list,
            input_quantize_list=input_quantize_list,
            name=f"{name}.{child_name}" if name else child_name,
        )


def find_quantized_layers(model, replacement_cls, name: str = ""):
    """
    Recursively find all modules that are instances of replacement_cls.

    Returns:
        Dict mapping full module name to the replacement module.
    """
    result = {}
    for child_name, child in model.named_children():
        full_name = f"{name}.{child_name}" if name else child_name
        if isinstance(child, replacement_cls):
            result[full_name] = child
        else:
            result.update(find_quantized_layers(child, replacement_cls, full_name))
    return result


if __name__ == "__main__":
    print("Loading pre-trained ViT-B/16 from timm (ImageNet-1K)...")
    model = load_pretrained_vit("vit_base_patch16_224", pretrained=True)
    model.eval()

    print("\n--- Model architecture (before quantization) ---")
    print(model)

    print("\n--- Applying quantization to nn.Linear layers ---")
    quantize_model(model, [(nn.Linear, QuantizedLinear, {"bits": 8})])

    print("\n--- Model architecture (after quantization) ---")
    print(model)

    replaced = find_quantized_layers(model, QuantizedLinear)
    print(f"\nQuantized {len(replaced)} layers: {list(replaced.keys())[:5]}...")

    print("\n--- Running dummy forward pass ---")
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print("Forward pass succeeded.")
