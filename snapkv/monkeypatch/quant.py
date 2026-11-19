"""Simulated ("fake") KV-cache quantization with a bit-width per layer.

Each cached tensor is quantized to `bits` with asymmetric min-max rounding in groups of GROUP
values and immediately dequantized back to fp16, following the KIVI recipe:
  keys    per channel: each group is GROUP consecutive tokens of one channel
  values  per token:   each group is GROUP consecutive channels of one token
This reproduces the accuracy effect of a quantized cache exactly; the memory saving is computed
analytically (mem_fraction), since the dequantized tensors are stored in fp16.

Memory per value = bits/8 bytes + one fp16 scale and one fp16 offset per group (4 / GROUP bytes).
With GROUP = 32: int8 = 56.25%, int4 = 31.25%, int2 = 18.75% of fp16.
"""
import torch

GROUP = 32


def bytes_per_value(bits, group=GROUP):
    return 2.0 if bits >= 16 else bits / 8 + 4 / group


def mem_fraction(bits_per_layer, group=GROUP):
    """Cache memory as a fraction of fp16 (all layers hold the same number of tokens)."""
    return sum(bytes_per_value(b, group) / 2 for b in bits_per_layer) / len(bits_per_layer)


def fake_quant(x, bits, group=GROUP, dim=-1):
    """Quantize-dequantize `x` in groups of `group` along `dim` (edge-padded to a full group)."""
    if bits >= 16:
        return x
    xt = x.movedim(dim, -1).float()
    n = xt.shape[-1]
    pad = (-n) % group
    if pad:
        xt = torch.cat([xt, xt[..., -1:].expand(*xt.shape[:-1], pad)], dim=-1)
    g = xt.reshape(*xt.shape[:-1], -1, group)
    lo = g.amin(dim=-1, keepdim=True)
    hi = g.amax(dim=-1, keepdim=True)
    levels = 2 ** bits - 1
    scale = (hi - lo).clamp(min=1e-8) / levels
    q = ((g - lo) / scale).round().clamp(0, levels)
    out = (q * scale + lo).reshape(*xt.shape[:-1], -1)[..., :n]
    return out.to(x.dtype).movedim(-1, dim)


def quantize_cache(past_key_values, bits_per_layer, group=GROUP):
    """Return a new cache tuple with layer l quantized to bits_per_layer[l].
    Tensors are [batch, heads, tokens, head_dim]: keys grouped along tokens, values along channels."""
    out = []
    for (k, v), bits in zip(past_key_values, bits_per_layer):
        out.append((fake_quant(k, bits, group, dim=2), fake_quant(v, bits, group, dim=3)))
    return tuple(out)


def parse_quant(spec, n_layers, mid):
    """'int8' | 'int4' | 'int2' -> that width in every layer;
    'mixed-A-B' -> A bits in the `mid` layers, B bits elsewhere (e.g. mixed-4-2)."""
    if spec.startswith("int"):
        return [int(spec[3:])] * n_layers
    if spec.startswith("mixed-"):
        a, b = (int(x) for x in spec[len("mixed-"):].split("-"))
        return [a if l in mid else b for l in range(n_layers)]
    raise ValueError(f"unknown quantization {spec!r} (use int8, int4, int2 or mixed-A-B)")


def is_quant_spec(s):
    return s.startswith("int") or s.startswith("mixed-")
