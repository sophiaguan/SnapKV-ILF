"""ILF-diversified SnapKV, optionally restricted to a band of layers.

SnapKV compresses each layer on its own during prefill. ILF needs every layer's scores
before choosing, so with ILF attached the prefill stores each layer's full KV and its SnapKV
scores, and a forward hook on the decoder compresses all layers once the last layer is done.

For token i (outside the observation window), with s_{i,u} the SnapKV score in unit u
(a unit is one (layer, head) pair, matching SnapKV's per-head selection):

    d_i     = #units u where s_{i,u} is among that unit's top-k   (k = tokens SnapKV keeps)
    ILF_i   = log((U + 1) / (d_i + 1)),  set to 0 when d_i = 0 if gated
    S_{i,u} = s_{i,u} * (1 + lam * ILF_i)     only in layers listed in `ilf_layers`

Layers outside `ilf_layers` keep the plain SnapKV scores, so with lam = 0 or ilf_layers = set()
the result is identical to SnapKV.
"""
import torch


class ILFCoordinator:
    def __init__(self, lam=1.0, ilf_layers=None, count_layers="all", mode="ilf", gated=True, seed=0):
        """
        lam:          ILF weight. 0 reproduces SnapKV.
        ilf_layers:   layer indices where the ILF (or random) scores are used. None = every layer.
        count_layers: "all"  - d_i counts over every layer
                      "band" - d_i counts only over `ilf_layers`
        mode:         "ilf"    - S = s * (1 + lam * ILF)
                      "random" - random scores in `ilf_layers` (control: de-correlation without scores)
        gated:        no ILF boost for tokens that no unit would keep (d_i = 0)
        """
        assert count_layers in ("all", "band") and mode in ("ilf", "random")
        self.lam, self.count_layers, self.mode, self.gated, self.seed = lam, count_layers, mode, gated, seed
        self.ilf_layers = None if ilf_layers is None else set(ilf_layers)
        self.scores = {}
        self.last_stats = {}

    def applies_to(self, layer_idx):
        return self.ilf_layers is None or layer_idx in self.ilf_layers

    def record(self, layer_idx, scores):
        self.scores[layer_idx] = scores

    def new_scores(self, k):
        """Scores to rank by, per layer. Unchanged layers get the original tensor back."""
        layers = sorted(self.scores)
        on = [self.applies_to(l) for l in layers]
        if not any(on) or (self.mode == "ilf" and self.lam == 0):
            return {l: self.scores[l] for l in layers}
        if self.mode == "random":
            gen = torch.Generator(device=self.scores[layers[0]].device).manual_seed(self.seed)
            return {l: (torch.rand(self.scores[l].shape, generator=gen, device=self.scores[l].device,
                                   dtype=torch.float32) if o else self.scores[l])
                    for l, o in zip(layers, on)}
        counted = [l for l, o in zip(layers, on) if self.count_layers == "all" or o]
        S = torch.stack([self.scores[l].float() for l in counted])          # [L', bsz, heads, M]
        tau = S.topk(k, dim=-1).values[..., -1:]                             # each unit's k-th best score
        d = (S >= tau).sum(dim=(0, 2))                                       # [bsz, M]
        units = S.shape[0] * S.shape[2]
        ilf = torch.log((units + 1) / (d.float() + 1))
        if self.gated:
            ilf = ilf * (d > 0)
        boost = (1 + self.lam * ilf)[:, None, :]                            # [bsz, 1, M]
        self.last_stats = {"units": units, "mean_d": d.float().mean().item(),
                           "frac_d0": (d == 0).float().mean().item(), "max_boost": boost.max().item()}
        return {l: ((self.scores[l].float() * boost) if o else self.scores[l]) for l, o in zip(layers, on)}

    def finalize(self, past_key_values, clusters):
        """Compress every recorded layer. Returns the (possibly new) past_key_values object."""
        layers = sorted(self.scores)
        k = clusters[layers[0]].max_capacity_prompt - clusters[layers[0]].window_size
        scores = self.new_scores(k)
        self.scores = {}
        legacy = isinstance(past_key_values, (tuple, list))
        kv = [list(x) for x in past_key_values] if legacy else None
        for l in layers:
            keys, values = (kv[l][0], kv[l][1]) if legacy else (past_key_values.key_cache[l],
                                                                  past_key_values.value_cache[l])
            keys, values = clusters[l].compress(keys, values, scores[l])
            if legacy:
                kv[l] = (keys, values)
            else:
                past_key_values.key_cache[l], past_key_values.value_cache[l] = keys, values
        return tuple(tuple(x) for x in kv) if legacy else past_key_values


def attach_ilf(model, **kwargs):
    """Turn on ILF for `model` (already patched with replace_mistral / replace_llama).
    Returns the coordinator; call detach_ilf(model) to go back to plain SnapKV."""
    detach_ilf(model)
    coord = ILFCoordinator(**kwargs)
    attns = [layer.self_attn for layer in model.model.layers]
    for a in attns:
        a.ilf = coord

    def hook(module, args, kwargs_, output):
        if coord.scores:  # a prefill just recorded scores: compress every layer now
            output.past_key_values = coord.finalize(output.past_key_values, [a.kv_cluster for a in attns])
        return output

    model._ilf_hook = model.model.register_forward_hook(hook, with_kwargs=True)
    return coord


def detach_ilf(model):
    if getattr(model, "_ilf_hook", None) is not None:
        model._ilf_hook.remove()
        model._ilf_hook = None
    for layer in model.model.layers:
        layer.self_attn.ilf = None
