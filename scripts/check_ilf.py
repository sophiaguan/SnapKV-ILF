"""GPU sanity checks for ILF on Mistral-7B-Instruct-v0.2 (run before any experiment).

1. ILF with lam=0 gives a cache identical to plain SnapKV.
2. ILF with ilf_layers=set() gives a cache identical to plain SnapKV.
3. ILF on the middle layers changes only the middle layers' caches.
4. Every layer keeps exactly max_capacity_prompt tokens.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from snapkv.monkeypatch.monkeypatch import replace_mistral
from snapkv.monkeypatch.ilf import attach_ilf, detach_ilf

replace_mistral()
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CAPACITY = 1024
MID = set(range(11, 22))  # layers 11-21 of 32

tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="cuda").eval()
model.config.window_size, model.config.max_capacity_prompt = 32, CAPACITY
model.config.kernel_size, model.config.pooling = 7, "maxpool"

filler = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
doc = filler * 120 + "The secret code of Alice is 48213. " + filler * 120
question = "What is the secret code of Alice? Answer with the number only."
prompt = tok.apply_chat_template([{"role": "user", "content": doc + "\n\n" + question}], tokenize=False)
ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()


@torch.no_grad()
def prefill_cache():
    for layer in model.model.layers:
        layer.self_attn.kv_seq_len = 0
    pkv = model(ids, use_cache=True).past_key_values
    return [(pkv[l][0].clone(), pkv[l][1].clone()) for l in range(len(model.model.layers))]


@torch.no_grad()
def answer():
    out = model.generate(ids, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def changed_layers(a, b):
    return [l for l in range(len(a)) if not (torch.equal(a[l][0], b[l][0]) and torch.equal(a[l][1], b[l][1]))]


detach_ilf(model)
snap = prefill_cache()
print(f"SnapKV            | answer {answer()!r}")

attach_ilf(model, lam=0.0)
c = prefill_cache()
assert not changed_layers(snap, c), "lam=0 should equal SnapKV"
print(f"ILF lam=0         | identical to SnapKV: yes | answer {answer()!r}")

attach_ilf(model, lam=1.0, ilf_layers=set())
c = prefill_cache()
assert not changed_layers(snap, c), "ilf_layers=set() should equal SnapKV"
print("ILF no layers     | identical to SnapKV: yes")

coord = attach_ilf(model, lam=1.0, ilf_layers=MID)
c = prefill_cache()
diff = changed_layers(snap, c)
assert set(diff) <= MID, f"layers outside MID changed: {sorted(set(diff) - MID)}"
assert all(k.shape[-2] == CAPACITY for k, _ in c), "every layer should keep CAPACITY tokens"
print(f"ILF middle lam=1  | changed layers {diff} (all inside {min(MID)}-{max(MID)}) | answer {answer()!r}")
print(f"                  | ILF stats: {coord.last_stats}")

detach_ilf(model)
print(f"peak GPU memory: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
print("ALL CHECKS PASSED")
