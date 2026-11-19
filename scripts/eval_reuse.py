"""Reused-prefix evaluation for SnapKV and ILF on Mistral-7B-Instruct-v0.2.

Every document is compressed ONCE, before any question is seen, and then every question is
asked on that same compressed cache. Tasks:

  niah      Needle-in-a-Haystack in the style of the original test (gkamradt/LLMTest_NeedleInAHaystack):
            the Paul Graham essays concatenated in file order, truncated to --doc_tokens, with 20
            "secret code" needles spread evenly and snapped back to the preceding period, and the
            original prompt wording. Documents differ by which essay they start from.
  haystack  natural prose (WikiText, digit-free) with 20 "secret code" sentences at evenly
            spaced depths; question: "What is the secret code of <name>?" (5-digit answer)
  ruler_mk  RULER-style multi-key needles: the whole haystack is made of needles
            "One of the special magic numbers for <key> is: <7 digits>." (the needle-haystack
            setting of RULER's niah_multikey_2); 20 keys per document are queried

Scoring is teacher-forced: the question and answer prefix are fed on top of the compressed
cache, followed by the gold digits, and we check the model's top prediction at each digit.
  exact      every digit right
  digit_acc  share of digits right
  nll        mean negative log-likelihood per gold digit (chance = ln 10 = 2.30)

Commands (run with the Python 3.10 environment):
  python scripts/eval_reuse.py selfcheck
  python scripts/eval_reuse.py run --task niah --methods full snapkv --budgets 0.1 0.2 0.3 0.5 --n_docs 3
  python scripts/eval_reuse.py run --task haystack --methods full snapkv ilf-all ilf-mid ilf-except-mid random-mid \
         --budgets 0.2 --lams 0.25 1
  python scripts/eval_reuse.py run --task niah --methods int8 int4 int2 mixed-4-2 snapkv+int2 --budgets 0.5
  python scripts/eval_reuse.py summarize --task haystack

Probe methods (probe-details, probe-repeat, probe-custom with --probe_text, probe-question) score
SnapKV with a probe window instead of the document's last 32 tokens: a short generic instruction is
appended after the document only for compression, used as SnapKV's observation window, then removed
from the cache. probe-question uses each actual question as the probe (one compression per question):
an oracle upper bound. `summarize` reports gap_closed = share of the SnapKV -> oracle gap recovered.
  python scripts/eval_reuse.py run --task niah --methods snapkv probe-details probe-repeat probe-question \
         --budgets 0.1 0.3 0.5 0.7 --n_docs 30

Quantization (int8 / int4 / int2 / mixed-A-B, alone or after eviction as "snapkv+int2") is simulated:
each layer's cache is quantized and dequantized right after compression (snapkv/monkeypatch/quant.py),
so accuracy is exact while `mem_frac` (cache memory as a fraction of fp16) is computed analytically,
and decode speed is not measured for those arms.

Each (method, budget) arm is saved to <out>/<task>/<arm>.csv as soon as it finishes; arms that
already have a file are skipped, so a disconnected Colab run can simply be started again.
"""
import argparse
import math
import os
import random
import re
import time
from pathlib import Path

MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
N_LAYERS = 32
WINDOW, KERNEL, POOLING = 32, 7, "maxpool"
DEFAULT_OUT = "/content/drive/MyDrive/snapkv_ilf_results"
FULL = "full"
TASKS = ["niah", "haystack", "ruler_mk"]
NIAH_REPO = "https://github.com/gkamradt/LLMTest_NeedleInAHaystack.git"
DEFAULT_ESSAYS = "/content/LLMTest_NeedleInAHaystack/needlehaystack/PaulGrahamEssays"
NIAH_INSTRUCTION = "Don't give information outside the document or repeat your findings"
# Probes: generic instructions used ONLY as SnapKV's observation window when compressing a document
# before any question is known; they are removed from the cache afterwards (see compress()).
PROBES = {
    "probe-details": "Read the text above carefully and remember every specific detail it contains.",
    "probe-repeat": "Repeat the text above.",
}
PROBE_METHODS = [*PROBES, "probe-custom", "probe-question"]

NAMES = ["Alice", "Bruno", "Chiara", "Dmitri", "Esme", "Farid", "Greta", "Hiro", "Ines", "Jonas",
         "Kemal", "Lena", "Mateo", "Nadia", "Oskar", "Priya", "Quentin", "Rosa", "Sven", "Tariq",
         "Uma", "Viktor", "Wen", "Ximena", "Yusuf", "Zara", "Anouk", "Bodhi", "Celine", "Darius",
         "Elif", "Felix", "Gemma", "Hamid", "Ilse", "Jasper", "Kiri", "Luca", "Mira", "Nikolai"]
ADJECTIVES = ["ancient", "bitter", "brave", "bright", "broken", "calm", "clever", "cold", "crimson",
              "curious", "dark", "eager", "early", "empty", "faint", "fancy", "fierce", "gentle",
              "giant", "golden", "grumpy", "hidden", "hollow", "humble", "icy", "jolly", "lazy",
              "little", "loud", "lucky", "mellow", "misty", "modern", "noble", "odd", "pale",
              "polite", "proud", "quick", "quiet", "rapid", "rough", "royal", "rusty", "shiny",
              "silent", "silver", "sleepy", "slow", "smooth", "soft", "solid", "sour", "spicy",
              "steady", "sticky", "stormy", "sunny", "swift", "tall", "tame", "tender", "tiny",
              "vivid", "warm", "wild", "wise", "witty", "young", "zesty"]
NOUNS = ["anchor", "apple", "arrow", "badger", "basket", "beacon", "bridge", "button", "canyon",
         "castle", "cedar", "comet", "coral", "cricket", "dagger", "desert", "dolphin", "dragon",
         "engine", "falcon", "feather", "forest", "fountain", "garden", "glacier", "hammer",
         "harbor", "helmet", "island", "jacket", "kettle", "ladder", "lantern", "lemon", "magnet",
         "meadow", "mirror", "monkey", "needle", "orchard", "otter", "pebble", "pepper", "piano",
         "pillow", "planet", "pocket", "rabbit", "ribbon", "river", "rocket", "saddle", "shadow",
         "spider", "statue", "summit", "teapot", "thunder", "ticket", "tiger", "tunnel", "turtle",
         "valley", "violin", "wagon", "walnut", "whistle", "window", "wizard", "zebra"]

FALLBACK_PROSE = [  # only used when WikiText is unavailable (e.g. offline tests)
    "The river bends around the old mill before reaching the quiet town square",
    "Several merchants opened their stalls early to catch the morning travelers",
    "A light rain fell over the hills while the shepherds gathered their flocks",
    "The library kept its rarest books in a locked room on the upper floor",
    "Workers repaired the stone bridge after the spring floods damaged its arches",
    "Children ran along the beach collecting shells and smooth pieces of glass",
]


# --------------------------------------------------------------------------------------------
# Documents and questions
# --------------------------------------------------------------------------------------------
def load_prose_paragraphs():
    """Digit-free WikiText paragraphs (so the hidden codes are the only numbers in a document)."""
    try:
        from datasets import load_dataset
        lines = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
    except Exception as e:  # pragma: no cover - offline fallback
        print(f"WikiText unavailable ({e!r}); using built-in filler prose")
        return [s + "." for s in FALLBACK_PROSE] * 200
    out = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("=") or re.search(r"\d", line):
            continue
        line = line.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
        line = re.sub(r" ([.,;:!?')])", r"\1", line).replace("( ", "(")
        out.append(line)
    return out


def split_sentences(text):
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def make_haystack_doc(tok, paragraphs, doc_tokens, n_facts, seed):
    rng = random.Random(seed)
    start = rng.randrange(len(paragraphs))
    sentences, n_tok, i = [], 0, start
    while n_tok < doc_tokens:
        para = paragraphs[i % len(paragraphs)]
        sentences += split_sentences(para)
        n_tok += len(tok(para, add_special_tokens=False).input_ids)
        i += 1
    names = rng.sample(NAMES, n_facts)
    codes = [str(rng.randint(10000, 99999)) for _ in names]
    n = len(sentences)
    slots = [int((j + 0.5) * n / n_facts) for j in range(n_facts)]
    for j in reversed(range(n_facts)):
        sentences.insert(slots[j], f"The secret code of {names[j]} is {codes[j]}.")
    intro = ("Below is a long document. Secret codes for several people are hidden in it. "
             "Remember every one of them.\n\n")
    questions = [{"key": nm, "value": cd, "depth": s / n,
                  "question": f"What is the secret code of {nm}?",
                  "answer_prefix": f"The secret code of {nm} is"}
                 for nm, cd, s in zip(names, codes, slots)]
    return intro + " ".join(sentences), questions


def load_pg_essays(essays_dir=DEFAULT_ESSAYS):
    """The Paul Graham essays shipped with the original Needle-in-a-Haystack repo, in file order.
    Cloned automatically if `essays_dir` does not exist."""
    import glob
    import subprocess
    if not os.path.isdir(essays_dir):
        repo = essays_dir.split("/needlehaystack/")[0]
        print(f"cloning {NIAH_REPO} -> {repo}")
        subprocess.run(["git", "clone", "-q", "--depth", "1", NIAH_REPO, repo], check=True)
    files = sorted(glob.glob(os.path.join(essays_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"no .txt essays in {essays_dir}")
    essays = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            essays.append(fh.read())
    return essays


def make_niah_doc(tok, essays, doc_tokens, n_facts, seed):
    """Original-style haystack: essays concatenated in order (starting at essay `seed`), cut to
    `doc_tokens` tokens; needles at evenly spaced depths, each moved back to the preceding period."""
    rng = random.Random(seed)
    buf, n_tok, i = [], 0, seed % len(essays)
    while n_tok < doc_tokens:
        essay = essays[i % len(essays)]
        buf.append(essay)
        n_tok += len(tok(essay, add_special_tokens=False).input_ids)
        i += 1
    ids = tok("".join(buf), add_special_tokens=False).input_ids[:doc_tokens]
    text = tok.decode(ids)
    names = rng.sample(NAMES, n_facts)
    codes = [str(rng.randint(10000, 99999)) for _ in names]
    spots = []
    for j in range(n_facts):
        idx = int((j + 0.5) / n_facts * len(text))
        period = text.rfind(".", max(0, idx - 1000), idx)  # snap back to the preceding period
        spots.append(period + 1 if period >= 0 else idx)
    n_chars = len(text)  # depths are reported against the haystack before insertion
    for j in reversed(range(n_facts)):  # insert from the end so earlier positions stay valid
        s = spots[j]
        text = text[:s] + f" The secret code of {names[j]} is {codes[j]}." + text[s:]
    questions = [{"key": nm, "value": cd, "depth": s / n_chars,
                  "question": f"What is the secret code of {nm}? {NIAH_INSTRUCTION}",
                  "answer_prefix": f"The secret code of {nm} is"}
                 for nm, cd, s in zip(names, codes, spots)]
    return text, questions


def make_ruler_mk_doc(tok, doc_tokens, n_queries, seed):
    rng = random.Random(seed)
    keys = [f"{a}-{b}" for a in ADJECTIVES for b in NOUNS]
    rng.shuffle(keys)
    needle = "One of the special magic numbers for {} is: {}."
    per_needle = len(tok(" " + needle.format(keys[0], 1234567), add_special_tokens=False).input_ids)
    n_needles = max(n_queries, doc_tokens // per_needle)
    pairs = [(keys[i], str(rng.randint(1000000, 9999999))) for i in range(n_needles)]
    picks = [int((j + 0.5) * n_needles / n_queries) for j in range(n_queries)]
    intro = ("Some special magic numbers are hidden within the following text. Make sure to memorize "
             "it. I will quiz you about the numbers afterwards.\n\n")
    body = " ".join(needle.format(k, v) for k, v in pairs)
    questions = [{"key": pairs[p][0], "value": pairs[p][1], "depth": p / n_needles,
                  "question": f"What is the special magic number for {pairs[p][0]} mentioned in the provided text?",
                  "answer_prefix": f"The special magic number for {pairs[p][0]} mentioned in the provided text is"}
                 for p in picks]
    return intro + body, questions


def build_docs(tok, task, n_docs, doc_tokens, n_questions, seed=0, essays_dir=DEFAULT_ESSAYS):
    if task == "niah":
        essays = load_pg_essays(essays_dir)
        return [make_niah_doc(tok, essays, doc_tokens, n_questions, seed + d) for d in range(n_docs)]
    if task == "haystack":
        paragraphs = load_prose_paragraphs()
        return [make_haystack_doc(tok, paragraphs, doc_tokens, n_questions, seed + d) for d in range(n_docs)]
    if task == "ruler_mk":
        return [make_ruler_mk_doc(tok, doc_tokens, n_questions, seed + d) for d in range(n_docs)]
    raise ValueError(f"unknown task {task!r}")


def encode_prefix(tok, body):
    """[INST] + document: the part that is compressed before any question is seen."""
    return tok("[INST] " + body, add_special_tokens=True).input_ids


def encode_suffix(tok, q, with_answer=True):
    """Question + [/INST] + answer prefix (+ gold digits). Returns (ids, number of gold tokens)."""
    text = "\n\n" + q["question"] + " [/INST] " + q["answer_prefix"]
    if not with_answer:
        return tok(text, add_special_tokens=False).input_ids, 0
    ids = tok(text + " " + q["value"], add_special_tokens=False).input_ids
    n = len(q["value"])
    gold = [tok.decode([t]).strip() for t in ids[-n:]]
    if gold != list(q["value"]):
        raise ValueError(f"expected one token per digit for {q['value']!r}, got {gold}")
    return ids, n


def encode_probe(tok, text):
    """A probe is formatted like a question: "\n\n<text> [/INST]"."""
    return tok("\n\n" + text + " [/INST]", add_special_tokens=False).input_ids


# --------------------------------------------------------------------------------------------
# Methods
# --------------------------------------------------------------------------------------------
def parse_layers(spec):
    a, b = spec.split("-")
    return set(range(int(a), int(b) + 1))


def method_arms(methods, lams, mid):
    """[(label, evicts, attach_ilf kwargs or None, quantization spec or None, probe or None)].

    A method is an eviction rule, a quantization, or both joined by '+':
      full | snapkv | ilf-all | ilf-mid | ilf-except-mid | ilf-mid-band | random-mid
      probe-details | probe-repeat | probe-custom | probe-question   (SnapKV scored by a probe window)
      int8 | int4 | int2 | mixed-A-B   (A bits in the --mid layers, B elsewhere; full cache)
      snapkv+int2, ilf-mid+mixed-4-2, ...   (evict to the budget, then quantize what is kept)
    """
    from snapkv.monkeypatch.quant import is_quant_spec
    allayers = set(range(N_LAYERS))
    arms = []
    for m in methods:
        base, q = (m.split("+", 1) if "+" in m else ((FULL, m) if is_quant_spec(m) else (m, None)))
        suffix = f"+{q}" if q and base != FULL else ""
        name = q if base == FULL and q else base + suffix
        if base == FULL:
            arms.append((name, False, None, q, None))
        elif base == "snapkv":
            arms.append((name, True, None, q, None))
        elif base in PROBE_METHODS:
            arms.append((name, True, None, q, base))
        elif base == "random-mid":
            arms.append((name, True, dict(mode="random", ilf_layers=mid), q, None))
        elif base.startswith("ilf-"):
            layers = {"ilf-all": None, "ilf-mid": mid, "ilf-except-mid": allayers - mid,
                      "ilf-mid-band": mid}[base]
            extra = {"count_layers": "band"} if base == "ilf-mid-band" else {}
            for lam in lams:
                arms.append((f"{name}@lam={lam:g}", True, dict(lam=lam, ilf_layers=layers, **extra), q, None))
        else:
            raise ValueError(f"unknown method {m!r}")
    return arms


def budget_tag(b):
    return FULL if b == FULL else f"{float(b):g}"


def capacity_for(budget, prefix_len):
    """SnapKV's max_capacity_prompt (tokens kept, window included) for a fraction of the prefix."""
    if budget == FULL:
        return 10 ** 9
    return max(WINDOW + 1, int(round(float(budget) * prefix_len)))


# --------------------------------------------------------------------------------------------
# Model-side helpers (need the GPU environment)
# --------------------------------------------------------------------------------------------
def load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snapkv.monkeypatch.monkeypatch import replace_mistral
    replace_mistral()
    tok = AutoTokenizer.from_pretrained(MODEL, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="flash_attention_2", device_map="cuda").eval()
    model.config.window_size, model.config.kernel_size, model.config.pooling = WINDOW, KERNEL, POOLING
    return model, tok


def set_seq_len(model, n):
    """SnapKV tracks each layer's original sequence length here (cache length != position)."""
    for layer in model.model.layers:
        layer.self_attn.kv_seq_len = n


def compress(model, prefix_ids, capacity, probe_ids=None):
    """Prefill the document and compress its cache to `capacity` document tokens.
    probe_ids=None: plain SnapKV (its window = the document's last WINDOW tokens, kept).
    probe_ids given: the probe is appended after the document and used as SnapKV's observation
    window (window_size = probe length), so SnapKV keeps the `capacity` document tokens the probe
    attends to most; the probe's own tokens are then removed from the cache.
    Returns (cache tuple, ms for prefill + compression)."""
    import torch
    if probe_ids is None or capacity >= len(prefix_ids):
        ids, window, cap, drop = prefix_ids, WINDOW, capacity, 0
    else:
        ids, window, drop = prefix_ids + probe_ids, len(probe_ids), len(probe_ids)
        cap = capacity + window  # SnapKV keeps (cap - window) document tokens + the probe
    model.config.window_size, model.config.max_capacity_prompt = window, cap
    set_seq_len(model, 0)
    x = torch.tensor([ids], device=model.device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        pkv = model.model(x, use_cache=True).past_key_values  # decoder only: no logits needed
        if drop:
            pkv = tuple((k[:, :, :-drop].contiguous(), v[:, :, :-drop].contiguous()) for k, v in pkv)
    torch.cuda.synchronize()
    model.config.window_size = WINDOW
    return pkv, 1000 * (time.perf_counter() - t0)


def score_question(model, pkv, doc_len, suffix_ids, n_gold):
    """Teacher-forced scoring of the gold digits on top of the compressed cache.
    The cache tuple itself is never modified, so it can be reused for every question."""
    import torch
    import torch.nn.functional as F
    set_seq_len(model, doc_len)
    inp = torch.tensor([suffix_ids[:-1]], device=model.device)
    pos = torch.arange(doc_len, doc_len + inp.shape[1], device=model.device)[None]
    with torch.no_grad():
        logits = model(inp, past_key_values=pkv, position_ids=pos, use_cache=True).logits[0, -n_gold:].float()
    gold = torch.tensor(suffix_ids[-n_gold:], device=model.device)
    right = (logits.argmax(-1) == gold)
    return {"exact": bool(right.all()), "digit_acc": right.float().mean().item(),
            "nll": F.cross_entropy(logits, gold).item()}


def decode_ms_per_token(model, pkv, doc_len, suffix_ids, n_tokens=32, warmup=4):
    """Greedy decoding speed from the compressed cache (median ms per generated token)."""
    import torch
    from transformers.cache_utils import DynamicCache
    set_seq_len(model, doc_len)
    cache = DynamicCache.from_legacy_cache(pkv)  # updates make new tensors; pkv is untouched
    inp = torch.tensor([suffix_ids], device=model.device)
    pos = doc_len + inp.shape[1]
    times = []
    with torch.no_grad():
        out = model(inp, past_key_values=cache, position_ids=torch.arange(doc_len, pos, device=model.device)[None],
                    use_cache=True)
        nxt = out.logits[:, -1:].argmax(-1)
        cache = out.past_key_values
        for step in range(warmup + n_tokens):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = model(nxt, past_key_values=cache, position_ids=torch.tensor([[pos]], device=model.device),
                        use_cache=True)
            torch.cuda.synchronize()
            if step >= warmup:
                times.append(1000 * (time.perf_counter() - t0))
            nxt, cache, pos = out.logits[:, -1:].argmax(-1), out.past_key_values, pos + 1
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------------
def cmd_run(args):
    import pandas as pd
    import torch
    from snapkv.monkeypatch.ilf import attach_ilf, detach_ilf

    model, tok = load_model()
    docs = build_docs(tok, args.task, args.n_docs, args.doc_tokens, args.n_questions, args.seed, args.essays_dir)
    encoded = []
    for body, questions in docs:
        prefix = encode_prefix(tok, body)
        encoded.append((prefix, [(q, *encode_suffix(tok, q)) for q in questions],
                        encode_suffix(tok, questions[0], with_answer=False)[0]))
    lens = [len(p) for p, _, _ in encoded]
    print(f"{args.task}: {len(docs)} docs x {args.n_questions} questions, prefix {min(lens)}-{max(lens)} tokens")

    out_dir = Path(args.out) / args.task
    out_dir.mkdir(parents=True, exist_ok=True)
    mid = parse_layers(args.mid)
    from snapkv.monkeypatch.quant import mem_fraction, parse_quant, quantize_cache
    probes = dict(PROBES, **({"probe-custom": args.probe_text} if args.probe_text else {}))
    for label, evicts, ilf_kwargs, qspec, probe in method_arms(args.methods, args.lams, mid):
        bits = parse_quant(qspec, N_LAYERS, mid) if qspec else [16] * N_LAYERS
        if probe == "probe-custom" and not args.probe_text:
            raise SystemExit("probe-custom needs --probe_text")
        static_probe = encode_probe(tok, probes[probe]) if probe in probes else None
        per_question = probe == "probe-question"  # oracle: re-compress with each question as the probe
        for budget in (args.budgets if evicts else [FULL]):
            arm = f"{label}__b={budget_tag(budget)}__n={args.n_docs}__t={args.doc_tokens}__s={args.seed}"
            path = out_dir / f"{arm}.csv"
            if path.exists() and not args.overwrite:
                print(f"[skip] {arm} (already saved)")
                continue
            detach_ilf(model)
            if ilf_kwargs is not None:
                attach_ilf(model, **ilf_kwargs)
            torch.cuda.reset_peak_memory_stats()
            rows, t_arm = [], time.perf_counter()
            for d, (prefix, qs, first_q_noans) in enumerate(encoded):
                cap = capacity_for(budget, len(prefix))

                def build(probe_ids):
                    pkv, ms = compress(model, prefix, cap, probe_ids)
                    n = pkv[0][0].shape[-2]
                    if qspec:  # simulated quantization: accuracy is exact, speed is not representative
                        pkv = quantize_cache(pkv, bits)
                    return pkv, ms, n

                if not per_question:  # one compressed cache shared by every question
                    pkv, prefill_ms, cache_len = build(static_probe)
                    dec_ms = (decode_ms_per_token(model, pkv, len(prefix), first_q_noans)
                              if d < args.latency_docs and not qspec else float("nan"))
                for qi, (q, suffix, n_gold) in enumerate(qs):
                    if per_question:
                        pkv, prefill_ms, cache_len = build(encode_probe(tok, q["question"]))
                        dec_ms = float("nan")
                    rows.append({"task": args.task, "method": label, "budget": budget_tag(budget), "doc": d,
                                 "question": qi, "depth": q["depth"], **score_question(model, pkv, len(prefix), suffix, n_gold),
                                 "prefix_tokens": len(prefix), "cache_tokens": cache_len,
                                 "kept_frac": cache_len / len(prefix),
                                 "mem_frac": cache_len / len(prefix) * mem_fraction(bits),
                                 "bits": qspec or "fp16", "prefill_ms": prefill_ms,
                                 "decode_ms_per_token": dec_ms if qi == 0 else float("nan"),
                                 "probe": "(the question)" if per_question else probes.get(probe, "")})
                del pkv
            detach_ilf(model)
            df = pd.DataFrame(rows)
            df["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
            tmp = path.with_suffix(".tmp")
            df.to_csv(tmp, index=False)
            os.replace(tmp, path)
            print(f"[done] {arm}: exact {df.exact.mean():.3f}  digit {df.digit_acc.mean():.3f}  "
                  f"nll {df.nll.mean():.2f}  kept {df.kept_frac.mean():.2f}  mem {df.mem_frac.mean():.3f}  "
                  f"({time.perf_counter() - t_arm:.0f}s)")


def bootstrap_ci(per_doc, n=2000, seed=0):
    import numpy as np
    rng = np.random.default_rng(seed)
    per_doc = np.asarray(per_doc, dtype=float)
    means = [per_doc[rng.integers(0, len(per_doc), len(per_doc))].mean() for _ in range(n)]
    return np.percentile(means, [2.5, 97.5])


def cmd_summarize(args):
    import pandas as pd
    files = sorted((Path(args.out) / args.task).glob("*.csv"))
    if not files:
        raise SystemExit(f"no results in {Path(args.out) / args.task}")
    df = pd.concat([pd.read_csv(f, dtype={"budget": str}) for f in files if "__n=3__" not in f.name or args.include_pilot],
                   ignore_index=True)
    if "mem_frac" not in df:
        df["mem_frac"] = df.kept_frac
    df["mem_frac"] = df.mem_frac.fillna(df.kept_frac)
    rows = []
    for (method, budget), g in df.groupby(["method", "budget"], sort=False):
        per_doc = g.groupby("doc").exact.mean()
        lo, hi = bootstrap_ci(per_doc.values)
        row = {"method": method, "budget": budget, "n_questions": len(g),
               "exact": g.exact.mean(), "exact_lo": lo, "exact_hi": hi,
               "digit_acc": g.digit_acc.mean(), "nll": g.nll.mean(), "kept_frac": g.kept_frac.mean(),
               "mem_frac": g.mem_frac.mean(),
               "prefill_ms": g.drop_duplicates("doc").prefill_ms.median(),
               "decode_ms_per_token": g.decode_ms_per_token.median(), "peak_mem_GiB": g.peak_mem_GiB.max()}
        base = df[(df.method == "snapkv") & (df.budget == budget)]
        if method != "snapkv" and len(base):
            m = g.merge(base, on=["doc", "question"], suffixes=("", "_snap"))
            diff = (m.exact.astype(float) - m.exact_snap.astype(float)).groupby(m.doc).mean()
            dlo, dhi = bootstrap_ci(diff.values)
            row.update({"vs_snapkv": diff.mean(), "vs_snapkv_lo": dlo, "vs_snapkv_hi": dhi})
            oracle = df[(df.method == "probe-question") & (df.budget == budget)]
            gap = oracle.exact.mean() - base.exact.mean() if len(oracle) else float("nan")
            if method.startswith("probe-") and method != "probe-question" and gap > 0:
                row["gap_closed"] = (g.exact.mean() - base.exact.mean()) / gap  # share of SnapKV->oracle gap
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["budget", "method"])
    path = Path(args.out) / f"summary_{args.task}.csv"
    summary.to_csv(path, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summary.round(3).to_string(index=False))
        print("\naccuracy vs cache memory (fraction of the fp16 full cache):")
        print(summary.sort_values("mem_frac")[["method", "budget", "mem_frac", "exact", "exact_lo", "exact_hi",
                                               "digit_acc", "nll"]].round(3).to_string(index=False))
        df["depth_bin"] = pd.cut(df.depth, [0, 0.25, 0.5, 0.75, 1.0])
        print("\nexact match by fact depth:")
        print(df.pivot_table(index=["budget", "method"], columns="depth_bin", values="exact",
                             aggfunc="mean", observed=False).round(2).to_string())
    print(f"\nsaved {path}")


def cmd_selfcheck(args):
    """Checks the reuse harness on the GPU before any experiment:
    1. full cache: compress-then-question gives the same predictions as one plain forward pass
       (i.e. positions and cache reuse are handled correctly);
    2. scoring a question does not modify the stored cache (it can be reused);
    3. both tasks tokenize every digit as its own token."""
    import torch
    model, tok = load_model()
    for task in TASKS:
        body, questions = build_docs(tok, task, 1, 2000, 4, seed=123)[0]
        prefix = encode_prefix(tok, body)
        pkv, _ = compress(model, prefix, capacity_for(FULL, len(prefix)))
        snapshot = [t.clone() for t in pkv[0]]
        agree, max_diff = 0, 0.0
        for q in questions:
            suffix, n = encode_suffix(tok, q)
            reused = score_question(model, pkv, len(prefix), suffix, n)
            with torch.no_grad():
                set_seq_len(model, 0)
                full_ids = torch.tensor([prefix + suffix[:-1]], device=model.device)
                ref = model(full_ids).logits[0, -n:].float()
                set_seq_len(model, len(prefix))
                pos = torch.arange(len(prefix), len(prefix) + len(suffix) - 1, device=model.device)[None]
                got = model(torch.tensor([suffix[:-1]], device=model.device), past_key_values=pkv,
                            position_ids=pos).logits[0, -n:].float()
            agree += int(torch.equal(ref.argmax(-1), got.argmax(-1)))
            max_diff = max(max_diff, (ref - got).abs().max().item())
            print(f"  {task}: {q['key']!r} gold {q['value']} | exact {reused['exact']} nll {reused['nll']:.2f}")
        unchanged = all(torch.equal(a, b) for a, b in zip(snapshot, pkv[0]))
        print(f"{task}: prefix {len(prefix)} tokens | one-pass vs reused argmax agree {agree}/{len(questions)} "
              f"| max logit diff {max_diff:.3f} | cache unchanged after scoring: {unchanged}")
        assert agree == len(questions) and unchanged, "reuse harness does not match a plain forward pass"
        from snapkv.monkeypatch.quant import quantize_cache
        fp16 = [score_question(model, pkv, len(prefix), *encode_suffix(tok, q)) for q in questions]
        for spec_bits in (8, 4, 2):
            qkv = quantize_cache(pkv, [spec_bits] * len(pkv))
            res = [score_question(model, qkv, len(prefix), *encode_suffix(tok, q)) for q in questions]
            print(f"  {task}: int{spec_bits} full cache | exact {sum(r['exact'] for r in res)}/{len(res)} "
                  f"| mean nll {sum(r['nll'] for r in res) / len(res):.2f}")
            if spec_bits == 8:
                assert [r["exact"] for r in res] == [r["exact"] for r in fp16], "int8 should match fp16"
        probe = encode_probe(tok, PROBES["probe-details"])
        cap = capacity_for(0.3, len(prefix))
        snap, _ = compress(model, prefix, cap)
        prb, _ = compress(model, prefix, cap, probe)
        assert snap[0][0].shape[-2] == cap and prb[0][0].shape[-2] == cap, "probe tokens must be removed"
        differ = not all(torch.equal(a[0], b[0]) for a, b in zip(snap, prb))
        near, _ = compress(model, prefix, len(prefix) - 1, probe)
        agree = [score_question(model, near, len(prefix), *encode_suffix(tok, q))["exact"] for q in questions]
        print(f"  {task}: 30% budget keeps {cap} document tokens with SnapKV and with the probe (probe removed) "
              f"| selections differ: {differ} | probe keeping all but 1 token: exact {sum(agree)}/{len(agree)} "
              f"(full cache {sum(r['exact'] for r in fp16)}/{len(fp16)})")
        assert differ, "the probe should change which tokens SnapKV keeps"
    print("SELFCHECK PASSED")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--task", choices=TASKS, required=True)
    r.add_argument("--methods", nargs="+", default=[FULL, "snapkv"],
                   help="full snapkv ilf-all ilf-mid ilf-except-mid ilf-mid-band random-mid | int8 int4 int2 "
                        "mixed-A-B | eviction+quantization, e.g. snapkv+int2 ilf-mid+mixed-4-2 | "
                        "probe-details probe-repeat probe-custom probe-question (e.g. probe-details+int2)")
    r.add_argument("--probe_text", default=None, help="probe wording for --methods probe-custom")
    r.add_argument("--budgets", nargs="+", type=float, default=[0.2],
                   help="fraction of the document's tokens SnapKV keeps (window included)")
    r.add_argument("--lams", nargs="+", type=float, default=[1.0])
    r.add_argument("--mid", default="11-21", help="middle layers, inclusive (Mistral has 32)")
    r.add_argument("--n_docs", type=int, default=30)
    r.add_argument("--n_questions", type=int, default=20)
    r.add_argument("--doc_tokens", type=int, default=8000)
    r.add_argument("--latency_docs", type=int, default=5, help="documents on which decode speed is timed")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--essays_dir", default=DEFAULT_ESSAYS, help="Paul Graham essays for --task niah")
    r.add_argument("--out", default=DEFAULT_OUT)
    r.add_argument("--overwrite", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--task", choices=TASKS, required=True)
    s.add_argument("--out", default=DEFAULT_OUT)
    s.add_argument("--include_pilot", action="store_true", help="also include 3-document pilot files")
    sub.add_parser("selfcheck")
    args = p.parse_args()
    {"run": cmd_run, "summarize": cmd_summarize, "selfcheck": cmd_selfcheck}[args.cmd](args)


if __name__ == "__main__":
    main()
