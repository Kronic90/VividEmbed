"""
benchmark_embed_vs_vivid.py — Head-to-head: Letta (MiniLM) vs VividEmbedder

Runs Mem2ActBench with two embedding conditions, each across 5 random seeds:
  1. embedding   — MemGPT/Letta-style: vanilla all-MiniLM-L6-v2 (ONNX, 384-d)
  2. vividembed  — Fine-tuned VividEmbedder with emotion/mood/importance tokens

Both use identical:
  • LLM (Gemma 3 12B Q4_K_M, temp=0.0)
  • System prompt + prompt shape
  • Session ingestion data (same passages)
  • Retrieval budget (top_k=10 + tool-first scan + implicit ref resolution)
  • Random seed → same shuffled subset per seed

Usage:
  python benchmark_embed_vs_vivid.py --max-eval 100 --seeds 42 123 456 789 1337
  python benchmark_embed_vs_vivid.py --max-eval 400
"""

import sys
import os
import json
import time
import re
import random
import argparse
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from collections import Counter

try:
    import torch  # noqa: F401
except ImportError:
    pass

import numpy as np
if not hasattr(np.ndarray, '__class_getitem__'):
    np.ndarray.__class_getitem__ = classmethod(lambda cls, *args: cls)
from tqdm import tqdm

# ── Paths ────────────────────────────────────────────────────────────────
WORKSPACE = Path(__file__).parent
STANDALONE_MEM = WORKSPACE / "standalone memory"
MEM2ACT_DIR = WORKSPACE / "Mem2ActBench_repo"
RESULTS_DIR = WORKSPACE / "benchmark_results"
RESULTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(STANDALONE_MEM))

VIVID_MODEL_PATH = str(STANDALONE_MEM / "vivid_model_output" / "best")

# ── Previous baselines ───────────────────────────────────────────────────
PREV_BASELINES = {
    "no_memory":  {"tool_accuracy": 0.052, "f1": 0.118, "bleu1": 0.273, "n": 400},
    "vividness":  {"tool_accuracy": 0.500, "f1": 0.573, "bleu1": 0.671, "n": 100},
    "combined":   {"tool_accuracy": 0.372, "f1": 0.457, "bleu1": 0.597, "n": 400},
    "embedding":  {"tool_accuracy": 0.440, "f1": 0.512, "bleu1": 0.641, "n": 100},
}

# ── LLM ──────────────────────────────────────────────────────────────────
MODEL_PATH = r"D:\AiStuff\google_gemma-3-12b-it-Q4_K_M.gguf"
CTX_SIZE = 8192
MAX_TOKENS = 512
_llm_cache = None


def load_llm():
    global _llm_cache
    if _llm_cache is not None:
        return _llm_cache
    from llama_cpp import Llama
    print(f"Loading LLM: {Path(MODEL_PATH).name}")
    t0 = time.time()
    _llm_cache = Llama(
        model_path=MODEL_PATH,
        n_ctx=CTX_SIZE,
        n_gpu_layers=48,
        verbose=False,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    return _llm_cache


def generate(llm, messages, max_tokens=MAX_TOKENS, temperature=0.0):
    total_chars = sum(len(m.get("content", "")) for m in messages)
    budget = int((CTX_SIZE - max_tokens - 200) * 3.5)
    while total_chars > budget and len(messages) > 2:
        messages = [messages[0]] + messages[2:]
        total_chars = sum(len(m.get("content", "")) for m in messages)
    if total_chars > budget:
        last = messages[-1]
        last["content"] = last["content"][:max(200, len(last["content"]) - (total_chars - budget))]
    try:
        resp = llm.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"  [gen error] {e}")
        return ""


# ══════════════════════════════════════════════════════════════════════════
#  EMBEDDING MEMORY — Letta/MemGPT baseline (vanilla MiniLM ONNX)
# ══════════════════════════════════════════════════════════════════════════

ONNX_MODEL_DIR = r"D:\AiStuff\all-MiniLM-L6-v2-onnx"
_embed_session = None
_embed_tokenizer = None


def _get_embedder():
    global _embed_session, _embed_tokenizer
    if _embed_session is not None:
        return _embed_session, _embed_tokenizer

    import onnxruntime as ort
    from tokenizers import Tokenizer

    _embed_tokenizer = Tokenizer.from_file(
        os.path.join(ONNX_MODEL_DIR, "tokenizer.json"))
    _embed_tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
    _embed_tokenizer.enable_truncation(max_length=256)

    _embed_session = ort.InferenceSession(
        os.path.join(ONNX_MODEL_DIR, "onnx", "model.onnx"),
        providers=["CPUExecutionProvider"],
    )
    return _embed_session, _embed_tokenizer


def embed_texts(texts):
    session, tokenizer = _get_embedder()
    encs = tokenizer.encode_batch(texts)
    ids = np.array([e.ids for e in encs], dtype=np.int64)
    mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
    tids = np.zeros_like(ids)
    outputs = session.run(
        None, {"input_ids": ids, "attention_mask": mask, "token_type_ids": tids})
    hidden = outputs[0]
    mask_exp = np.expand_dims(mask, -1).astype(np.float32)
    pooled = (hidden * mask_exp).sum(axis=1) / np.maximum(mask_exp.sum(axis=1), 1e-9)
    norms = np.linalg.norm(pooled, axis=1, keepdims=True)
    return pooled / np.maximum(norms, 1e-9)


class EmbeddingMemory:
    """MemGPT/Letta-style archival memory using vanilla MiniLM embeddings."""

    def __init__(self):
        self.passages = []
        self.embeddings = None

    def store(self, text):
        if not text.strip():
            return
        self.passages.append(text.strip())
        emb = embed_texts([text.strip()])
        if self.embeddings is None:
            self.embeddings = emb
        else:
            self.embeddings = np.vstack([self.embeddings, emb])

    def retrieve(self, query, top_k=10):
        if not self.passages or self.embeddings is None:
            return []
        q_emb = embed_texts([query])
        sims = (self.embeddings @ q_emb.T).squeeze()
        if sims.ndim == 0:
            sims = np.array([float(sims)])
        k = min(top_k, len(self.passages))
        top_idx = np.argsort(sims)[-k:][::-1]
        return [self.passages[i] for i in top_idx if sims[i] > 0.0]


# ══════════════════════════════════════════════════════════════════════════
#  VIVIDEMBED MEMORY — Fine-tuned VividEmbedder
# ══════════════════════════════════════════════════════════════════════════

_vivid_embed_loaded = False


def _ensure_vivid_embed():
    """Lazy-load VividEmbed class once."""
    global _vivid_embed_loaded
    if not _vivid_embed_loaded:
        from VividEmbed import VividEmbed  # noqa: F401
        _vivid_embed_loaded = True


class VividEmbedMemory:
    """Wrapper around VividEmbed for benchmark — mirrors EmbeddingMemory API."""

    def __init__(self):
        _ensure_vivid_embed()
        from VividEmbed import VividEmbed
        self.ve = VividEmbed(
            persist_dir=None,
            model_name=VIVID_MODEL_PATH,
        )

    def store(self, text, emotion="neutral", importance=6):
        self.ve.add(
            content=text,
            emotion=emotion,
            importance=importance,
            source="conversation",
        )

    def retrieve(self, query, mood="neutral", top_k=10):
        results = self.ve.query(text=query, top_k=top_k, mood=mood)
        return [r["content"] for r in results]

    @property
    def all_contents(self):
        return [e.content for e in self.ve._entries]


# ══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════

def load_qa_dataset():
    path = MEM2ACT_DIR / "Mem2ActBench" / "qa_dataset.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def build_session_index():
    path = MEM2ACT_DIR / "Mem2ActBench" / "toolmem_conversation.jsonl"
    idx = {}
    for line in path.open(encoding="utf-8"):
        sess = json.loads(line)
        for oci in sess.get("original_conversation_ids", []):
            idx[oci] = sess
    return idx


def get_sessions_for_qa(qa, si):
    seen, out = set(), []
    for src in qa["source_conversation_ids"]:
        s = si.get(src)
        if s and s["session_id"] not in seen:
            seen.add(s["session_id"])
            out.append(s)
    return out


# ══════════════════════════════════════════════════════════════════════════
#  SESSION INGESTION (identical data for both conditions)
# ══════════════════════════════════════════════════════════════════════════

def _parse_tool_args(args_raw):
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except (json.JSONDecodeError, TypeError):
            args = {}
    else:
        args = args_raw if isinstance(args_raw, dict) else {}
    if not isinstance(args, dict):
        args = {}
    return args


def store_sessions_embedding(emem, sessions):
    """Store session data into Letta-style embedding memory."""
    for sess in sessions:
        for turn in sess["turns"]:
            role = turn["role"]
            content = (turn.get("content", "") or "").strip()
            if role == "user" and content:
                emem.store(f"User said: {content[:300]}")
            tool_calls = turn.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = _parse_tool_args(fn.get("arguments", ""))
                    emem.store(f"Used tool {name} with args: {json.dumps(args)}")


def store_sessions_vividembed(vmem, sessions):
    """Store session data into VividEmbed — same passages, with emotion/importance."""
    for sess in sessions:
        for turn in sess["turns"]:
            role = turn["role"]
            content = (turn.get("content", "") or "").strip()
            if role == "user" and content:
                vmem.store(
                    f"User said: {content[:300]}",
                    emotion="neutral",
                    importance=6,
                )
            tool_calls = turn.get("tool_calls")
            if tool_calls:
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args = _parse_tool_args(fn.get("arguments", ""))
                    vmem.store(
                        f"Used tool {name} with args: {json.dumps(args)}",
                        emotion="neutral",
                        importance=7,
                    )


# ══════════════════════════════════════════════════════════════════════════
#  RETRIEVAL (with tool-first scan + implicit reference resolution)
# ══════════════════════════════════════════════════════════════════════════

_IMPLICIT_MARKERS = frozenset({
    "that", "those", "the", "same", "usual", "always", "usually",
    "my", "our", "we", "typical", "regular", "normal",
    "again", "previous", "earlier", "back",
})

_RESOLVE_PROMPT = """You are a memory lookup assistant. The user's query contains vague or implicit references like "that stock", "the usual", "my favorite", "back home", etc.

Given the query and the memory context below, identify EVERY specific entity, value, or name the user is implicitly referring to.

Return ONLY a comma-separated list of the resolved values. No explanation.
If nothing needs resolving, return: NONE"""


def _resolve_implicit_refs(llm, query, memory_context):
    if not memory_context:
        return []
    msgs = [
        {"role": "system", "content": _RESOLVE_PROMPT},
        {"role": "user", "content": f"Memory Context:\n{memory_context[:2000]}\n\nQuery: {query}\n\nResolved values:"},
    ]
    try:
        raw = generate(llm, msgs, max_tokens=100, temperature=0.0)
        raw = raw.strip().strip('"').strip()
        if not raw or raw.upper() == "NONE":
            return []
        return [v.strip() for v in raw.split(",") if v.strip()]
    except Exception:
        return []


def retrieve_embedding(emem, query, tool_schema, llm=None):
    """Retrieve from Letta-style embedding memory (mirrors benchmark_full.py)."""
    tool_name = tool_schema.get("name", "")
    search = f"{query} {tool_name} {tool_schema.get('description', '')[:200]}"
    passages = emem.retrieve(search, top_k=10)

    # Tool-first scan
    if tool_name:
        tool_lower = tool_name.lower()
        for p in emem.passages:
            if p not in passages and tool_lower in p.lower():
                passages.append(p)

    if not passages:
        return ""

    lines = ["## Retrieved Memories (Embedding/Letta-style)"]
    for p in passages[:15]:
        lines.append(f"- {p}")
    initial_ctx = "\n".join(lines)

    # Implicit reference resolution
    query_words = set(query.lower().split())
    has_implicit = bool(query_words & _IMPLICIT_MARKERS)
    if has_implicit and llm:
        resolved = _resolve_implicit_refs(llm, query, initial_ctx)
        if resolved:
            for term in resolved:
                extra = emem.retrieve(f"{term} {tool_name}", top_k=5)
                for p in extra:
                    marker = f"- {p}"
                    if marker not in initial_ctx:
                        lines.append(marker)
                term_lower = term.lower()
                for p in emem.passages:
                    marker = f"- {p}"
                    if marker not in initial_ctx and term_lower in p.lower():
                        lines.append(marker)

    return "\n".join(lines) if lines else ""


def retrieve_vividembed(vmem, query, tool_schema, llm=None):
    """Retrieve from VividEmbed memory (same shape as embedding retriever)."""
    tool_name = tool_schema.get("name", "")
    search = f"{query} {tool_name} {tool_schema.get('description', '')[:200]}"
    passages = vmem.retrieve(search, mood="neutral", top_k=10)

    # Tool-first scan
    if tool_name:
        tool_lower = tool_name.lower()
        for p in vmem.all_contents:
            if p not in passages and tool_lower in p.lower():
                passages.append(p)

    if not passages:
        return ""

    lines = ["## Retrieved Memories (VividEmbedder)"]
    for p in passages[:15]:
        lines.append(f"- {p}")
    initial_ctx = "\n".join(lines)

    # Implicit reference resolution
    query_words = set(query.lower().split())
    has_implicit = bool(query_words & _IMPLICIT_MARKERS)
    if has_implicit and llm:
        resolved = _resolve_implicit_refs(llm, query, initial_ctx)
        if resolved:
            for term in resolved:
                extra = vmem.retrieve(f"{term} {tool_name}", mood="neutral", top_k=5)
                for p in extra:
                    marker = f"- {p}"
                    if marker not in initial_ctx:
                        lines.append(marker)
                term_lower = term.lower()
                for p in vmem.all_contents:
                    marker = f"- {p}"
                    if marker not in initial_ctx and term_lower in p.lower():
                        lines.append(marker)

    return "\n".join(lines) if lines else ""


# ══════════════════════════════════════════════════════════════════════════
#  PROMPT & PARSE
# ══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are an AI assistant that helps users by calling tools. Given a user query, past memory context, and a target tool schema, you must generate the correct tool call with filled-in arguments.

CRITICAL RULES:
1. You must respond ONLY with a JSON object in this exact format:
   {"name": "<tool_name>", "arguments": {<key>: <value>, ...}}
2. ALWAYS use EXACT values from the memory context when filling arguments.
   - If memory says the tool was called with keyword='Bitcoin', use 'Bitcoin' — NOT 'crypto'.
   - If memory says text='Hytale', use 'Hytale' — NOT 'sandbox RPG'.
   - If memory says symbol='GOOGL', use 'GOOGL' — NOT 'Google' or 'GOOG'.
   - NEVER echo or paraphrase the user's query as an argument value.
   - NEVER substitute your own knowledge for values that appear in memory context.
3. When a user refers to something implicitly ("that stock", "the usual", "my favorite"),
   look up the SPECIFIC value from memory context and use it exactly as stored.
4. Include ALL required parameters from the schema, using memory context to fill values.

Do NOT include any other text, explanation, or markdown formatting. Just the raw JSON object."""


def build_prompt(query, tool_schema, memory_context=""):
    schema_text = json.dumps(tool_schema, indent=2, ensure_ascii=False)
    user = f"Target Tool Schema:\n{schema_text}\n\n"
    if memory_context:
        user += f"{memory_context}\n\n"
    user += f"User Query: {query}\n\nGenerate the tool call JSON:"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_tool_call(raw):
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    match = re.search(r'\{[^{}]*"name"\s*:.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end + 1])
        except json.JSONDecodeError:
            pass
    return None


# ══════════════════════════════════════════════════════════════════════════
#  METRICS
# ══════════════════════════════════════════════════════════════════════════

def normalize_value(v):
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return v.strip().lower()
    return json.dumps(v, sort_keys=True).lower()


def compute_tool_accuracy(pred, gold):
    if pred is None:
        return 0.0
    if pred.get("name", "").strip().lower() != gold["name"].strip().lower():
        return 0.0
    for k, v in gold.get("arguments", {}).items():
        if k not in pred.get("arguments", {}):
            return 0.0
        if normalize_value(pred["arguments"][k]) != normalize_value(v):
            return 0.0
    return 1.0


def compute_f1(pred, gold):
    if pred is None:
        return 0.0, 0.0, 0.0
    gold_pairs = {(k, normalize_value(v)) for k, v in gold.get("arguments", {}).items()}
    pred_pairs = {(k, normalize_value(v)) for k, v in (pred.get("arguments", {}) or {}).items()}
    if not gold_pairs and not pred_pairs:
        return 1.0, 1.0, 1.0
    if not pred_pairs or not gold_pairs:
        return 0.0, 0.0, 0.0
    tp = len(gold_pairs & pred_pairs)
    prec = tp / len(pred_pairs)
    rec = tp / len(gold_pairs)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def compute_bleu1(pred, gold):
    if pred is None:
        return 0.0
    gold_args = gold.get("arguments", {})
    pred_args = pred.get("arguments", {}) if pred else {}
    gold_tok, pred_tok = [], []
    for k, v in sorted(gold_args.items()):
        gold_tok.extend(str(k).lower().split())
        gold_tok.extend(str(v).lower().split())
    for k, v in sorted(pred_args.items()):
        pred_tok.extend(str(k).lower().split())
        pred_tok.extend(str(v).lower().split())
    if not gold_tok or not pred_tok:
        return 0.0
    gc = Counter(gold_tok)
    pc = Counter(pred_tok)
    clipped = sum(min(pc[w], gc[w]) for w in pc)
    prec = clipped / sum(pc.values())
    bp = min(1.0, len(pred_tok) / len(gold_tok))
    return bp * prec


# ══════════════════════════════════════════════════════════════════════════
#  EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════

def run_condition(llm, qa_items, session_index, condition, max_eval, seed):
    """Run one condition with a specific random seed."""

    # Shuffle with seed for reproducibility
    shuffled = list(qa_items)
    random.seed(seed)
    random.shuffle(shuffled)
    items = shuffled[:max_eval]

    print(f"\n  {condition} | seed={seed} | n={len(items)}")

    metrics = {"tool_accuracy": [], "f1": [], "precision": [], "recall": [], "bleu1": []}
    level_metrics = {}
    predictions = []

    for i, qa in enumerate(tqdm(items, desc=f"{condition}-s{seed}")):
        query = qa["query"]
        gold = qa["tool_call"]
        schema = qa["target_tool_schema"]
        level = qa["complexity_metadata"]["level"]
        sessions = get_sessions_for_qa(qa, session_index)

        if condition == "embedding":
            emem = EmbeddingMemory()
            store_sessions_embedding(emem, sessions)
            memory_ctx = retrieve_embedding(emem, query, schema, llm=llm)
        elif condition == "vividembed":
            vmem = VividEmbedMemory()
            store_sessions_vividembed(vmem, sessions)
            memory_ctx = retrieve_vividembed(vmem, query, schema, llm=llm)
        else:
            memory_ctx = ""

        msgs = build_prompt(query, schema, memory_ctx)
        raw = generate(llm, msgs)
        pred = parse_tool_call(raw)

        ta = compute_tool_accuracy(pred, gold)
        prec, rec, f1 = compute_f1(pred, gold)
        b1 = compute_bleu1(pred, gold)

        metrics["tool_accuracy"].append(ta)
        metrics["f1"].append(f1)
        metrics["precision"].append(prec)
        metrics["recall"].append(rec)
        metrics["bleu1"].append(b1)

        if level not in level_metrics:
            level_metrics[level] = {"tool_accuracy": [], "f1": [], "bleu1": []}
        level_metrics[level]["tool_accuracy"].append(ta)
        level_metrics[level]["f1"].append(f1)
        level_metrics[level]["bleu1"].append(b1)

        predictions.append({
            "qa_id": qa["qa_id"], "level": level, "query": query,
            "gold_tool": gold["name"], "gold_args": gold["arguments"],
            "pred_raw": raw[:500], "pred_parsed": pred,
            "tool_accuracy": ta, "f1": f1, "bleu1": b1,
        })

        if (i + 1) % 25 == 0:
            avg_ta = sum(metrics["tool_accuracy"]) / len(metrics["tool_accuracy"])
            avg_f1 = sum(metrics["f1"]) / len(metrics["f1"])
            print(f"    [{i+1}/{len(items)}] TA={avg_ta:.3f} F1={avg_f1:.3f}")

    n = len(metrics["tool_accuracy"])
    report = {
        "condition": condition,
        "seed": seed,
        "n_evaluated": n,
        "timestamp": datetime.now().isoformat(),
        "overall": {k: sum(v) / max(n, 1) for k, v in metrics.items()},
        "per_level": {},
    }
    for lvl, lm in sorted(level_metrics.items()):
        ln = len(lm["tool_accuracy"])
        report["per_level"][lvl] = {
            "n": ln,
            "tool_accuracy": sum(lm["tool_accuracy"]) / ln,
            "f1": sum(lm["f1"]) / ln,
            "bleu1": sum(lm["bleu1"]) / ln,
        }
    return report, predictions


# ══════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════

def print_seed_report(report):
    o = report["overall"]
    label = f"{report['condition']} (s={report['seed']})"
    print(f"    {label:<25} TA={o['tool_accuracy']:.4f}  F1={o['f1']:.4f}  BLEU1={o['bleu1']:.4f}")


def print_final_comparison(all_reports, seeds):
    """Print averaged comparison table."""
    print(f"\n{'='*75}")
    print(f"  EMBEDDING vs VIVIDEMBEDDER — Mem2ActBench")
    print(f"  Seeds: {seeds}")
    print(f"{'='*75}")

    # Group by condition
    by_cond = {}
    for r in all_reports:
        cond = r["condition"]
        by_cond.setdefault(cond, []).append(r)

    # Per-seed results
    print(f"\n  Per-Seed Results:")
    print(f"  {'Run':<30} {'TA':>8} {'F1':>8} {'BLEU1':>8}  {'n':>4}")
    print(f"  {'-'*62}")

    for cond in ["embedding", "vividembed"]:
        for r in by_cond.get(cond, []):
            o = r["overall"]
            label = f"{cond} (s={r['seed']})"
            print(f"  {label:<30} {o['tool_accuracy']:>8.4f} {o['f1']:>8.4f} {o['bleu1']:>8.4f}  {r['n_evaluated']:>4}")
        if cond in by_cond:
            print()

    # Averaged results
    print(f"  {'─'*62}")
    print(f"  Averaged Results (across {len(seeds)} seeds):")
    print(f"  {'Condition':<30} {'TA':>8} {'F1':>8} {'BLEU1':>8} {'Prec':>8} {'Rec':>8}")
    print(f"  {'-'*70}")

    # Previous baselines
    for prev_name, prev in PREV_BASELINES.items():
        label = f"[PREV] {prev_name}"
        print(f"  {label:<30} {prev['tool_accuracy']:>8.4f} {prev['f1']:>8.4f} {prev['bleu1']:>8.4f} {'':>8} {'':>8}  (n={prev['n']})")

    print(f"  {'-'*70}")

    avg_results = {}
    for cond in ["embedding", "vividembed"]:
        runs = by_cond.get(cond, [])
        if not runs:
            continue
        n_seeds = len(runs)
        avg_ta = sum(r["overall"]["tool_accuracy"] for r in runs) / n_seeds
        avg_f1 = sum(r["overall"]["f1"] for r in runs) / n_seeds
        avg_b1 = sum(r["overall"]["bleu1"] for r in runs) / n_seeds
        avg_prec = sum(r["overall"]["precision"] for r in runs) / n_seeds
        avg_rec = sum(r["overall"]["recall"] for r in runs) / n_seeds
        avg_n = runs[0]["n_evaluated"]

        # Standard deviation
        std_ta = (sum((r["overall"]["tool_accuracy"] - avg_ta)**2 for r in runs) / n_seeds) ** 0.5
        std_f1 = (sum((r["overall"]["f1"] - avg_f1)**2 for r in runs) / n_seeds) ** 0.5
        std_b1 = (sum((r["overall"]["bleu1"] - avg_b1)**2 for r in runs) / n_seeds) ** 0.5

        avg_results[cond] = {
            "tool_accuracy": avg_ta, "f1": avg_f1, "bleu1": avg_b1,
            "precision": avg_prec, "recall": avg_rec,
            "std_ta": std_ta, "std_f1": std_f1, "std_b1": std_b1,
        }

        label = f"[AVG] {cond}"
        print(f"  {label:<30} {avg_ta:>8.4f} {avg_f1:>8.4f} {avg_b1:>8.4f} {avg_prec:>8.4f} {avg_rec:>8.4f}  (n={avg_n})")
        label2 = f"  [STD] {cond}"
        print(f"  {label2:<30} {std_ta:>8.4f} {std_f1:>8.4f} {std_b1:>8.4f}")

    # Delta
    if "embedding" in avg_results and "vividembed" in avg_results:
        e = avg_results["embedding"]
        v = avg_results["vividembed"]
        dta = v["tool_accuracy"] - e["tool_accuracy"]
        df1 = v["f1"] - e["f1"]
        db1 = v["bleu1"] - e["bleu1"]
        print(f"\n  {'Delta (vivid − letta)':<30} {dta:>+8.4f} {df1:>+8.4f} {db1:>+8.4f}")
        if e["tool_accuracy"] > 0:
            pct_ta = dta / e["tool_accuracy"] * 100
            pct_f1 = df1 / e["f1"] * 100 if e["f1"] > 0 else 0
            pct_b1 = db1 / e["bleu1"] * 100 if e["bleu1"] > 0 else 0
            print(f"  {'Relative improvement':<30} {pct_ta:>+7.1f}% {pct_f1:>+7.1f}% {pct_b1:>+7.1f}%")

    # Per-level averaged comparison
    all_levels = sorted(set(
        lvl for r in all_reports for lvl in r["per_level"]
    ))
    if all_levels:
        print(f"\n  Per-Level Averaged Breakdown:")
        for lvl in all_levels:
            print(f"\n    {lvl}:")
            for cond in ["embedding", "vividembed"]:
                runs = by_cond.get(cond, [])
                lvl_runs = [r["per_level"][lvl] for r in runs if lvl in r["per_level"]]
                if lvl_runs:
                    avg_ta = sum(l["tool_accuracy"] for l in lvl_runs) / len(lvl_runs)
                    avg_f1 = sum(l["f1"] for l in lvl_runs) / len(lvl_runs)
                    avg_b1 = sum(l["bleu1"] for l in lvl_runs) / len(lvl_runs)
                    avg_n = lvl_runs[0]["n"]
                    print(f"      {cond:<20} TA={avg_ta:.4f}  F1={avg_f1:.4f}  B1={avg_b1:.4f}  (n={avg_n})")

    return avg_results


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Letta vs VividEmbedder — Mem2ActBench head-to-head")
    parser.add_argument("--max-eval", type=int, default=100,
                        help="Max QA items per seed (100 or 400)")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 123, 456, 789, 1337],
                        help="Random seeds (default: 5 seeds)")
    parser.add_argument("--conditions", nargs="+",
                        default=["embedding", "vividembed"],
                        choices=["embedding", "vividembed"],
                        help="Which conditions to run")
    args = parser.parse_args()

    print("Loading Mem2ActBench data...")
    qa_items = load_qa_dataset()
    session_index = build_session_index()
    print(f"  {len(qa_items)} QA, {len(session_index)} sessions")
    print(f"  Seeds: {args.seeds}")
    print(f"  Max eval per seed: {args.max_eval}")
    print(f"  Conditions: {args.conditions}")

    llm = load_llm()

    tag = datetime.now().strftime("%Y%m%d_%H%M")
    all_reports = []

    for seed in args.seeds:
        for cond in args.conditions:
            t0 = time.time()
            report, preds = run_condition(
                llm, qa_items, session_index, cond, args.max_eval, seed)
            elapsed = time.time() - t0
            all_reports.append(report)
            print_seed_report(report)
            print(f"      Time: {elapsed:.0f}s")

            # Save per-run results
            rp = RESULTS_DIR / f"EmbedBench_{cond}_s{seed}_{tag}.json"
            pp = RESULTS_DIR / f"EmbedBench_{cond}_s{seed}_{tag}_preds.json"
            rp.write_text(json.dumps(report, indent=2), encoding="utf-8")
            pp.write_text(json.dumps(preds, indent=2, ensure_ascii=False), encoding="utf-8")

    # Final comparison
    avg_results = print_final_comparison(all_reports, args.seeds)

    # Save combined report
    combined = {
        "benchmark": "EmbedBench_Letta_vs_VividEmbedder",
        "timestamp": datetime.now().isoformat(),
        "seeds": args.seeds,
        "max_eval": args.max_eval,
        "conditions": args.conditions,
        "model": "Gemma-3-12B-IT-Q4_K_M",
        "runs": all_reports,
        "averaged": avg_results,
        "previous_baselines": PREV_BASELINES,
    }
    cf = RESULTS_DIR / f"EmbedBench_combined_{tag}.json"
    cf.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\n  Combined report: {cf.name}")
    print("\nDone!")


if __name__ == "__main__":
    main()
