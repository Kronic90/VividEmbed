"""
train_vivid_model.py — Fine-tune an Embedding Model for Companion Memory
=========================================================================
Trains a purpose-built sentence-transformer where the geometry itself
encodes emotional congruence, contradiction awareness, mood-conditioned
retrieval, and vividness magnitude.

This is (as far as we know) the first embedding model designed
specifically for persistent AI companion memory.

Architecture
------------
  Base model : all-MiniLM-L6-v2  (22 M params, 384-d)
  Modification : final Normalize layer REMOVED  →  ||v|| encodes vividness
  Special tokens : [EMO:x], [MOOD:x], [IMP:n], [QUERY] added to vocab
  Training objectives (5):
    1. Emotional congruence   — TripletLoss (cosine)
    2. Semantic similarity    — MultipleNegativesRankingLoss
    3. Contradiction repulsion — TripletLoss (cosine)
    4. Mood-conditioned retrieval — TripletLoss (cosine)
    5. Vividness magnitude    — custom MagnitudeMSELoss on ||v||

Key insight: cosine-based losses train the DIRECTION of vectors
(semantic + emotion + mood), while the magnitude loss independently
trains the NORM (vividness).  These are orthogonal objectives — they
don't interfere with each other.

Usage
-----
  # 1. Generate training data first:
  python build_training_data.py

  # 2. Train the model:
  python train_vivid_model.py

  # 3. The fine-tuned model is saved to ./vivid_model_output/

Requirements
------------
  pip install sentence-transformers datasets
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from sentence_transformers import SentenceTransformer
from datasets import Dataset, load_from_disk

# ══════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════

BASE_DIR    = Path(__file__).parent
DATA_DIR    = BASE_DIR / "vivid_training_data"
OUTPUT_DIR  = BASE_DIR / "vivid_model_output"
BASE_MODEL  = "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════
# 1.  SPECIAL TOKENS
# ══════════════════════════════════════════════════════════════════

def _build_special_tokens() -> list[str]:
    """All conditioning tokens the model needs to learn."""
    manifest_path = DATA_DIR / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)["special_tokens"]

    # Fallback if manifest not found
    emotions = [
        "happy", "excited", "proud", "joyful", "loving", "grateful",
        "calm", "content", "hopeful", "nostalgic",
        "sad", "lonely", "melancholy", "disappointed", "guilty",
        "angry", "frustrated", "anxious", "stressed", "furious",
        "neutral", "curious",
    ]
    moods = [
        "happy", "sad", "angry", "anxious", "calm",
        "excited", "lonely", "hopeful", "nostalgic", "neutral",
    ]
    tokens = [f"[EMO:{e}]" for e in emotions]
    tokens += [f"[MOOD:{m}]" for m in moods]
    tokens += [f"[IMP:{i}]" for i in range(1, 11)]
    tokens += ["[QUERY]"]
    return tokens


# ══════════════════════════════════════════════════════════════════
# 2.  MODEL SETUP
# ══════════════════════════════════════════════════════════════════

def setup_model() -> SentenceTransformer:
    """Load base model, add special tokens, remove Normalize layer."""
    print(f"Loading base model: {BASE_MODEL}")
    model = SentenceTransformer(BASE_MODEL)

    # ── Remove final Normalize layer (so magnitude is free) ──
    # SentenceTransformer modules are ordered: 0=Transformer, 1=Pooling, [2=Normalize]
    module_keys = list(model._modules.keys())
    last_key = module_keys[-1]
    last_module = model._modules[last_key]

    # Check if it's a Normalize layer by name or type
    module_type = type(last_module).__name__
    if module_type == "Normalize":
        del model._modules[last_key]
        print(f"  Removed final Normalize layer (was module '{last_key}')")
        print(f"  → Vector magnitude is now free to encode vividness")
    else:
        print(f"  Note: last module is {module_type}, not Normalize — leaving as-is")

    # ── Add special tokens to tokenizer ──────────────────────
    special_tokens = _build_special_tokens()
    tokenizer = model.tokenizer
    num_added = tokenizer.add_tokens(special_tokens)
    print(f"  Added {num_added} special tokens to vocabulary")

    # Resize transformer embeddings to match new vocab size
    transformer = model[0]  # first module is the Transformer wrapper
    if hasattr(transformer, "auto_model"):
        transformer.auto_model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings: {len(tokenizer)} tokens")
    elif hasattr(transformer, "model"):
        transformer.model.resize_token_embeddings(len(tokenizer))
        print(f"  Resized embeddings: {len(tokenizer)} tokens")

    return model


# ══════════════════════════════════════════════════════════════════
# 3.  CUSTOM LOSS FUNCTIONS
# ══════════════════════════════════════════════════════════════════

class MagnitudeMSELoss(nn.Module):
    """Custom loss: train ||embedding|| to match a target vividness norm.

    Dataset format: {"text": str, "label": float}

    Targets are in the natural MiniLM norm range (~2-8):
      IMP:1 → 2.5,  IMP:5 → 4.5,  IMP:10 → 7.0

    The weight factor scales this loss down so it doesn't dominate
    the cosine-based objectives (which are typically 0.1-0.5).
    """

    def __init__(self, model: SentenceTransformer, weight: float = 0.05):
        super().__init__()
        self.model = model
        self.weight = weight

    def forward(self, sentence_features: list[dict], labels: torch.Tensor):
        reps = self.model(sentence_features[0])["sentence_embedding"]
        norms = torch.norm(reps, p=2, dim=1)
        return self.weight * F.mse_loss(norms, labels.float())


class CosineTripletLoss(nn.Module):
    """Triplet loss using cosine distance.

    Dataset format: {"anchor": str, "positive": str, "negative": str}

    margin: minimum gap between pos_dist and neg_dist.
    """

    def __init__(self, model: SentenceTransformer, margin: float = 0.3):
        super().__init__()
        self.model = model
        self.margin = margin

    def forward(self, sentence_features: list[dict], labels: torch.Tensor):
        rep_anchor = self.model(sentence_features[0])["sentence_embedding"]
        rep_pos    = self.model(sentence_features[1])["sentence_embedding"]
        rep_neg    = self.model(sentence_features[2])["sentence_embedding"]

        # Cosine distance = 1 - cosine_similarity
        cos_ap = F.cosine_similarity(rep_anchor, rep_pos)
        cos_an = F.cosine_similarity(rep_anchor, rep_neg)

        # We want cos_ap > cos_an + margin
        loss = F.relu(cos_an - cos_ap + self.margin)
        return loss.mean()


class ContrastivePairsLoss(nn.Module):
    """Multiple negatives ranking loss for (anchor, positive) pairs.

    Uses in-batch negatives: within a batch, every other positive
    becomes a negative for each anchor.

    Dataset format: {"anchor": str, "positive": str}
    """

    def __init__(self, model: SentenceTransformer, scale: float = 20.0):
        super().__init__()
        self.model = model
        self.scale = scale

    def forward(self, sentence_features: list[dict], labels: torch.Tensor):
        rep_anchor = self.model(sentence_features[0])["sentence_embedding"]
        rep_pos    = self.model(sentence_features[1])["sentence_embedding"]

        # Normalise for cosine sim
        a_norm = F.normalize(rep_anchor, p=2, dim=1)
        p_norm = F.normalize(rep_pos,    p=2, dim=1)

        # Similarity matrix: (batch, batch) — each anchor vs all positives
        scores = torch.mm(a_norm, p_norm.t()) * self.scale  # scaled cosine

        # Labels: diagonal (i-th anchor should match i-th positive)
        target = torch.arange(scores.size(0), device=scores.device)
        return F.cross_entropy(scores, target)


# ══════════════════════════════════════════════════════════════════
# 4.  DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_datasets() -> dict[str, Dataset]:
    """Load all training datasets from disk."""
    datasets = {}
    expected = [
        "emotion_triplets",
        "semantic_pairs",
        "contradiction_triplets",
        "mood_triplets",
        "factual_triplets",
        "magnitude_examples",
    ]
    for name in expected:
        path = DATA_DIR / name
        if path.exists():
            datasets[name] = load_from_disk(str(path))
            print(f"  Loaded {name}: {len(datasets[name]):,} examples")
        else:
            print(f"  WARNING: {name} not found at {path}")

    return datasets


# ══════════════════════════════════════════════════════════════════
# 5.  CUSTOM TRAINING LOOP
#     We use a custom loop instead of SentenceTransformerTrainer
#     because we need the MagnitudeMSELoss which has a non-standard
#     dataset format (single text + scalar label).
# ══════════════════════════════════════════════════════════════════

def _collate_triplet(batch: list[dict], tokenizer, max_length: int = 128):
    """Collate a batch of triplets → three sets of tokenized features."""
    anchors  = [item["anchor"]   for item in batch]
    positives = [item["positive"] for item in batch]
    negatives = [item["negative"] for item in batch]

    feat_a = tokenizer(anchors,   padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    feat_p = tokenizer(positives, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    feat_n = tokenizer(negatives, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")

    return [dict(feat_a), dict(feat_p), dict(feat_n)], torch.zeros(len(batch))


def _collate_pairs(batch: list[dict], tokenizer, max_length: int = 128):
    """Collate a batch of pairs → two sets of tokenized features."""
    anchors  = [item["anchor"]   for item in batch]
    positives = [item["positive"] for item in batch]

    feat_a = tokenizer(anchors,   padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")
    feat_p = tokenizer(positives, padding=True, truncation=True,
                       max_length=max_length, return_tensors="pt")

    return [dict(feat_a), dict(feat_p)], torch.zeros(len(batch))


def _collate_magnitude(batch: list[dict], tokenizer, max_length: int = 128):
    """Collate a batch of magnitude examples → features + scalar labels."""
    texts  = [item["text"]  for item in batch]
    labels = [item["label"] for item in batch]

    feat = tokenizer(texts, padding=True, truncation=True,
                     max_length=max_length, return_tensors="pt")

    return [dict(feat)], torch.tensor(labels, dtype=torch.float32)


class MultiObjectiveTrainer:
    """Round-robin trainer across multiple objectives."""

    def __init__(
        self,
        model: SentenceTransformer,
        datasets: dict[str, Dataset],
        losses: dict[str, nn.Module],
        collate_fns: dict[str, callable],
        batch_size: int = 16,
        epochs: int = 10,
        lr: float = 2e-5,
        warmup_fraction: float = 0.1,
        output_dir: str | Path = OUTPUT_DIR,
    ):
        self.model = model
        self.datasets = datasets
        self.losses = losses
        self.collate_fns = collate_fns
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.warmup_fraction = warmup_fraction
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n  Device: {self.device}")
        if self.device.type == "cuda":
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    def _make_batches(self, dataset: Dataset) -> list[list[dict]]:
        """Split dataset into batches of dicts."""
        data = list(dataset)
        random_indices = list(range(len(data)))
        import random
        random.shuffle(random_indices)
        batches = []
        for i in range(0, len(data), self.batch_size):
            batch_indices = random_indices[i:i + self.batch_size]
            if len(batch_indices) < 2:
                continue   # skip tiny batches
            batches.append([data[j] for j in batch_indices])
        return batches

    def train(self):
        """Run the multi-objective training loop."""
        print(f"\n{'='*60}")
        print(f"  TRAINING VividEmbedder")
        print(f"{'='*60}")
        print(f"  Objectives: {', '.join(self.datasets.keys())}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Learning rate: {self.lr}")
        print(f"{'='*60}\n")

        # Move model to device
        self.model.to(self.device)
        for loss_fn in self.losses.values():
            loss_fn.to(self.device)

        # Optimizer — only collect model params once (losses reference
        # the same model, so their .parameters() would double-count)
        all_params = list(self.model.parameters())
        optimizer = torch.optim.AdamW(all_params, lr=self.lr, weight_decay=0.01)

        # Estimate total steps for warmup schedule
        total_batches_per_epoch = sum(
            len(ds) // self.batch_size for ds in self.datasets.values()
        )
        total_steps = total_batches_per_epoch * self.epochs
        warmup_steps = int(total_steps * self.warmup_fraction)

        def lr_schedule(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

        global_step = 0
        best_avg_loss = float("inf")
        tokenizer = self.model.tokenizer

        for epoch in range(self.epochs):
            epoch_start = time.time()
            epoch_losses = {name: [] for name in self.datasets}

            # Create batches for each objective
            all_batches: list[tuple[str, list[dict]]] = []
            for name, ds in self.datasets.items():
                for batch in self._make_batches(ds):
                    all_batches.append((name, batch))

            # Shuffle the combined batch list (round-robin-ish)
            import random
            random.shuffle(all_batches)

            self.model.train()
            for batch_idx, (name, batch) in enumerate(all_batches):
                collate_fn = self.collate_fns[name]
                sentence_features, labels = collate_fn(
                    batch, tokenizer
                )

                # Move to device
                sentence_features = [
                    {k: v.to(self.device) for k, v in sf.items()}
                    for sf in sentence_features
                ]
                labels = labels.to(self.device)

                loss = self.losses[name](sentence_features, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

                epoch_losses[name].append(loss.item())
                global_step += 1

                if global_step % 100 == 0:
                    lr_now = scheduler.get_last_lr()[0]
                    recent = {
                        n: np.mean(v[-50:]) if v else 0
                        for n, v in epoch_losses.items()
                    }
                    loss_str = " | ".join(f"{n}: {v:.4f}" for n, v in recent.items())
                    print(f"  Step {global_step:,} | lr={lr_now:.2e} | {loss_str}")

            # Epoch summary
            elapsed = time.time() - epoch_start
            avg_loss = np.mean([
                np.mean(v) for v in epoch_losses.values() if v
            ])
            print(f"\n  Epoch {epoch+1}/{self.epochs}  —  "
                  f"avg loss: {avg_loss:.4f}  —  "
                  f"{elapsed:.0f}s")
            for name, vals in epoch_losses.items():
                if vals:
                    print(f"    {name:30s}: {np.mean(vals):.4f}")

            # Save best checkpoint
            if avg_loss < best_avg_loss:
                best_avg_loss = avg_loss
                self.model.save(str(self.output_dir / "best"))
                print(f"    ★ New best model saved (loss={avg_loss:.4f})")

            print()

        # Save final model
        self.model.save(str(self.output_dir / "final"))
        print(f"  Final model saved to: {self.output_dir / 'final'}")
        print(f"  Best model saved to:  {self.output_dir / 'best'}")
        print(f"  Total steps: {global_step:,}")

        return self.model


# ══════════════════════════════════════════════════════════════════
# 6.  EVALUATION
# ══════════════════════════════════════════════════════════════════

def evaluate_model(model: SentenceTransformer):
    """Quick evaluation of trained model properties."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION")
    print(f"{'='*60}\n")

    def _encode(texts: list[str]) -> np.ndarray:
        return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # ── Test 1: Emotional clustering ────────────────────────
    happy_mems = [
        "[EMO:happy] [IMP:7] Had a wonderful day at the park",
        "[EMO:happy] [IMP:8] The party was fantastic and everyone had fun",
        "[EMO:happy] [IMP:6] Got great news about my test results",
    ]
    sad_mems = [
        "[EMO:sad] [IMP:7] Lost a close friend and I'm heartbroken",
        "[EMO:sad] [IMP:8] The loneliness is getting unbearable",
        "[EMO:sad] [IMP:6] Found out about the bad news today",
    ]

    happy_vecs = _encode(happy_mems)
    sad_vecs   = _encode(sad_mems)

    intra_happy = np.mean([_cosine(happy_vecs[i], happy_vecs[j])
                           for i in range(3) for j in range(i+1, 3)])
    intra_sad   = np.mean([_cosine(sad_vecs[i], sad_vecs[j])
                           for i in range(3) for j in range(i+1, 3)])
    cross       = np.mean([_cosine(happy_vecs[i], sad_vecs[j])
                           for i in range(3) for j in range(3)])

    print(f"  Emotion clustering:")
    print(f"    Happy intra-sim:  {intra_happy:.4f}")
    print(f"    Sad intra-sim:    {intra_sad:.4f}")
    print(f"    Cross-emotion:    {cross:.4f}")
    gap = ((intra_happy + intra_sad) / 2) - cross
    print(f"    Cluster gap:      {gap:.4f}  {'✓' if gap > 0.05 else '✗'}\n")

    # ── Test 2: Contradiction awareness ─────────────────────
    love_job  = _encode(["I love my job and look forward to every day"])[0]
    hate_job  = _encode(["I hate my job and dread going to work"])[0]
    agree_job = _encode(["Work is wonderful and I'm excited about my career"])[0]

    sim_agree = _cosine(love_job, agree_job)
    sim_contra = _cosine(love_job, hate_job)
    print(f"  Contradiction awareness:")
    print(f"    Love-job ↔ agree:     {sim_agree:.4f}")
    print(f"    Love-job ↔ contradict:{sim_contra:.4f}")
    print(f"    Gap:                  {sim_agree - sim_contra:.4f}  "
          f"{'✓' if sim_agree > sim_contra else '✗'}\n")

    # ── Test 3: Mood conditioning ───────────────────────────
    query_sad   = _encode(["[MOOD:sad] [QUERY] what's been happening"])[0]
    query_happy = _encode(["[MOOD:happy] [QUERY] what's been happening"])[0]
    mem_sad     = _encode(["[EMO:sad] [IMP:7] The week has been really tough"])[0]
    mem_happy   = _encode(["[EMO:happy] [IMP:7] Everything has been going great"])[0]

    sim_sad_sad     = _cosine(query_sad, mem_sad)
    sim_sad_happy   = _cosine(query_sad, mem_happy)
    sim_happy_happy = _cosine(query_happy, mem_happy)
    sim_happy_sad   = _cosine(query_happy, mem_sad)

    print(f"  Mood conditioning:")
    print(f"    [MOOD:sad] query ↔ sad memory:    {sim_sad_sad:.4f}")
    print(f"    [MOOD:sad] query ↔ happy memory:  {sim_sad_happy:.4f}")
    print(f"    [MOOD:happy] query ↔ happy memory:{sim_happy_happy:.4f}")
    print(f"    [MOOD:happy] query ↔ sad memory:  {sim_happy_sad:.4f}")
    congruence = (sim_sad_sad - sim_sad_happy + sim_happy_happy - sim_happy_sad) / 2
    print(f"    Mood congruence gap: {congruence:.4f}  {'✓' if congruence > 0 else '✗'}\n")

    # ── Test 4: Magnitude ↔ importance ──────────────────────
    low_imp  = _encode(["[EMO:neutral] [IMP:2] Had cereal for breakfast"])[0]
    mid_imp  = _encode(["[EMO:neutral] [IMP:5] Finished a decent book"])[0]
    high_imp = _encode(["[EMO:neutral] [IMP:9] Got married today"])[0]

    norm_low  = np.linalg.norm(low_imp)
    norm_mid  = np.linalg.norm(mid_imp)
    norm_high = np.linalg.norm(high_imp)

    print(f"  Magnitude ↔ importance:")
    print(f"    ||IMP:2||: {norm_low:.4f}  (target ≈ 3.0)")
    print(f"    ||IMP:5||: {norm_mid:.4f}  (target ≈ 4.5)")
    print(f"    ||IMP:9||: {norm_high:.4f}  (target ≈ 6.5)")
    print(f"    Monotonic: {'✓' if norm_low < norm_mid < norm_high else '✗'}\n")

    # ── Test 5: Backward compatibility (no prefix) ──────────
    raw_a = _encode(["I went for a run in the park"])[0]
    raw_b = _encode(["Jogged through the park this morning"])[0]
    raw_c = _encode(["The stock market closed higher today"])[0]

    sim_ab = _cosine(raw_a, raw_b)
    sim_ac = _cosine(raw_a, raw_c)
    print(f"  Semantic quality (no prefix):")
    print(f"    Run ↔ Jog:    {sim_ab:.4f}")
    print(f"    Run ↔ Stock:  {sim_ac:.4f}")
    print(f"    Coherent:     {'✓' if sim_ab > sim_ac else '✗'}\n")

    # ── Test 6: Factual retrieval (tool-call matching) ──────
    query_fact = _encode(["[MOOD:neutral] [QUERY] what stock did I look up"])[0]
    tool_right = _encode(["[EMO:neutral] [IMP:7] Used tool get_stock_price with args: {\"symbol\": \"AAPL\"}"])[0]
    tool_wrong = _encode(["[EMO:neutral] [IMP:7] Used tool play_music with args: {\"track\": \"Yesterday\"}"])[0]
    user_noise = _encode(["[EMO:neutral] [IMP:6] User said: I'm worried about the stock market"])[0]

    sim_right = _cosine(query_fact, tool_right)
    sim_wrong = _cosine(query_fact, tool_wrong)
    sim_noise = _cosine(query_fact, user_noise)

    print(f"  Factual retrieval (tool-call matching):")
    print(f"    Query ↔ correct tool:  {sim_right:.4f}")
    print(f"    Query ↔ wrong tool:    {sim_wrong:.4f}")
    print(f"    Query ↔ noisy user:    {sim_noise:.4f}")
    fact_gap = sim_right - max(sim_wrong, sim_noise)
    print(f"    Gap (right − best distractor): {fact_gap:.4f}  {'✓' if fact_gap > 0 else '✗'}\n")


# ══════════════════════════════════════════════════════════════════
# 7.  MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    if not DATA_DIR.exists():
        print(f"Training data not found at {DATA_DIR}")
        print("Run build_training_data.py first!")
        sys.exit(1)

    # Load model
    model = setup_model()

    # Load datasets
    print("\nLoading datasets:")
    datasets = load_datasets()
    if not datasets:
        print("No datasets found!")
        sys.exit(1)

    # Setup loss functions and collate functions
    losses = {}
    collate_fns = {}

    if "emotion_triplets" in datasets:
        losses["emotion_triplets"] = CosineTripletLoss(model, margin=0.3)
        collate_fns["emotion_triplets"] = _collate_triplet

    if "semantic_pairs" in datasets:
        losses["semantic_pairs"] = ContrastivePairsLoss(model, scale=20.0)
        collate_fns["semantic_pairs"] = _collate_pairs

    if "contradiction_triplets" in datasets:
        losses["contradiction_triplets"] = CosineTripletLoss(model, margin=0.5)
        collate_fns["contradiction_triplets"] = _collate_triplet

    if "mood_triplets" in datasets:
        losses["mood_triplets"] = CosineTripletLoss(model, margin=0.3)
        collate_fns["mood_triplets"] = _collate_triplet

    if "factual_triplets" in datasets:
        losses["factual_triplets"] = CosineTripletLoss(model, margin=0.4)
        collate_fns["factual_triplets"] = _collate_triplet

    if "magnitude_examples" in datasets:
        losses["magnitude_examples"] = MagnitudeMSELoss(model, weight=0.05)
        collate_fns["magnitude_examples"] = _collate_magnitude

    # Train
    trainer = MultiObjectiveTrainer(
        model=model,
        datasets=datasets,
        losses=losses,
        collate_fns=collate_fns,
        batch_size=16,
        epochs=3,
        lr=2e-5,
        warmup_fraction=0.1,
        output_dir=OUTPUT_DIR,
    )
    trained_model = trainer.train()

    # Evaluate
    trained_model.eval()
    with torch.no_grad():
        evaluate_model(trained_model)

    print("\nDone! Model saved to:", OUTPUT_DIR)
    print("\nTo use in VividEmbed:")
    print('  ve = VividEmbed(model_name="./vivid_model_output/best")')


if __name__ == "__main__":
    main()
