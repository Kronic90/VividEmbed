"""
VividEmbed  —  Emotion-Vividness-Aware Semantic Embeddings
==========================================================
Drop-in retrieval backend for VividnessMem that replaces pure-lexical
matching with *hybrid* embeddings:  dense semantic vectors enriched with
emotion (PAD), importance, and temporal-decay dimensions.

Architecture
------------
    base_embedding (384-d, all-MiniLM-L6-v2)
  + emotion_pad    (3-d, Pleasure-Arousal-Dominance)
  + importance_dim (1-d, normalised 0-1)
  + stability_dim  (1-d, normalised 0-1)
  = VividVector    (389-d)

At query time the raw cosine similarity on the 389-d vector is further
modulated by:

    1. **Vividness decay**  — faded memories score lower, just like
       human recall.  `score *= vividness_weight`
    2. **Mood congruence**  — current mood biases toward emotionally
       matching memories.  `score *= (1 + 0.15 * mood_dot)`
    3. **Recency nudge**    — a tiny bonus for fresh memories to break
       ties, mimicking availability heuristic.

The result is a retrieval system that "remembers" the way humans do:
vivid, emotionally resonant, recent memories surface first — unless an
old memory is so semantically relevant it overcomes the decay.

Usage
-----
    from VividEmbed import VividEmbed

    ve = VividEmbed()                       # loads model on first call
    ve.add("I love morning runs", emotion="happy", importance=7)
    ve.add("The meeting was brutal", emotion="frustrated", importance=5)

    results = ve.query("exercise routine", mood=("excited",), top_k=3)
    for text, score in results:
        print(f"  {score:.3f}  {text}")

Integration with VividnessMem
-----------------------------
    from VividnessMem import VividnessMem
    from VividEmbed  import VividEmbed

    mem = VividnessMem("Luna")
    mem.embed_backend = VividEmbed()        # plug in as retrieval layer

(Full integration hooks are in VividnessMem >= 1.0.7.)

Requirements
------------
    pip install sentence-transformers numpy
"""

from __future__ import annotations

import math
import time
import hashlib
import json
import os
import struct
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# ──────────────────────────────────────────────────────────────────
# Lazy model loading — the transformer is only downloaded / loaded
# when the first embedding is actually requested.
# ──────────────────────────────────────────────────────────────────
_MODEL_NAME = "all-MiniLM-L6-v2"
_BASE_DIM   = 384          # output dim of MiniLM-L6-v2
_PAD_DIM    = 3            # Pleasure-Arousal-Dominance
_META_DIMS  = 2            # importance (1) + stability (1)
EMBED_DIM   = _BASE_DIM + _PAD_DIM + _META_DIMS   # 389

_model = None              # singleton SentenceTransformer


def _get_model():
    """Lazy-load the sentence-transformer model (thread-safe enough)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ──────────────────────────────────────────────────────────────────
# PAD emotion vectors  (copied from VividnessMem for standalone use,
# but will import from VividnessMem when used together)
# ──────────────────────────────────────────────────────────────────
EMOTION_VECTORS: dict[str, tuple[float, float, float]] = {
    # ─── Positive-Calm ───
    "content":      ( 0.7, -0.3,  0.3),
    "peaceful":     ( 0.8, -0.5,  0.4),
    "serene":       ( 0.8, -0.6,  0.3),
    "grateful":     ( 0.8,  0.1,  0.4),
    "warm":         ( 0.7,  0.1,  0.3),
    "relieved":     ( 0.6, -0.4,  0.3),
    "calm":         ( 0.5, -0.5,  0.3),
    "relaxed":      ( 0.6, -0.5,  0.3),
    "comfortable":  ( 0.6, -0.3,  0.3),
    "satisfied":    ( 0.7, -0.2,  0.4),
    "nostalgic":    ( 0.3, -0.2,  0.0),
    "tender":       ( 0.7,  0.0,  0.1),
    "hopeful":      ( 0.6,  0.2,  0.3),
    "compassionate":( 0.7,  0.1,  0.2),
    "gentle":       ( 0.6, -0.3,  0.1),
    # ─── Positive-Active ───
    "happy":        ( 0.8,  0.4,  0.5),
    "joyful":       ( 0.9,  0.6,  0.5),
    "excited":      ( 0.7,  0.8,  0.5),
    "proud":        ( 0.7,  0.4,  0.7),
    "motivated":    ( 0.6,  0.5,  0.6),
    "amused":       ( 0.7,  0.5,  0.4),
    "playful":      ( 0.7,  0.6,  0.4),
    "enthusiastic": ( 0.7,  0.7,  0.5),
    "inspired":     ( 0.7,  0.5,  0.5),
    "triumphant":   ( 0.8,  0.7,  0.8),
    "confident":    ( 0.6,  0.3,  0.7),
    "loving":       ( 0.9,  0.4,  0.4),
    "affectionate": ( 0.8,  0.3,  0.3),
    "elated":       ( 0.9,  0.7,  0.6),
    "curious":      ( 0.4,  0.5,  0.3),
    "fascinated":   ( 0.5,  0.6,  0.3),
    "surprised":    ( 0.3,  0.7,  0.0),
    "interested":   ( 0.4,  0.4,  0.3),
    "determined":   ( 0.4,  0.5,  0.7),
    # ─── Negative-Low Arousal ───
    "sad":          (-0.6, -0.3, -0.3),
    "lonely":       (-0.7, -0.4, -0.5),
    "guilty":       (-0.5,  0.1, -0.6),
    "ashamed":      (-0.6,  0.2, -0.7),
    "melancholy":   (-0.4, -0.4, -0.2),
    "bored":        (-0.3, -0.6, -0.2),
    "tired":        (-0.2, -0.6, -0.2),
    "disappointed": (-0.5, -0.1, -0.3),
    "resigned":     (-0.4, -0.4, -0.4),
    "empty":        (-0.5, -0.5, -0.5),
    "numb":         (-0.3, -0.6, -0.4),
    "regretful":    (-0.5,  0.0, -0.4),
    "wistful":      (-0.1, -0.3,  0.0),
    "vulnerable":   (-0.3,  0.2, -0.6),
    "helpless":     (-0.6,  0.1, -0.8),
    # ─── Negative-High Arousal ───
    "anxious":      (-0.5,  0.7, -0.4),
    "angry":        (-0.7,  0.8,  0.2),
    "afraid":       (-0.7,  0.8, -0.6),
    "frustrated":   (-0.6,  0.6, -0.1),
    "irritated":    (-0.5,  0.5,  0.0),
    "jealous":      (-0.5,  0.6, -0.3),
    "disgusted":    (-0.6,  0.4,  0.1),
    "overwhelmed":  (-0.4,  0.7, -0.5),
    "stressed":     (-0.5,  0.7, -0.3),
    "panicked":     (-0.7,  0.9, -0.7),
    "furious":      (-0.8,  0.9,  0.3),
    "bitter":       (-0.6,  0.3, -0.1),
    "resentful":    (-0.6,  0.4, -0.1),
    "hostile":      (-0.7,  0.7,  0.3),
    "contemptuous": (-0.6,  0.3,  0.4),
    "envious":      (-0.5,  0.5, -0.3),
    # ─── Neutral / Complex ───
    "neutral":      ( 0.0,  0.0,  0.0),
    "conflicted":   ( 0.0,  0.4, -0.2),
    "ambivalent":   ( 0.0,  0.1, -0.1),
    "pensive":      ( 0.1, -0.2,  0.1),
    "contemplative":( 0.2, -0.2,  0.2),
    "bittersweet":  ( 0.1,  0.1, -0.1),
}


def _emotion_to_pad(emotion: str) -> np.ndarray:
    """Convert an emotion label to a PAD vector (3-d numpy array)."""
    if not emotion:
        return np.zeros(3, dtype=np.float32)
    key = emotion.lower().strip()
    if key in EMOTION_VECTORS:
        return np.array(EMOTION_VECTORS[key], dtype=np.float32)
    # Fuzzy prefix match
    for k, v in EMOTION_VECTORS.items():
        if key.startswith(k) or k.startswith(key):
            return np.array(v, dtype=np.float32)
    return np.zeros(3, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────
# VividEntry  —  one indexed memory
# ──────────────────────────────────────────────────────────────────
@dataclass
class VividEntry:
    """A single memory stored in the VividEmbed index."""
    content:    str
    emotion:    str            = "neutral"
    importance: int            = 5          # 1-10
    stability:  float          = 3.0        # days (spaced-rep)
    timestamp:  str            = ""         # ISO-8601
    source:     str            = "reflection"
    entity:     str            = ""
    vector:     np.ndarray     = field(default_factory=lambda: np.zeros(EMBED_DIM, dtype=np.float32))
    uid:        str            = ""         # unique id (content hash)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.uid:
            self.uid = hashlib.sha256(
                (self.content + self.timestamp).encode()
            ).hexdigest()[:16]

    # ── Age & vividness (mirroring VividnessMem) ─────────────
    @property
    def age_days(self) -> float:
        dt = datetime.fromisoformat(self.timestamp)
        return (datetime.now() - dt).total_seconds() / 86400.0

    @property
    def vividness(self) -> float:
        """importance × exp(-age / stability)"""
        return self.importance * math.exp(
            -self.age_days / max(self.stability, 0.1)
        )

    def mood_adjusted_vividness(self, mood_pad: np.ndarray) -> float:
        """Vividness with mood-congruence boost (±15 %)."""
        base = self.vividness
        mem_pad = _emotion_to_pad(self.emotion)
        if np.allclose(mem_pad, 0) or np.allclose(mood_pad, 0):
            return base
        dot = float(np.dot(mem_pad, mood_pad))
        clamped = max(-1.0, min(1.0, dot))
        # Reappraisal cap for negative-valence
        if mem_pad[0] < 0 and mood_pad[0] < 0 and clamped > 0:
            clamped = min(clamped, 0.10) * math.exp(
                -self.age_days / (14.0 / math.log(2))
            )
        return base * (1.0 + 0.15 * clamped)

    # ── Serialisation ────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "content":    self.content,
            "emotion":    self.emotion,
            "importance": self.importance,
            "stability":  round(self.stability, 4),
            "timestamp":  self.timestamp,
            "source":     self.source,
            "entity":     self.entity,
            "uid":        self.uid,
        }

    @classmethod
    def from_dict(cls, d: dict, vector: np.ndarray | None = None) -> "VividEntry":
        entry = cls(
            content    = d["content"],
            emotion    = d.get("emotion", "neutral"),
            importance = d.get("importance", 5),
            stability  = d.get("stability", 3.0),
            timestamp  = d.get("timestamp", ""),
            source     = d.get("source", "reflection"),
            entity     = d.get("entity", ""),
            uid        = d.get("uid", ""),
        )
        if vector is not None:
            entry.vector = vector
        return entry


# ──────────────────────────────────────────────────────────────────
# VividEmbed  —  the main class
# ──────────────────────────────────────────────────────────────────

# Weighting knobs for the composite score
_W_SEMANTIC   = 0.45       # weight of pure cosine similarity
_W_VIVIDNESS  = 0.20       # weight of vividness (decayed importance)
_W_MOOD       = 0.20       # weight of mood-congruence boost
_W_RECENCY    = 0.15       # weight of recency nudge

# How strongly PAD + meta dims contribute vs the base text embedding
_PAD_SCALE    = 0.45       # scale factor for emotion dims in the vector
_META_SCALE   = 0.20       # scale factor for importance/stability dims

# Recency half-life in days (for the availability-heuristic nudge)
_RECENCY_HALFLIFE = 5.0


class VividEmbed:
    """Emotion-vividness-aware semantic embedding index.

    Parameters
    ----------
    persist_dir : str or Path, optional
        Directory to save/load the index.  ``None`` = in-memory only.
    model_name : str
        HuggingFace model id.  Default: ``all-MiniLM-L6-v2``.
    weights : dict, optional
        Override scoring weights.  Keys: ``semantic``, ``vividness``,
        ``mood``, ``recency``.
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        model_name: str = _MODEL_NAME,
        weights: dict[str, float] | None = None,
    ):
        global _MODEL_NAME
        _MODEL_NAME = model_name

        # Detect fine-tuned VividEmbedder (has [EMO:*] special tokens,
        # no Normalize layer, magnitude encodes vividness).
        self._is_vivid_model: bool = False
        self._embed_dim: int = EMBED_DIM  # 384 for VividEmbedder, 389 vanilla

        # Scoring weights (normalised to sum=1)
        w = weights or {}
        self._w_sem  = w.get("semantic",  _W_SEMANTIC)
        self._w_viv  = w.get("vividness", _W_VIVIDNESS)
        self._w_mood = w.get("mood",      _W_MOOD)
        self._w_rec  = w.get("recency",   _W_RECENCY)
        total = self._w_sem + self._w_viv + self._w_mood + self._w_rec
        if total > 0:
            self._w_sem  /= total
            self._w_viv  /= total
            self._w_mood /= total
            self._w_rec  /= total

        # Index storage
        self._entries: list[VividEntry] = []
        self._uid_set: set[str] = set()

        # Persistence
        self._persist_dir: Path | None = None
        if persist_dir is not None:
            self._persist_dir = Path(persist_dir)
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            self._load()

    # ──────────────────────────────────────────────────────────
    # VividEmbedder detection + encoding helpers
    # ──────────────────────────────────────────────────────────

    def _detect_vivid_model(self):
        """Check (once) if the loaded model is a fine-tuned VividEmbedder
        by looking for [EMO:happy] in the tokenizer vocabulary."""
        model = _get_model()
        tok = model.tokenizer
        self._is_vivid_model = "[EMO:happy]" in tok.get_vocab()
        self._embed_dim = _BASE_DIM if self._is_vivid_model else EMBED_DIM

    def _encode_memory(
        self,
        text: str,
        emotion: str = "neutral",
        importance: int = 5,
    ) -> np.ndarray:
        """Encode a memory for storage.

        VividEmbedder: prepends [EMO:x] [IMP:n] tokens, returns
                       un-normalised 384-d (‖v‖ ∝ importance).
        Vanilla model: returns L2-normalised 384-d base embedding.
        """
        if not hasattr(self, "_vivid_checked"):
            self._detect_vivid_model()
            self._vivid_checked = True

        model = _get_model()
        if self._is_vivid_model:
            tagged = f"[EMO:{emotion}] [IMP:{importance}] {text}"
            return np.array(
                model.encode(tagged, normalize_embeddings=False,
                             show_progress_bar=False),
                dtype=np.float32,
            )
        else:
            return np.array(
                model.encode(text, normalize_embeddings=True,
                             show_progress_bar=False),
                dtype=np.float32,
            )

    def _encode_query(
        self,
        text: str,
        mood: str = "neutral",
    ) -> np.ndarray:
        """Encode a retrieval query.

        VividEmbedder: prepends [MOOD:x] [QUERY] tokens.
        Vanilla model: returns L2-normalised 384-d base embedding.
        """
        if not hasattr(self, "_vivid_checked"):
            self._detect_vivid_model()
            self._vivid_checked = True

        model = _get_model()
        if self._is_vivid_model:
            mood_label = mood if isinstance(mood, str) else "neutral"
            tagged = f"[MOOD:{mood_label}] [QUERY] {text}"
            return np.array(
                model.encode(tagged, normalize_embeddings=False,
                             show_progress_bar=False),
                dtype=np.float32,
            )
        else:
            return np.array(
                model.encode(text, normalize_embeddings=True,
                             show_progress_bar=False),
                dtype=np.float32,
            )

    # ──────────────────────────────────────────────────────────
    # Public API — Add
    # ──────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        emotion: str = "neutral",
        importance: int = 5,
        stability: float = 3.0,
        source: str = "reflection",
        entity: str = "",
        timestamp: str = "",
    ) -> VividEntry:
        """Embed and index a new memory.

        Returns the created VividEntry.
        """
        entry = VividEntry(
            content    = content,
            emotion    = emotion,
            importance = max(1, min(10, importance)),
            stability  = max(0.1, stability),
            source     = source,
            entity     = entity,
            timestamp  = timestamp or datetime.now().isoformat(),
        )
        entry.vector = self._build_vector(entry)

        if entry.uid in self._uid_set:
            # Update existing entry instead of duplicating
            for i, e in enumerate(self._entries):
                if e.uid == entry.uid:
                    self._entries[i] = entry
                    break
        else:
            self._entries.append(entry)
            self._uid_set.add(entry.uid)

        return entry

    def add_batch(
        self,
        items: list[dict],
    ) -> list[VividEntry]:
        """Add multiple memories at once (batched embedding for speed).

        Each dict in *items* should have keys matching ``add()`` params:
        ``content``, ``emotion``, ``importance``, etc.
        """
        if not items:
            return []

        # Build entries (without vectors)
        entries: list[VividEntry] = []
        for d in items:
            entry = VividEntry(
                content    = d["content"],
                emotion    = d.get("emotion", "neutral"),
                importance = max(1, min(10, d.get("importance", 5))),
                stability  = max(0.1, d.get("stability", 3.0)),
                source     = d.get("source", "reflection"),
                entity     = d.get("entity", ""),
                timestamp  = d.get("timestamp", "") or datetime.now().isoformat(),
            )
            entries.append(entry)

        # Batch-encode all texts at once
        if not hasattr(self, "_vivid_checked"):
            self._detect_vivid_model()
            self._vivid_checked = True

        if self._is_vivid_model:
            # VividEmbedder: each text gets its own [EMO:x] [IMP:n] tags
            for entry in entries:
                entry.vector = self._encode_memory(
                    entry.content,
                    emotion=entry.emotion,
                    importance=entry.importance,
                )
                if entry.uid not in self._uid_set:
                    self._entries.append(entry)
                    self._uid_set.add(entry.uid)
        else:
            texts = [e.content for e in entries]
            base_vecs = _get_model().encode(texts, normalize_embeddings=True,
                                             show_progress_bar=False)
            for entry, base in zip(entries, base_vecs):
                entry.vector = self._assemble_vector(
                    base_embedding = np.array(base, dtype=np.float32),
                    emotion        = entry.emotion,
                    importance     = entry.importance,
                    stability      = entry.stability,
                )
                if entry.uid not in self._uid_set:
                    self._entries.append(entry)
                    self._uid_set.add(entry.uid)

        return entries

    # ──────────────────────────────────────────────────────────
    # Public API — Query
    # ──────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        top_k: int = 5,
        mood: str | tuple[float, float, float] = "neutral",
        entity_filter: str | None = None,
        source_filter: str | None = None,
        min_importance: int = 0,
        include_vector: bool = False,
    ) -> list[dict]:
        """Retrieve the most relevant memories for a query.

        Parameters
        ----------
        text : str
            The query text.
        top_k : int
            Max results to return.
        mood : str or (float, float, float)
            Current mood — either a label ("happy") or a raw PAD tuple.
        entity_filter : str, optional
            Only return memories about this entity.
        source_filter : str, optional
            Only return memories from this source.
        min_importance : int
            Floor on importance (0 = no filter).
        include_vector : bool
            If True, include the raw 389-d vector in results.

        Returns
        -------
        list of dict
            Each dict has: ``content``, ``score``, ``emotion``,
            ``importance``, ``vividness``, ``age_days``, ``uid``,
            and optionally ``vector``.
        """
        if not self._entries:
            return []

        # Resolve mood to PAD vector
        if isinstance(mood, str):
            mood_pad = _emotion_to_pad(mood)
            mood_label = mood
        else:
            mood_pad = np.array(mood, dtype=np.float32)
            mood_label = "neutral"

        # Build query vector
        if not hasattr(self, "_vivid_checked"):
            self._detect_vivid_model()
            self._vivid_checked = True

        if self._is_vivid_model:
            q_vec = self._encode_query(text, mood=mood_label)
        else:
            q_base = _get_model().encode(text, normalize_embeddings=True,
                                          show_progress_bar=False)
            q_vec = self._assemble_vector(
                base_embedding = np.array(q_base, dtype=np.float32),
                emotion        = "neutral",
                importance     = 5,
                stability      = 3.0,
            )

        # Score every entry
        scored: list[tuple[float, VividEntry]] = []
        for entry in self._entries:
            # Filters
            if entity_filter and entry.entity.lower() != entity_filter.lower():
                continue
            if source_filter and entry.source.lower() != source_filter.lower():
                continue
            if entry.importance < min_importance:
                continue

            score = self._score(q_vec, entry, mood_pad)
            scored.append((score, entry))

        # Sort descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Build results
        results: list[dict] = []
        for score, entry in scored[:top_k]:
            r = {
                "content":    entry.content,
                "score":      round(score, 4),
                "emotion":    entry.emotion,
                "importance": entry.importance,
                "vividness":  round(entry.vividness, 4),
                "age_days":   round(entry.age_days, 2),
                "source":     entry.source,
                "entity":     entry.entity,
                "uid":        entry.uid,
            }
            if include_vector:
                r["vector"] = entry.vector.tolist()
            results.append(r)

        return results

    def query_by_emotion(
        self,
        emotion: str,
        top_k: int = 5,
        min_importance: int = 0,
    ) -> list[dict]:
        """Find memories closest to a given emotion in PAD space.

        Useful for "what memories feel like this?" searches.
        """
        target_pad = _emotion_to_pad(emotion)
        if np.allclose(target_pad, 0):
            return []

        scored: list[tuple[float, VividEntry]] = []
        for entry in self._entries:
            if entry.importance < min_importance:
                continue
            entry_pad = _emotion_to_pad(entry.emotion)
            if np.allclose(entry_pad, 0):
                continue
            # Cosine sim in PAD space
            dot = float(np.dot(target_pad, entry_pad))
            norm = float(np.linalg.norm(target_pad) * np.linalg.norm(entry_pad))
            sim = dot / max(norm, 1e-8)
            # Weight by vividness
            score = sim * (entry.vividness / 10.0)
            scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "content":    e.content,
                "score":      round(s, 4),
                "emotion":    e.emotion,
                "importance": e.importance,
                "vividness":  round(e.vividness, 4),
                "uid":        e.uid,
            }
            for s, e in scored[:top_k]
        ]

    def find_contradictions(
        self,
        text: str,
        emotion: str = "neutral",
        threshold: float = 0.70,
    ) -> list[dict]:
        """Find stored memories that might contradict the given text.

        Uses a combination of:
        - High semantic similarity (they're about the same topic)
        - Opposite emotional valence (PAD pleasure axis)
        """
        if not hasattr(self, "_vivid_checked"):
            self._detect_vivid_model()
            self._vivid_checked = True

        if self._is_vivid_model:
            q_base = self._encode_query(text, mood="neutral")
        else:
            q_base = np.array(
                _get_model().encode(text, normalize_embeddings=True,
                                    show_progress_bar=False),
                dtype=np.float32,
            )
        q_pad = _emotion_to_pad(emotion)

        results: list[dict] = []
        for entry in self._entries:
            # Semantic similarity on base embedding only
            if self._is_vivid_model:
                e_base = entry.vector
            else:
                e_base = entry.vector[:_BASE_DIM]
            # Cosine similarity (explicit normalisation for VividEmbedder)
            qn = np.linalg.norm(q_base)
            en = np.linalg.norm(e_base)
            cos_sim = float(np.dot(q_base, e_base) / max(qn * en, 1e-8))

            if cos_sim < threshold:
                continue  # not similar enough to be a contradiction

            # Emotional opposition
            e_pad = _emotion_to_pad(entry.emotion)
            valence_diff = abs(float(q_pad[0]) - float(e_pad[0]))

            if valence_diff > 0.6:
                contradiction_score = cos_sim * (0.5 + valence_diff * 0.5)
                results.append({
                    "content":       entry.content,
                    "contradiction": round(contradiction_score, 4),
                    "semantic_sim":  round(cos_sim, 4),
                    "valence_diff":  round(valence_diff, 4),
                    "emotion":       entry.emotion,
                    "uid":           entry.uid,
                })

        results.sort(key=lambda x: x["contradiction"], reverse=True)
        return results

    # ──────────────────────────────────────────────────────────
    # Public API — Index management
    # ──────────────────────────────────────────────────────────

    def remove(self, uid: str) -> bool:
        """Remove a memory by its uid.  Returns True if found."""
        for i, entry in enumerate(self._entries):
            if entry.uid == uid:
                self._entries.pop(i)
                self._uid_set.discard(uid)
                return True
        return False

    def update_importance(self, uid: str, new_importance: int) -> bool:
        """Update a memory's importance and re-embed its meta dims."""
        for entry in self._entries:
            if entry.uid == uid:
                entry.importance = max(1, min(10, new_importance))
                entry.vector = self._build_vector(entry)
                return True
        return False

    def touch(self, uid: str, spacing_bonus: float = 1.8,
              min_spacing_days: float = 0.5,
              diminishing_rate: float = 0.85) -> bool:
        """Simulate a spaced-repetition touch on a memory."""
        for entry in self._entries:
            if entry.uid == uid:
                # Only boost stability if enough time has passed
                if entry.age_days >= min_spacing_days:
                    effective = 1.0 + (spacing_bonus - 1.0) * (
                        diminishing_rate ** max(0, int(entry.stability / 3))
                    )
                    entry.stability = min(entry.stability * effective, 180.0)
                    # Re-embed to update stability dim
                    entry.vector = self._build_vector(entry)
                return True
        return False

    @property
    def size(self) -> int:
        """Number of indexed memories."""
        return len(self._entries)

    def entries(self) -> list[VividEntry]:
        """Return a copy of all entries."""
        return list(self._entries)

    def get(self, uid: str) -> VividEntry | None:
        """Look up a single entry by uid."""
        for entry in self._entries:
            if entry.uid == uid:
                return entry
        return None

    def clear(self):
        """Remove all entries from the index."""
        self._entries.clear()
        self._uid_set.clear()

    # ──────────────────────────────────────────────────────────
    # Public API — Analytics
    # ──────────────────────────────────────────────────────────

    def emotion_clusters(self, n_clusters: int = 5) -> list[dict]:
        """Group memories by emotional similarity (k-means on PAD).

        Returns a list of clusters, each with a centroid emotion label
        and the entries in that cluster.
        """
        if len(self._entries) < n_clusters:
            return [{"label": "all", "entries": [e.to_dict() for e in self._entries]}]

        # Extract PAD vectors
        pads = np.array([_emotion_to_pad(e.emotion) for e in self._entries],
                        dtype=np.float32)

        # Simple k-means (no sklearn dependency)
        centroids = pads[np.random.choice(len(pads), n_clusters, replace=False)]
        for _ in range(20):  # 20 iterations
            dists = np.linalg.norm(pads[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)
            for k in range(n_clusters):
                mask = labels == k
                if mask.any():
                    centroids[k] = pads[mask].mean(axis=0)

        # Build output
        clusters: list[dict] = []
        for k in range(n_clusters):
            mask = labels == k
            c_pad = centroids[k]
            # Find closest named emotion
            best_label = "neutral"
            best_dot = -999.0
            for name, vec in EMOTION_VECTORS.items():
                d = float(np.dot(c_pad, np.array(vec)))
                if d > best_dot:
                    best_dot = d
                    best_label = name
            cluster_entries = [self._entries[i].to_dict()
                               for i in range(len(self._entries)) if labels[i] == k]
            clusters.append({
                "label":    best_label,
                "centroid": c_pad.tolist(),
                "count":    int(mask.sum()),
                "entries":  cluster_entries,
            })

        clusters.sort(key=lambda c: c["count"], reverse=True)
        return clusters

    def vividness_distribution(self) -> dict:
        """Summary statistics of the current index's vividness."""
        if not self._entries:
            return {"count": 0}
        vivs = [e.vividness for e in self._entries]
        return {
            "count":  len(vivs),
            "mean":   round(float(np.mean(vivs)), 4),
            "median": round(float(np.median(vivs)), 4),
            "std":    round(float(np.std(vivs)), 4),
            "min":    round(min(vivs), 4),
            "max":    round(max(vivs), 4),
            "faded":  sum(1 for v in vivs if v < 1.0),
            "vivid":  sum(1 for v in vivs if v >= 5.0),
        }

    # ──────────────────────────────────────────────────────────
    # Persistence  (JSON metadata + binary vectors)
    # ──────────────────────────────────────────────────────────

    def save(self):
        """Persist the index to disk (if persist_dir was set)."""
        if self._persist_dir is None:
            return
        meta_path = self._persist_dir / "vividembed_meta.json"
        vec_path  = self._persist_dir / "vividembed_vectors.bin"

        # Metadata (include vector dim for safe reloading)
        meta = {
            "embed_dim": self._embed_dim,
            "is_vivid_model": self._is_vivid_model,
            "entries": [e.to_dict() for e in self._entries],
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # Vectors (packed float32)
        with open(vec_path, "wb") as f:
            for entry in self._entries:
                f.write(entry.vector.astype(np.float32).tobytes())

    def _load(self):
        """Load index from persist_dir if files exist."""
        if self._persist_dir is None:
            return
        meta_path = self._persist_dir / "vividembed_meta.json"
        vec_path  = self._persist_dir / "vividembed_vectors.bin"

        if not meta_path.exists():
            return

        raw_meta = json.loads(meta_path.read_text(encoding="utf-8"))

        # Support new format (dict with embed_dim) and legacy (bare list)
        if isinstance(raw_meta, dict):
            dim = raw_meta.get("embed_dim", EMBED_DIM)
            self._embed_dim = dim
            self._is_vivid_model = raw_meta.get("is_vivid_model", False)
            entries_list = raw_meta.get("entries", [])
        else:
            dim = EMBED_DIM
            entries_list = raw_meta

        vectors: list[np.ndarray] = []

        if vec_path.exists():
            raw = vec_path.read_bytes()
            stride = dim * 4  # float32 = 4 bytes
            for i in range(len(entries_list)):
                start = i * stride
                end   = start + stride
                if end <= len(raw):
                    vec = np.frombuffer(raw[start:end], dtype=np.float32).copy()
                    vectors.append(vec)
                else:
                    vectors.append(np.zeros(dim, dtype=np.float32))

        for i, d in enumerate(entries_list):
            vec = vectors[i] if i < len(vectors) else None
            entry = VividEntry.from_dict(d, vector=vec)
            if entry.uid not in self._uid_set:
                self._entries.append(entry)
                self._uid_set.add(entry.uid)

    # ──────────────────────────────────────────────────────────
    # Bridge to VividnessMem
    # ──────────────────────────────────────────────────────────

    def index_from_vividnessmem(self, memories: list[dict]) -> int:
        """Bulk-import memories exported from VividnessMem.

        Expects the dicts returned by ``Memory.to_dict()``.
        Returns the number of entries added.
        """
        items = []
        for m in memories:
            items.append({
                "content":    m["content"],
                "emotion":    m.get("emotion", "neutral"),
                "importance": m.get("importance", 5),
                "stability":  m.get("stability", 3.0),
                "source":     m.get("source", "reflection"),
                "entity":     m.get("entity", ""),
                "timestamp":  m.get("timestamp", ""),
            })
        added = self.add_batch(items)
        return len(added)

    # ──────────────────────────────────────────────────────────
    # Internal — vector construction
    # ──────────────────────────────────────────────────────────

    def _build_vector(self, entry: VividEntry) -> np.ndarray:
        """Encode text + emotion + meta into a vector.

        VividEmbedder: 384-d with emotion/importance encoded by the model.
        Vanilla model: 389-d = 384 base + 3 PAD + 2 meta.
        """
        if not hasattr(self, "_vivid_checked"):
            self._detect_vivid_model()
            self._vivid_checked = True

        if self._is_vivid_model:
            return self._encode_memory(
                entry.content,
                emotion=entry.emotion,
                importance=entry.importance,
            )
        base = _get_model().encode(entry.content, normalize_embeddings=True,
                                    show_progress_bar=False)
        return self._assemble_vector(
            base_embedding = np.array(base, dtype=np.float32),
            emotion        = entry.emotion,
            importance     = entry.importance,
            stability      = entry.stability,
        )

    @staticmethod
    def _assemble_vector(
        base_embedding: np.ndarray,
        emotion: str,
        importance: int,
        stability: float,
    ) -> np.ndarray:
        """Concatenate base embedding with PAD + meta dimensions.

        The PAD and meta dims are scaled so they contribute meaningfully
        to cosine similarity without drowning out semantic content.
        """
        pad = _emotion_to_pad(emotion) * _PAD_SCALE
        meta = np.array([
            importance / 10.0,           # normalised to [0, 1]
            min(stability, 180.0) / 180.0,  # normalised to [0, 1]
        ], dtype=np.float32) * _META_SCALE

        full = np.concatenate([base_embedding, pad, meta])
        # L2-normalise the full vector for cosine similarity via dot product
        norm = np.linalg.norm(full)
        if norm > 0:
            full /= norm
        return full

    # ──────────────────────────────────────────────────────────
    # Internal — scoring
    # ──────────────────────────────────────────────────────────

    def _score(
        self,
        query_vec: np.ndarray,
        entry: VividEntry,
        mood_pad: np.ndarray,
    ) -> float:
        """Composite score combining semantic, vividness, mood, recency.

        When using a fine-tuned VividEmbedder, the cosine similarity
        already encodes emotion + mood congruence (the model learned it),
        and the vector magnitude encodes importance.  Post-hoc vividness
        decay and recency are still applied since they depend on wall-
        clock time, which the model can't observe.
        """
        if self._is_vivid_model:
            # ── VividEmbedder scoring ─────────────────────────
            # Cosine similarity encodes semantic + emotion + mood
            q_norm = np.linalg.norm(query_vec)
            m_norm = np.linalg.norm(entry.vector)
            if q_norm > 0 and m_norm > 0:
                cos_sim = float(np.dot(query_vec, entry.vector)
                                / (q_norm * m_norm))
            else:
                cos_sim = 0.0
            cos_sim = max(0.0, cos_sim)

            # Magnitude ratio: how vivid this memory is
            # (normalised so baseline ~1.0)
            viv_signal = m_norm / 5.0  # ~5 is the natural MiniLM norm
            viv_signal = max(0.0, min(2.0, viv_signal))

            # Time-based vividness decay (still needed — model can't
            # observe wall-clock time)
            decay = math.exp(-entry.age_days / max(entry.stability, 0.1))

            # Recency nudge
            recency = math.exp(-entry.age_days * math.log(2) / _RECENCY_HALFLIFE)

            return cos_sim * (0.7 + 0.3 * viv_signal * decay) + 0.1 * recency

        # ── Vanilla model scoring (unchanged) ─────────────────

        # 1. Semantic similarity (cosine, since vectors are L2-normed)
        cos_sim = float(np.dot(query_vec, entry.vector))
        cos_sim = max(0.0, cos_sim)  # floor at 0

        # 2. Vividness weight (normalised to ~[0, 1])
        viv = entry.vividness / 10.0  # max importance=10 → max=1.0
        viv = max(0.0, min(1.0, viv))

        # 3. Mood congruence
        mem_pad = _emotion_to_pad(entry.emotion)
        mood_dot = 0.0
        if not np.allclose(mem_pad, 0) and not np.allclose(mood_pad, 0):
            mood_dot = float(np.dot(mem_pad, mood_pad))
            mood_dot = max(-1.0, min(1.0, mood_dot))
        # Map from [-1, 1] to [0, 1] range for scoring
        mood_score = (1.0 + mood_dot) / 2.0

        # 4. Recency nudge (availability heuristic)
        recency = math.exp(-entry.age_days * math.log(2) / _RECENCY_HALFLIFE)

        # Composite
        score = (
            self._w_sem  * cos_sim
          + self._w_viv  * viv
          + self._w_mood * mood_score
          + self._w_rec  * recency
        )
        return score

    # ──────────────────────────────────────────────────────────
    # repr
    # ──────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        model_tag = "VividEmbedder" if self._is_vivid_model else _MODEL_NAME
        return (
            f"VividEmbed(entries={len(self._entries)}, "
            f"dim={self._embed_dim}, model={model_tag!r})"
        )

    # ──────────────────────────────────────────────────────────
    # Visualisation (lazy-loaded)
    # ──────────────────────────────────────────────────────────

    @property
    def viz(self) -> "VividViz":
        """Access the visualisation toolkit.  ``ve.viz.embedding_map()``"""
        if not hasattr(self, "_viz"):
            self._viz = VividViz(self)
        return self._viz


# ══════════════════════════════════════════════════════════════════
# VividCortex  —  Tier 3: LLM-Powered Intelligence Layer
# ══════════════════════════════════════════════════════════════════
#
# This is what puts VividEmbed ahead of both vanilla RAG and Letta.
#
# Letta (MemGPT) has:
#   - Core memory (always-in-context persona + human blocks)
#   - Archival memory (embedding retrieval)
#   - LLM-in-the-loop deciding when to search
#
# VividCortex has ALL of that, PLUS:
#   - Emotion-aware retrieval (PAD vectors in the embedding space)
#   - Vividness decay (old memories fade like human recall)
#   - Mood-congruent retrieval (sad mood → sad memories surface)
#   - Query decomposition (LLM rewrites vague queries)
#   - Agentic memory ops (self-edit, promote, consolidate, forget)
#   - Contradiction detection (built into the embedding layer)
#   - Memory reflection (periodic self-analysis of stored memories)
#
# Architecture:
#
#  ┌─────────────────────────────────────────────────────┐
#  │  VividCortex                                        │
#  │                                                     │
#  │  ┌───────────────┐   ┌───────────────────────────┐  │
#  │  │  Core Memory   │   │  Working Memory (recent)  │  │
#  │  │  (always in    │   │  Last N turns, rolling    │  │
#  │  │   context)     │   │  window                   │  │
#  │  └───────────────┘   └───────────────────────────┘  │
#  │                                                     │
#  │  ┌───────────────────────────────────────────────┐  │
#  │  │  VividEmbed (archival / long-term memory)      │  │
#  │  │  389-d hybrid vectors + decay + mood scoring   │  │
#  │  └───────────────────────────────────────────────┘  │
#  │                                                     │
#  │  ┌───────────────────────────────────────────────┐  │
#  │  │  LLM Interface (pluggable — llama.cpp/API)     │  │
#  │  │  • Query decomposition                         │  │
#  │  │  • Memory extraction from conversation         │  │
#  │  │  • Self-edit / promote / consolidate / forget   │  │
#  │  │  • Reflection & contradiction resolution        │  │
#  │  └───────────────────────────────────────────────┘  │
#  └─────────────────────────────────────────────────────┘
#

import json as _json
import re as _re
from typing import Any, Callable

# Type alias for LLM callable
# Signature: llm_fn(messages: list[dict]) -> str
LLMCallable = Callable[[list[dict[str, str]]], str]


class CoreMemory:
    """Always-in-context memory blocks (like Letta's persona + human blocks).

    Core memory is NEVER searched — it's injected directly into every
    LLM prompt.  It's the AI's persistent identity and key facts about
    the user.

    Blocks
    ------
    persona : str
        Who the AI is — personality, mannerisms, backstory.
    user : str
        Key facts about the user — name, preferences, relationships.
    system : str
        Operational rules — what the AI should/shouldn't do.
    scratch : str
        Working scratchpad — the AI can write temp notes here.
    """

    _MAX_BLOCK_TOKENS = 2000  # soft limit per block (~8000 chars)

    def __init__(
        self,
        persona: str = "",
        user: str = "",
        system: str = "",
        scratch: str = "",
    ):
        self.persona = persona
        self.user = user
        self.system = system
        self.scratch = scratch

    def render(self) -> str:
        """Render core memory as a string for injection into prompts."""
        parts = []
        if self.persona:
            parts.append(f"<core_memory type=\"persona\">\n{self.persona}\n</core_memory>")
        if self.user:
            parts.append(f"<core_memory type=\"user\">\n{self.user}\n</core_memory>")
        if self.system:
            parts.append(f"<core_memory type=\"system\">\n{self.system}\n</core_memory>")
        if self.scratch:
            parts.append(f"<core_memory type=\"scratch\">\n{self.scratch}\n</core_memory>")
        return "\n".join(parts)

    def update_block(self, block: str, content: str):
        """Replace an entire block."""
        if block not in ("persona", "user", "system", "scratch"):
            return
        setattr(self, block, content[:self._MAX_BLOCK_TOKENS * 4])

    def append_to_block(self, block: str, text: str):
        """Append text to a block."""
        if block not in ("persona", "user", "system", "scratch"):
            return
        current = getattr(self, block)
        setattr(self, block, (current + "\n" + text).strip()[:self._MAX_BLOCK_TOKENS * 4])

    def to_dict(self) -> dict:
        return {
            "persona": self.persona,
            "user": self.user,
            "system": self.system,
            "scratch": self.scratch,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CoreMemory":
        return cls(**{k: d.get(k, "") for k in ("persona", "user", "system", "scratch")})


class VividCortex:
    """Tier 3 intelligence layer — LLM-powered memory management.

    Wraps VividEmbed with an LLM that can:
    - Decompose vague queries into precise retrievals
    - Extract memories from conversation automatically
    - Edit, promote, consolidate, or forget memories
    - Maintain always-in-context core memory
    - Detect and resolve contradictions
    - Reflect on memory patterns periodically

    Parameters
    ----------
    embed : VividEmbed
        The embedding index to manage.
    llm : callable
        ``llm(messages: list[dict]) -> str``
        Any function that takes OpenAI-style messages and returns text.
        Works with llama.cpp's ``create_chat_completion``, OpenAI API,
        or any wrapper.
    core : CoreMemory, optional
        Pre-populated core memory. Created empty if not provided.
    working_memory_turns : int
        How many recent conversation turns to keep in working memory.
    auto_extract : bool
        If True, automatically extract memories from every conversation
        turn (default True).
    """

    # ── System prompts for the LLM ──────────────────────────

    _DECOMPOSE_PROMPT = """You are a memory query optimizer. Given a vague or conversational query, decompose it into 1-3 precise search queries that would find the right memories in an embedding database.

Rules:
- Each query should target a different aspect of what the user might want
- Keep queries short and semantically rich (5-15 words)
- If the original query is already precise, return just that one
- Output ONLY a JSON array of strings, nothing else

Examples:
Input: "that thing with Sarah"
Output: ["conversations with Sarah", "events involving Sarah", "Sarah conflict or argument"]

Input: "why am I feeling down"
Output: ["recent negative experiences", "sources of stress or sadness", "unresolved problems or worries"]

Input: "what's Jordan's favourite food"
Output: ["Jordan food preferences"]"""

    _EXTRACT_PROMPT = """You are a memory extraction system. From the conversation below, extract discrete facts and experiences worth remembering long-term.

For each memory, output a JSON array of objects with these fields:
- "content": the fact or experience (1-2 sentences, third-person or factual)
- "emotion": the dominant emotion (one word: happy, sad, angry, anxious, proud, etc.)
- "importance": 1-10 (10 = life-changing, 1 = trivial)
- "entity": person/thing this is about (empty string if general)
- "source": "conversation" or "reflection"

Rules:
- Only extract genuinely new information, not pleasantries
- Consolidate related facts into single memories
- Skip greetings, acknowledgments, filler
- If nothing worth remembering, return []
- Output ONLY the JSON array, nothing else"""

    _EDIT_PROMPT = """You are a memory management system. You have access to the following stored memories and a new piece of information. Decide what memory operations to perform.

Available operations:
- UPDATE: modify an existing memory's content (e.g., correct outdated info)
- PROMOTE: increase a memory's importance (it's more significant than stored)
- DEMOTE: decrease a memory's importance (it's less significant than thought)
- FORGET: mark a memory for removal (it's wrong, outdated, or irrelevant)
- CONSOLIDATE: merge two memories into one (they overlap)
- NONE: no changes needed

Output a JSON array of operations:
[{"op": "UPDATE", "uid": "...", "new_content": "..."}, ...]
[{"op": "PROMOTE", "uid": "...", "new_importance": 8}, ...]
[{"op": "FORGET", "uid": "..."}, ...]
[{"op": "CONSOLIDATE", "uid_keep": "...", "uid_remove": "...", "merged_content": "..."}, ...]

If no operations needed, return []
Output ONLY the JSON array."""

    _REFLECT_PROMPT = """You are a memory reflection system. Review the following memories and surface any insights, patterns, contradictions, or things that need attention.

Output a JSON object with:
- "insights": array of string observations about patterns
- "contradictions": array of {"memory_a": "uid", "memory_b": "uid", "description": "..."}
- "promotions": array of {"uid": "...", "reason": "..."} for under-valued memories
- "demotions": array of {"uid": "...", "reason": "..."} for over-valued memories

Output ONLY the JSON object."""

    def __init__(
        self,
        embed: VividEmbed,
        llm: LLMCallable,
        core: CoreMemory | None = None,
        working_memory_turns: int = 20,
        auto_extract: bool = True,
    ):
        self.embed = embed
        self._llm = llm
        self.core = core or CoreMemory()
        self._working_memory: list[dict[str, str]] = []  # recent turns
        self._max_turns = working_memory_turns
        self._auto_extract = auto_extract
        self._turn_count = 0

    # ──────────────────────────────────────────────────────────
    # Public API — Query (the smart retrieval pipeline)
    # ──────────────────────────────────────────────────────────

    def query(
        self,
        text: str,
        mood: str = "neutral",
        top_k: int = 5,
        decompose: bool = True,
        **kwargs,
    ) -> list[dict]:
        """Smart query with optional LLM decomposition.

        If ``decompose=True``, uses the LLM to break vague queries into
        precise sub-queries, runs each through VividEmbed, and merges
        results by max score (de-duplicated by uid).

        Parameters
        ----------
        text : str
            The user's query (can be vague/conversational).
        mood : str
            Current mood for mood-congruent retrieval.
        top_k : int
            Max results to return (after merge).
        decompose : bool
            Whether to use LLM query decomposition.
        **kwargs
            Forwarded to ``VividEmbed.query()`` (entity_filter, etc.)

        Returns
        -------
        list[dict]
            Merged, de-duplicated results ranked by score.
        """
        if not decompose:
            return self.embed.query(text, top_k=top_k, mood=mood, **kwargs)

        # Decompose into sub-queries
        sub_queries = self._decompose_query(text)

        # Run each sub-query and merge
        seen: dict[str, dict] = {}  # uid → best result dict
        for sq in sub_queries:
            results = self.embed.query(sq, top_k=top_k, mood=mood, **kwargs)
            for r in results:
                uid = r["uid"]
                if uid not in seen or r["score"] > seen[uid]["score"]:
                    seen[uid] = r

        # Sort merged results by score
        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
        return merged[:top_k]

    # ──────────────────────────────────────────────────────────
    # Public API — Conversation processing
    # ──────────────────────────────────────────────────────────

    def process_turn(
        self,
        role: str,
        content: str,
        mood: str = "neutral",
    ) -> list[VividEntry]:
        """Process a conversation turn — add to working memory and
        optionally extract long-term memories.

        Parameters
        ----------
        role : str
            "user" or "assistant"
        content : str
            The message text.
        mood : str
            Current emotional state (for mood-tagged memory extraction).

        Returns
        -------
        list[VividEntry]
            Any new memories extracted and stored.
        """
        # Add to working memory
        self._working_memory.append({"role": role, "content": content})
        if len(self._working_memory) > self._max_turns * 2:
            self._working_memory = self._working_memory[-self._max_turns * 2:]

        self._turn_count += 1
        extracted: list[VividEntry] = []

        if self._auto_extract and role == "user":
            extracted = self._extract_memories(mood=mood)

        return extracted

    def get_context_block(self, mood: str = "neutral") -> str:
        """Build the full context block for an LLM prompt.

        Returns a string containing:
        1. Core memory (always present)
        2. Relevant retrieved memories (based on recent conversation)
        3. Working memory summary
        """
        parts = []

        # Core memory
        core_text = self.core.render()
        if core_text:
            parts.append(core_text)

        # Retrieve relevant memories based on last user message
        last_user = ""
        for turn in reversed(self._working_memory):
            if turn["role"] == "user":
                last_user = turn["content"]
                break

        if last_user and self.embed.size > 0:
            results = self.query(last_user, mood=mood, top_k=5, decompose=False)
            if results:
                mem_lines = []
                for r in results:
                    age = r.get("age_days", 0)
                    if age < 1:
                        age_str = "today"
                    elif age < 7:
                        age_str = f"{age:.0f}d ago"
                    elif age < 30:
                        age_str = f"{age/7:.0f}w ago"
                    else:
                        age_str = f"{age/30:.0f}mo ago"
                    mem_lines.append(
                        f"- [{r['emotion']}] ({age_str}, imp={r['importance']}) "
                        f"{r['content']}"
                    )
                parts.append(
                    "<retrieved_memories>\n"
                    + "\n".join(mem_lines)
                    + "\n</retrieved_memories>"
                )

        return "\n\n".join(parts)

    # ──────────────────────────────────────────────────────────
    # Public API — Agentic memory operations
    # ──────────────────────────────────────────────────────────

    def edit_memories(self, new_info: str, mood: str = "neutral") -> list[dict]:
        """Use the LLM to decide how to update existing memories based
        on new information.

        Returns a list of operations performed.
        """
        # Find potentially relevant memories
        results = self.query(new_info, mood=mood, top_k=10, decompose=True)
        if not results:
            return []

        mem_context = "\n".join(
            f"[uid={r['uid']}] [{r['emotion']}] (imp={r['importance']}) {r['content']}"
            for r in results
        )

        messages = [
            {"role": "system", "content": self._EDIT_PROMPT},
            {"role": "user", "content": (
                f"Stored memories:\n{mem_context}\n\n"
                f"New information:\n{new_info}"
            )},
        ]

        raw = self._llm(messages)
        ops = self._parse_json_array(raw)
        executed = []

        for op in ops:
            op_type = op.get("op", "").upper()
            try:
                if op_type == "UPDATE":
                    entry = self.embed.get(op["uid"])
                    if entry:
                        self.embed.remove(op["uid"])
                        self.embed.add(
                            op["new_content"],
                            emotion=entry.emotion,
                            importance=entry.importance,
                            stability=entry.stability,
                            source=entry.source,
                            entity=entry.entity,
                        )
                        executed.append(op)

                elif op_type == "PROMOTE":
                    self.embed.update_importance(op["uid"], op.get("new_importance", 8))
                    executed.append(op)

                elif op_type == "DEMOTE":
                    self.embed.update_importance(op["uid"], op.get("new_importance", 3))
                    executed.append(op)

                elif op_type == "FORGET":
                    self.embed.remove(op["uid"])
                    executed.append(op)

                elif op_type == "CONSOLIDATE":
                    keep = self.embed.get(op["uid_keep"])
                    if keep:
                        self.embed.remove(op.get("uid_remove", ""))
                        self.embed.remove(op["uid_keep"])
                        self.embed.add(
                            op["merged_content"],
                            emotion=keep.emotion,
                            importance=max(keep.importance, 7),
                            stability=keep.stability,
                            source=keep.source,
                            entity=keep.entity,
                        )
                        executed.append(op)
            except (KeyError, TypeError):
                continue  # malformed op — skip

        return executed

    def reflect(self, sample_size: int = 30) -> dict:
        """Trigger a reflection cycle — the LLM reviews a sample of
        memories and surfaces patterns, contradictions, and re-evaluations.

        Returns the reflection result dict.
        """
        entries = self.embed.entries()
        if not entries:
            return {"insights": [], "contradictions": [], "promotions": [], "demotions": []}

        # Sample: mix of recent, important, and random
        recent = sorted(entries, key=lambda e: e.age_days)[:10]
        important = sorted(entries, key=lambda e: e.importance, reverse=True)[:10]
        remaining = [e for e in entries if e not in recent and e not in important]
        import random
        random_sample = random.sample(remaining, min(10, len(remaining)))
        sample = list({e.uid: e for e in recent + important + random_sample}.values())[:sample_size]

        mem_lines = "\n".join(
            f"[uid={e.uid}] [{e.emotion}] (imp={e.importance}, age={e.age_days:.0f}d, "
            f"viv={e.vividness:.2f}) {e.content}"
            for e in sample
        )

        messages = [
            {"role": "system", "content": self._REFLECT_PROMPT},
            {"role": "user", "content": f"Memories to review:\n{mem_lines}"},
        ]

        raw = self._llm(messages)
        result = self._parse_json_object(raw)

        # Apply any promotions/demotions
        for p in result.get("promotions", []):
            uid = p.get("uid", "")
            entry = self.embed.get(uid)
            if entry:
                self.embed.update_importance(uid, min(entry.importance + 2, 10))

        for d in result.get("demotions", []):
            uid = d.get("uid", "")
            entry = self.embed.get(uid)
            if entry:
                self.embed.update_importance(uid, max(entry.importance - 2, 1))

        return result

    def update_core(self, block: str, content: str):
        """Update a core memory block directly."""
        self.core.update_block(block, content)

    def append_core(self, block: str, text: str):
        """Append to a core memory block."""
        self.core.append_to_block(block, text)

    # ──────────────────────────────────────────────────────────
    # Public API — Convenience
    # ──────────────────────────────────────────────────────────

    @property
    def size(self) -> int:
        """Number of entries in the archival memory (VividEmbed)."""
        return self.embed.size

    @property
    def working_memory(self) -> list[dict[str, str]]:
        """Recent conversation turns."""
        return list(self._working_memory)

    def save(self):
        """Save both the embedding index and core memory."""
        self.embed.save()
        if self.embed._persist_dir is not None:
            core_path = self.embed._persist_dir / "core_memory.json"
            with open(core_path, "w", encoding="utf-8") as f:
                _json.dump(self.core.to_dict(), f, indent=2)

    def load_core(self):
        """Load core memory from the persist directory."""
        if self.embed._persist_dir is not None:
            core_path = self.embed._persist_dir / "core_memory.json"
            if core_path.exists():
                with open(core_path, "r", encoding="utf-8") as f:
                    self.core = CoreMemory.from_dict(_json.load(f))

    # ──────────────────────────────────────────────────────────
    # Internal — LLM interactions
    # ──────────────────────────────────────────────────────────

    def _decompose_query(self, text: str) -> list[str]:
        """Use LLM to decompose a vague query into precise sub-queries."""
        messages = [
            {"role": "system", "content": self._DECOMPOSE_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            raw = self._llm(messages)
            queries = self._parse_json_array(raw)
            # Validate: must be list of strings
            if queries and all(isinstance(q, str) for q in queries):
                return queries[:3]
        except Exception:
            pass
        # Fallback: use original query as-is
        return [text]

    def _extract_memories(self, mood: str = "neutral") -> list[VividEntry]:
        """Extract memories from recent conversation turns."""
        # Use last 4 turns for context
        recent = self._working_memory[-4:]
        if not recent:
            return []

        conv_text = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in recent
        )

        messages = [
            {"role": "system", "content": self._EXTRACT_PROMPT},
            {"role": "user", "content": conv_text},
        ]

        try:
            raw = self._llm(messages)
            items = self._parse_json_array(raw)
        except Exception:
            return []

        extracted: list[VividEntry] = []
        for item in items:
            if not isinstance(item, dict) or "content" not in item:
                continue
            # Check for near-duplicates before storing
            existing = self.embed.query(
                item["content"], top_k=1, mood=mood,
            ) if hasattr(self.embed, "query") else []
            if existing and existing[0]["score"] > 0.85:
                continue  # too similar to existing memory

            entry = self.embed.add(
                content=item["content"],
                emotion=item.get("emotion", "neutral"),
                importance=item.get("importance", 5),
                source=item.get("source", "conversation"),
                entity=item.get("entity", ""),
            )
            extracted.append(entry)

        return extracted

    # ──────────────────────────────────────────────────────────
    # Internal — JSON parsing helpers
    # ──────────────────────────────────────────────────────────

    @staticmethod
    def _parse_json_array(text: str) -> list:
        """Extract a JSON array from LLM output (handles markdown fences)."""
        text = text.strip()
        # Remove markdown code fences
        text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        try:
            result = _json.loads(text)
            if isinstance(result, list):
                return result
        except _json.JSONDecodeError:
            pass
        # Try to find array in the text
        match = _re.search(r"\[.*\]", text, _re.DOTALL)
        if match:
            try:
                return _json.loads(match.group())
            except _json.JSONDecodeError:
                pass
        return []

    @staticmethod
    def _parse_json_object(text: str) -> dict:
        """Extract a JSON object from LLM output."""
        text = text.strip()
        text = _re.sub(r"^```(?:json)?\s*\n?", "", text)
        text = _re.sub(r"\n?```\s*$", "", text)
        text = text.strip()
        try:
            result = _json.loads(text)
            if isinstance(result, dict):
                return result
        except _json.JSONDecodeError:
            pass
        match = _re.search(r"\{.*\}", text, _re.DOTALL)
        if match:
            try:
                return _json.loads(match.group())
            except _json.JSONDecodeError:
                pass
        return {}

    def __repr__(self) -> str:
        return (
            f"VividCortex(archival={self.embed.size}, "
            f"working={len(self._working_memory)} turns, "
            f"core_blocks={sum(1 for b in ('persona','user','system','scratch') if getattr(self.core, b))})"
        )


# ──────────────────────────────────────────────────────────────────
# VividViz  —  Visualisation Toolkit
# ──────────────────────────────────────────────────────────────────

class VividViz:
    """Visualisation helpers attached to a VividEmbed instance.

    All methods pop up a matplotlib window (or return the figure if
    ``show=False``).  Requires: ``matplotlib``, ``scikit-learn``.
    """

    # Colour palette: emotion-valence → colour
    _VALENCE_CMAP = {
        "pos_calm":   "#4CAF50",   # green
        "pos_active": "#FF9800",   # orange
        "neg_low":    "#2196F3",   # blue
        "neg_high":   "#F44336",   # red
        "neutral":    "#9E9E9E",   # grey
    }

    def __init__(self, embed: VividEmbed):
        self._embed = embed

    @staticmethod
    def _lazy_imports():
        import matplotlib
        matplotlib.use("TkAgg")    # ensure interactive backend on Windows
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
        return plt

    def _emotion_colour(self, emotion: str) -> str:
        """Map an emotion to a colour based on its PAD valence/arousal."""
        pad = _emotion_to_pad(emotion)
        p, a = float(pad[0]), float(pad[1])
        if abs(p) < 0.15 and abs(a) < 0.15:
            return self._VALENCE_CMAP["neutral"]
        if p >= 0 and a < 0.3:
            return self._VALENCE_CMAP["pos_calm"]
        if p >= 0:
            return self._VALENCE_CMAP["pos_active"]
        if a < 0.3:
            return self._VALENCE_CMAP["neg_low"]
        return self._VALENCE_CMAP["neg_high"]

    # ── 1. Embedding Map (t-SNE / PCA) ──────────────────────

    def embedding_map(
        self,
        method: str = "tsne",
        colour_by: str = "emotion",
        label: bool = True,
        max_label_len: int = 40,
        figsize: tuple[int, int] = (14, 10),
        show: bool = True,
    ):
        """2-D scatter plot of all indexed memories.

        Parameters
        ----------
        method : "tsne" or "pca"
        colour_by : "emotion", "importance", "source", or "age"
        label : bool — annotate each dot with a text snippet
        """
        plt = self._lazy_imports()
        entries = self._embed.entries()
        if len(entries) < 2:
            print("Need at least 2 entries to plot.")
            return

        vecs = np.array([e.vector for e in entries], dtype=np.float32)

        if method == "tsne":
            from sklearn.manifold import TSNE
            perp = min(30, max(2, len(entries) - 1))
            coords = TSNE(n_components=2, perplexity=perp,
                          random_state=42, init="pca").fit_transform(vecs)
        else:
            from sklearn.decomposition import PCA
            coords = PCA(n_components=2, random_state=42).fit_transform(vecs)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        # Colour mapping
        if colour_by == "importance":
            cvals = [e.importance for e in entries]
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=cvals, cmap="YlOrRd", s=120, edgecolors="white",
                            linewidths=0.5, alpha=0.9, vmin=1, vmax=10)
            plt.colorbar(sc, ax=ax, label="Importance", shrink=0.7)
        elif colour_by == "age":
            cvals = [e.age_days for e in entries]
            sc = ax.scatter(coords[:, 0], coords[:, 1],
                            c=cvals, cmap="cool", s=120, edgecolors="white",
                            linewidths=0.5, alpha=0.9)
            plt.colorbar(sc, ax=ax, label="Age (days)", shrink=0.7)
        elif colour_by == "source":
            sources = list({e.source for e in entries})
            src_colours = plt.cm.Set2(np.linspace(0, 1, max(len(sources), 1)))
            src_map = {s: src_colours[i] for i, s in enumerate(sources)}
            colours = [src_map[e.source] for e in entries]
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=colours, s=120, edgecolors="white",
                       linewidths=0.5, alpha=0.9)
            for s, c in src_map.items():
                ax.scatter([], [], c=[c], label=s, s=80)
            ax.legend(loc="upper right", facecolor="#1a1a2e",
                      edgecolor="#444", labelcolor="white")
        else:  # emotion (default)
            colours = [self._emotion_colour(e.emotion) for e in entries]
            ax.scatter(coords[:, 0], coords[:, 1],
                       c=colours, s=120, edgecolors="white",
                       linewidths=0.5, alpha=0.9)
            # Legend
            for cat, col in self._VALENCE_CMAP.items():
                ax.scatter([], [], c=col, label=cat.replace("_", " ").title(), s=80)
            ax.legend(loc="upper right", facecolor="#1a1a2e",
                      edgecolor="#444", labelcolor="white")

        # Labels
        if label:
            for i, entry in enumerate(entries):
                snippet = entry.content[:max_label_len]
                if len(entry.content) > max_label_len:
                    snippet += "..."
                ax.annotate(
                    snippet,
                    (coords[i, 0], coords[i, 1]),
                    fontsize=7, color="white", alpha=0.85,
                    textcoords="offset points", xytext=(6, 4),
                )

        ax.set_title(f"VividEmbed — Memory Map ({method.upper()}, colour={colour_by})",
                      color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#666")
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 2. PAD Emotion Space ────────────────────────────────

    def emotion_space(
        self,
        figsize: tuple[int, int] = (12, 9),
        show: bool = True,
    ):
        """3-D scatter of memories in Pleasure-Arousal-Dominance space.

        Point size = importance, colour = vividness.
        """
        plt = self._lazy_imports()
        entries = self._embed.entries()
        if not entries:
            print("No entries to plot.")
            return

        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("#16213e")

        pads = np.array([_emotion_to_pad(e.emotion) for e in entries])
        sizes = np.array([e.importance for e in entries]) * 25
        vivs = np.array([min(e.vividness, 10) for e in entries])

        sc = ax.scatter(pads[:, 0], pads[:, 1], pads[:, 2],
                        c=vivs, cmap="plasma", s=sizes,
                        edgecolors="white", linewidths=0.3, alpha=0.85)
        plt.colorbar(sc, ax=ax, label="Vividness", shrink=0.6, pad=0.1)

        # Label each point with emotion
        for i, e in enumerate(entries):
            ax.text(pads[i, 0], pads[i, 1], pads[i, 2],
                    f"  {e.emotion}", fontsize=7, color="white", alpha=0.7)

        ax.set_xlabel("Pleasure", color="white", fontsize=10)
        ax.set_ylabel("Arousal",  color="white", fontsize=10)
        ax.set_zlabel("Dominance", color="white", fontsize=10)
        ax.set_title("VividEmbed — PAD Emotion Space",
                      color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#666")
        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 3. Vividness Decay Curves ───────────────────────────

    def decay_curves(
        self,
        max_days: float = 30.0,
        figsize: tuple[int, int] = (12, 7),
        show: bool = True,
    ):
        """Plot how each memory's vividness fades over time.

        Shows actual current position on each curve.
        """
        plt = self._lazy_imports()
        entries = self._embed.entries()
        if not entries:
            print("No entries to plot.")
            return

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        days = np.linspace(0, max_days, 300)

        for entry in entries:
            curve = entry.importance * np.exp(-days / max(entry.stability, 0.1))
            colour = self._emotion_colour(entry.emotion)
            snippet = entry.content[:35] + ("..." if len(entry.content) > 35 else "")
            ax.plot(days, curve, color=colour, alpha=0.7, linewidth=1.5,
                    label=f"[{entry.emotion}] {snippet}")
            # Mark current position
            ax.scatter([entry.age_days], [entry.vividness],
                       color=colour, s=80, zorder=5, edgecolors="white")

        ax.set_xlabel("Age (days)", color="white", fontsize=11)
        ax.set_ylabel("Vividness", color="white", fontsize=11)
        ax.set_title("VividEmbed — Vividness Decay Curves",
                      color="white", fontsize=14, fontweight="bold")
        ax.tick_params(colors="#aaa")
        ax.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
                  edgecolor="#444", labelcolor="white")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.axhline(y=1.0, color="#F44336", linestyle="--", alpha=0.4,
                   label="Fade threshold")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 4. Query Radar ──────────────────────────────────────

    def query_radar(
        self,
        query_text: str,
        mood: str = "neutral",
        top_k: int = 5,
        figsize: tuple[int, int] = (10, 10),
        show: bool = True,
    ):
        """Radar chart showing how top results score across dimensions.

        Axes: Semantic, Vividness, Mood Match, Recency.
        """
        plt = self._lazy_imports()
        results = self._embed.query(query_text, top_k=top_k, mood=mood)
        if not results:
            print("No results to plot.")
            return

        categories = ["Semantic", "Vividness", "Mood", "Recency"]
        n = len(categories)
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        mood_pad = _emotion_to_pad(mood) if isinstance(mood, str) else np.array(mood)

        colours = plt.cm.viridis(np.linspace(0.2, 0.9, len(results)))

        for idx, (r, colour) in enumerate(zip(results, colours)):
            entry = self._embed.get(r["uid"])
            if not entry:
                continue

            # Breakdown: re-compute individual components
            if self._embed._is_vivid_model:
                q_vec = self._embed._encode_query(query_text, mood="neutral")
                qn = np.linalg.norm(q_vec)
                en = np.linalg.norm(entry.vector)
                cos_sim = max(0.0, float(
                    np.dot(q_vec, entry.vector) / max(qn * en, 1e-8)))
                viv_signal = en / 5.0
                viv = min(1.0, max(0.0, viv_signal))
            else:
                cos_sim = max(0.0, float(np.dot(
                    self._embed._assemble_vector(
                        np.array(_get_model().encode(query_text, normalize_embeddings=True,
                                                      show_progress_bar=False), dtype=np.float32),
                        "neutral", 5, 3.0),
                    entry.vector)))
                viv = min(1.0, max(0.0, entry.vividness / 10.0))
            mem_pad = _emotion_to_pad(entry.emotion)
            mood_dot = 0.0
            if not np.allclose(mem_pad, 0) and not np.allclose(mood_pad, 0):
                mood_dot = float(np.dot(mem_pad, mood_pad))
            mood_score = (1.0 + max(-1.0, min(1.0, mood_dot))) / 2.0
            recency = math.exp(-entry.age_days * math.log(2) / _RECENCY_HALFLIFE)

            values = [cos_sim, viv, mood_score, recency]
            values += values[:1]

            snippet = r["content"][:30] + "..."
            ax.plot(angles, values, 'o-', color=colour, linewidth=2,
                    label=f"#{idx+1} {snippet}", alpha=0.8)
            ax.fill(angles, values, color=colour, alpha=0.15)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, color="white", fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title(f'Query: "{query_text[:40]}"  (mood: {mood})',
                      color="white", fontsize=13, fontweight="bold", pad=20)
        ax.tick_params(colors="#666")
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
                  fontsize=8, facecolor="#1a1a2e",
                  edgecolor="#444", labelcolor="white")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 5. Similarity Heatmap ───────────────────────────────

    def similarity_heatmap(
        self,
        figsize: tuple[int, int] = (14, 12),
        show: bool = True,
    ):
        """Pairwise cosine-similarity heatmap of all memories.

        Helps spot clusters, duplicates, and potential contradictions.
        """
        plt = self._lazy_imports()
        entries = self._embed.entries()
        n = len(entries)
        if n < 2:
            print("Need at least 2 entries.")
            return

        vecs = np.array([e.vector for e in entries], dtype=np.float32)
        sim_matrix = vecs @ vecs.T  # cosine sim (already L2-normed)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        im = ax.imshow(sim_matrix, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

        labels = [f"[{e.emotion}] {e.content[:25]}..." for e in entries]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right",
                           fontsize=7, color="white")
        ax.set_yticklabels(labels, fontsize=7, color="white")
        ax.set_title("VividEmbed — Pairwise Similarity",
                      color="white", fontsize=14, fontweight="bold")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 6. Mood Congruence Comparison ───────────────────────

    def mood_comparison(
        self,
        query_text: str,
        moods: list[str] | None = None,
        top_k: int = 5,
        figsize: tuple[int, int] = (14, 8),
        show: bool = True,
    ):
        """Side-by-side bar chart: same query under different moods.

        Shows how mood shifts retrieval rankings.
        """
        plt = self._lazy_imports()
        if moods is None:
            moods = ["happy", "sad", "anxious", "calm", "angry"]

        all_results: dict[str, list[dict]] = {}
        all_contents: set[str] = set()
        for mood in moods:
            res = self._embed.query(query_text, top_k=top_k, mood=mood)
            all_results[mood] = res
            for r in res:
                all_contents.add(r["content"][:35])

        content_list = sorted(all_contents)
        x = np.arange(len(content_list))
        width = 0.8 / len(moods)

        fig, ax = plt.subplots(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        ax.set_facecolor("#16213e")

        colours = plt.cm.Set1(np.linspace(0, 1, len(moods)))

        for i, mood in enumerate(moods):
            scores = []
            for c in content_list:
                found = 0.0
                for r in all_results[mood]:
                    if r["content"][:35] == c:
                        found = r["score"]
                        break
                scores.append(found)
            ax.bar(x + i * width, scores, width, label=mood,
                   color=colours[i], alpha=0.85, edgecolor="white",
                   linewidth=0.3)

        ax.set_xticks(x + width * len(moods) / 2)
        ax.set_xticklabels([c[:30] + "..." for c in content_list],
                           rotation=35, ha="right", fontsize=7, color="white")
        ax.set_ylabel("Score", color="white", fontsize=11)
        ax.set_title(f'Mood Comparison: "{query_text[:40]}"',
                      color="white", fontsize=14, fontweight="bold")
        ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white")
        ax.tick_params(colors="#aaa")
        for spine in ax.spines.values():
            spine.set_color("#333")
        plt.tight_layout()
        if show:
            plt.show()
        return fig

    # ── 7. Dashboard (all-in-one) ───────────────────────────

    def dashboard(
        self,
        query_text: str = "tell me something interesting",
        mood: str = "curious",
        figsize: tuple[int, int] = (20, 16),
        show: bool = True,
    ):
        """Full 4-panel dashboard: map + decay + heatmap + radar."""
        plt = self._lazy_imports()
        entries = self._embed.entries()
        if len(entries) < 3:
            print("Need at least 3 entries for the dashboard.")
            return

        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("#1a1a2e")
        fig.suptitle("VividEmbed Dashboard",
                      color="white", fontsize=18, fontweight="bold", y=0.98)

        # Panel 1: Embedding map (PCA for speed)
        ax1 = fig.add_subplot(2, 2, 1)
        ax1.set_facecolor("#16213e")
        vecs = np.array([e.vector for e in entries], dtype=np.float32)
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2, random_state=42).fit_transform(vecs)
        colours = [self._emotion_colour(e.emotion) for e in entries]
        ax1.scatter(coords[:, 0], coords[:, 1], c=colours, s=100,
                    edgecolors="white", linewidths=0.5, alpha=0.9)
        for i, e in enumerate(entries):
            ax1.annotate(e.content[:25] + "...", (coords[i, 0], coords[i, 1]),
                         fontsize=6, color="white", alpha=0.7,
                         textcoords="offset points", xytext=(5, 3))
        ax1.set_title("Memory Map (PCA)", color="white", fontsize=12)
        ax1.tick_params(colors="#666")
        for s in ax1.spines.values(): s.set_color("#333")

        # Panel 2: Decay curves
        ax2 = fig.add_subplot(2, 2, 2)
        ax2.set_facecolor("#16213e")
        days = np.linspace(0, 30, 200)
        for entry in entries:
            curve = entry.importance * np.exp(-days / max(entry.stability, 0.1))
            ax2.plot(days, curve, color=self._emotion_colour(entry.emotion),
                     alpha=0.6, linewidth=1.2)
            ax2.scatter([entry.age_days], [entry.vividness],
                        color=self._emotion_colour(entry.emotion),
                        s=50, zorder=5, edgecolors="white")
        ax2.set_xlabel("Days", color="white", fontsize=9)
        ax2.set_ylabel("Vividness", color="white", fontsize=9)
        ax2.set_title("Decay Curves", color="white", fontsize=12)
        ax2.axhline(y=1.0, color="#F44336", linestyle="--", alpha=0.3)
        ax2.tick_params(colors="#666")
        for s in ax2.spines.values(): s.set_color("#333")

        # Panel 3: Similarity heatmap
        ax3 = fig.add_subplot(2, 2, 3)
        ax3.set_facecolor("#16213e")
        sim = vecs @ vecs.T
        im = ax3.imshow(sim, cmap="RdYlGn", vmin=-0.2, vmax=1.0)
        plt.colorbar(im, ax=ax3, shrink=0.7)
        labels = [e.emotion for e in entries]
        ax3.set_xticks(range(len(entries)))
        ax3.set_yticks(range(len(entries)))
        ax3.set_xticklabels(labels, rotation=45, ha="right",
                            fontsize=7, color="white")
        ax3.set_yticklabels(labels, fontsize=7, color="white")
        ax3.set_title("Similarity Matrix", color="white", fontsize=12)

        # Panel 4: Query results breakdown
        ax4 = fig.add_subplot(2, 2, 4)
        ax4.set_facecolor("#16213e")
        results = self._embed.query(query_text, top_k=min(6, len(entries)),
                                     mood=mood)
        if results:
            names = [r["content"][:28] + "..." for r in results]
            scores = [r["score"] for r in results]
            bar_colours = [self._emotion_colour(r["emotion"]) for r in results]
            y_pos = np.arange(len(names))
            ax4.barh(y_pos, scores, color=bar_colours, edgecolor="white",
                     linewidth=0.3, alpha=0.85)
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(names, fontsize=7, color="white")
            ax4.invert_yaxis()
        ax4.set_xlabel("Score", color="white", fontsize=9)
        ax4.set_title(f'Query: "{query_text[:30]}" (mood: {mood})',
                       color="white", fontsize=12)
        ax4.tick_params(colors="#666")
        for s in ax4.spines.values(): s.set_color("#333")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        if show:
            plt.show()
        return fig


# ──────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("VividEmbed — quick demo")
    print("=" * 50)

    ve = VividEmbed()

    # Add some memories
    ve.add("I love my morning runs through the park", emotion="happy", importance=8)
    ve.add("The argument with my boss left me drained", emotion="frustrated", importance=6)
    ve.add("Grandma's apple pie recipe reminds me of childhood", emotion="nostalgic", importance=9)
    ve.add("I need to fix the leaking pipe in the kitchen", emotion="stressed", importance=4)
    ve.add("Our anniversary dinner was magical", emotion="loving", importance=9)
    ve.add("I've been anxious about the presentation all week", emotion="anxious", importance=7)
    ve.add("Learning to play guitar has been incredibly rewarding", emotion="proud", importance=7)
    ve.add("The sunset from the mountain peak was breathtaking", emotion="serene", importance=8)

    print(f"\nIndexed {ve.size} memories  ({ve._embed_dim}-d vectors)\n")

    # Query with different moods
    for query, mood in [
        ("exercise and fitness", "excited"),
        ("family traditions",    "warm"),
        ("work problems",        "anxious"),
        ("beautiful moments",    "peaceful"),
    ]:
        print(f'Query: "{query}"  (mood: {mood})')
        results = ve.query(query, top_k=3, mood=mood)
        for r in results:
            print(f"  {r['score']:.3f}  [{r['emotion']:>12}]  {r['content'][:60]}")
        print()

    # Emotion clusters
    print("Emotion clusters:")
    for c in ve.emotion_clusters(n_clusters=3):
        print(f"  {c['label']:>12} ({c['count']} memories)")

    # Stats
    print(f"\nVividness distribution: {ve.vividness_distribution()}")

    # Visualisation demo
    import sys
    if "--viz" in sys.argv:
        print("\nLaunching dashboard...")
        ve.viz.dashboard(query_text="family and love", mood="warm")
    elif "--map" in sys.argv:
        ve.viz.embedding_map(method="tsne", colour_by="emotion")
    elif "--emotion" in sys.argv:
        ve.viz.emotion_space()
    elif "--decay" in sys.argv:
        ve.viz.decay_curves()
    elif "--heatmap" in sys.argv:
        ve.viz.similarity_heatmap()
    elif "--radar" in sys.argv:
        ve.viz.query_radar("family traditions", mood="warm")
    elif "--mood" in sys.argv:
        ve.viz.mood_comparison("beautiful experiences")
    else:
        print("\nTip: run with --viz for dashboard, --map, --emotion, "
              "--decay, --heatmap, --radar, or --mood")
