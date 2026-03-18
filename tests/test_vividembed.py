"""
test_vividembed.py  —  Visual Verification Suite for VividEmbed
================================================================
This isn't just assert-based testing — it produces a multi-page visual
report so you can SEE whether:

  1. Emotion clusters form properly (sad near sad, happy near happy)
  2. Semantic topics group (sports together, work together, etc.)
  3. Faded memories shrink / fresh ones are large
  4. Mood congruence actually shifts retrieval order
  5. Importance is reflected in the embedding space
  6. Contradictions are detected between opposing statements
  7. Persistence round-trip preserves vectors exactly
  8. Scores are sane and the ranking makes human sense

Run:  python test_vividembed.py          (headless — saves PNGs)
      python test_vividembed.py --show   (interactive — pops up windows)
"""

from __future__ import annotations
import sys, os, math, json, shutil, tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

# ── Ensure we can import from the same folder ──────────────
sys.path.insert(0, str(Path(__file__).parent))

# IMPORTANT: import torch/sentence_transformers BEFORE sklearn
# on Windows, sklearn's C extensions load DLLs that conflict with
# torch's c10.dll if torch loads second.
from VividEmbed import (
    VividEmbed, VividEntry, EMBED_DIM, EMOTION_VECTORS,
    _emotion_to_pad, _get_model, _BASE_DIM, _PAD_SCALE, _META_SCALE,
    _RECENCY_HALFLIFE, VividCortex, CoreMemory,
)
# Pre-load the model so torch is initialised before sklearn
_get_model()

# ── matplotlib setup ────────────────────────────────────────
import matplotlib
INTERACTIVE = "--show" in sys.argv
if not INTERACTIVE:
    matplotlib.use("Agg")  # headless — save to file only
else:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

OUT_DIR = Path(__file__).parent / "vividembed_test_report"
OUT_DIR.mkdir(exist_ok=True)

# ── Counters ────────────────────────────────────────────────
_pass = 0
_fail = 0
_total = 0

def check(condition: bool, label: str):
    global _pass, _fail, _total
    _total += 1
    if condition:
        _pass += 1
        print(f"  PASS: {label}")
    else:
        _fail += 1
        print(f"  FAIL: {label}")

def save_fig(fig, name: str):
    path = OUT_DIR / f"{name}.png"
    fig.savefig(str(path), dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  -> Saved: {path.name}")

def dark_fig(figsize=(14, 10)):
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("#1a1a2e")
    return fig

def dark_ax(fig, *args, **kwargs):
    ax = fig.add_subplot(*args, **kwargs)
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="#aaa")
    for s in ax.spines.values():
        s.set_color("#333")
    return ax

VALENCE_COLOURS = {
    "pos_calm":   "#4CAF50",
    "pos_active": "#FF9800",
    "neg_low":    "#2196F3",
    "neg_high":   "#F44336",
    "neutral":    "#9E9E9E",
}

def emotion_colour(emotion: str) -> str:
    pad = _emotion_to_pad(emotion)
    p, a = float(pad[0]), float(pad[1])
    if abs(p) < 0.15 and abs(a) < 0.15: return VALENCE_COLOURS["neutral"]
    if p >= 0 and a < 0.3:              return VALENCE_COLOURS["pos_calm"]
    if p >= 0:                           return VALENCE_COLOURS["pos_active"]
    if a < 0.3:                          return VALENCE_COLOURS["neg_low"]
    return VALENCE_COLOURS["neg_high"]


# ══════════════════════════════════════════════════════════════
# TEST 1:  EMOTION CLUSTERING
# Do sad memories cluster near sad, happy near happy, etc.?
# ══════════════════════════════════════════════════════════════

def test_emotion_clustering():
    print("\n" + "="*60)
    print("  TEST 1: Emotion Clustering")
    print("="*60)

    ve = VividEmbed()

    # Add memories in clear emotional groups
    sad_mems = [
        ("I cried after the phone call with my mother", "sad", 7),
        ("The funeral was the hardest day of my life", "sad", 9),
        ("I felt completely alone in the empty apartment", "lonely", 8),
        ("Losing my pet broke something inside me", "melancholy", 8),
        ("I couldn't stop the tears during the movie", "sad", 5),
    ]
    happy_mems = [
        ("The surprise birthday party was amazing", "happy", 8),
        ("I laughed so hard my sides ached", "joyful", 7),
        ("Getting the job offer made my whole week", "elated", 9),
        ("Dancing in the rain with my best friend", "playful", 7),
        ("The kids giggling at the park was infectious", "amused", 6),
    ]
    angry_mems = [
        ("The driver cut me off and I slammed the horn", "angry", 6),
        ("I was furious when they cancelled my flight", "furious", 8),
        ("The unfair decision at work made my blood boil", "frustrated", 7),
        ("Someone stole my bike right off the porch", "hostile", 8),
        ("The rude customer screamed at me for nothing", "irritated", 6),
    ]
    calm_mems = [
        ("Watching the sunrise over the lake in silence", "serene", 8),
        ("The meditation session left me perfectly still", "peaceful", 7),
        ("Sipping tea on the porch as rain fell softly", "calm", 6),
        ("The warm bath after a long hike was bliss", "relaxed", 7),
        ("Reading by the fire on a cold evening", "content", 8),
    ]

    groups = [
        ("Sad/Lonely",  sad_mems,   "#2196F3"),
        ("Happy/Joyful", happy_mems, "#FF9800"),
        ("Angry/Hostile", angry_mems, "#F44336"),
        ("Calm/Serene",  calm_mems,  "#4CAF50"),
    ]

    all_entries = []
    group_labels = []
    for group_name, mems, _ in groups:
        for content, emotion, importance in mems:
            e = ve.add(content, emotion=emotion, importance=importance)
            all_entries.append(e)
            group_labels.append(group_name)

    # ── Visualise with t-SNE ────────────────────────────────
    vecs = np.array([e.vector for e in all_entries], dtype=np.float32)
    perp = min(8, len(all_entries) - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                   init="pca").fit_transform(vecs)

    fig = dark_fig((14, 10))
    ax = dark_ax(fig, 111)
    for i, (group_name, _, colour) in enumerate(groups):
        mask = [j for j, g in enumerate(group_labels) if g == group_name]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, s=140, label=group_name,
                   edgecolors="white", linewidths=0.5, alpha=0.9, zorder=3)
        for j in mask:
            ax.annotate(all_entries[j].emotion,
                        (coords[j, 0], coords[j, 1]),
                        fontsize=7, color="white", alpha=0.8,
                        textcoords="offset points", xytext=(6, 4))

    ax.set_title("TEST 1: Do Emotions Cluster Together?  (t-SNE of 389-d vectors)",
                  color="white", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", facecolor="#1a1a2e",
              edgecolor="#444", labelcolor="white", fontsize=10)
    save_fig(fig, "01_emotion_clustering")

    # ── Quantitative check: intra-group similarity > inter-group ──
    group_indices = {}
    for i, g in enumerate(group_labels):
        group_indices.setdefault(g, []).append(i)

    sim_matrix = vecs @ vecs.T
    intra_sims = []
    inter_sims = []
    for gname, indices in group_indices.items():
        for i in indices:
            for j in indices:
                if i < j:
                    intra_sims.append(sim_matrix[i, j])
            for other_name, other_indices in group_indices.items():
                if other_name != gname:
                    for j in other_indices:
                        if i < j:
                            inter_sims.append(sim_matrix[i, j])

    avg_intra = float(np.mean(intra_sims))
    avg_inter = float(np.mean(inter_sims))
    print(f"  Avg intra-group similarity: {avg_intra:.4f}")
    print(f"  Avg inter-group similarity: {avg_inter:.4f}")
    check(avg_intra > avg_inter,
          f"Intra-group sim ({avg_intra:.3f}) > inter-group sim ({avg_inter:.3f})")

    # Check each group is tighter than average
    for gname, indices in group_indices.items():
        g_sims = [sim_matrix[i, j] for i in indices for j in indices if i < j]
        g_avg = float(np.mean(g_sims)) if g_sims else 0
        check(g_avg > avg_inter,
              f"{gname} internal similarity ({g_avg:.3f}) > inter-group avg ({avg_inter:.3f})")

    return ve


# ══════════════════════════════════════════════════════════════
# TEST 2:  SEMANTIC TOPIC GROUPING
# Do memories about similar topics cluster regardless of emotion?
# ══════════════════════════════════════════════════════════════

def test_semantic_grouping():
    print("\n" + "="*60)
    print("  TEST 2: Semantic Topic Grouping")
    print("="*60)

    ve = VividEmbed()

    topics = {
        "Sports": [
            ("I ran a personal best in the marathon today", "proud", 8),
            ("The football match ended in a dramatic penalty", "excited", 7),
            ("Swimming laps at dawn clears my head completely", "calm", 6),
            ("My boxing training has really improved my footwork", "motivated", 7),
            ("We won the basketball tournament against all odds", "triumphant", 9),
        ],
        "Work": [
            ("The quarterly review went better than expected", "relieved", 6),
            ("I got promoted to senior engineer last week", "proud", 9),
            ("The deadline pressure is crushing my motivation", "stressed", 7),
            ("My colleague undermined me in the meeting", "frustrated", 7),
            ("The new project proposal was approved by leadership", "excited", 8),
        ],
        "Food": [
            ("The homemade pasta was absolutely perfect", "satisfied", 7),
            ("I tried sushi for the first time and loved it", "curious", 6),
            ("Baking bread from scratch is so therapeutic", "peaceful", 7),
            ("The restaurant's steak was the best I've ever had", "happy", 8),
            ("Grandma's chicken soup recipe is pure comfort", "nostalgic", 9),
        ],
        "Nature": [
            ("The northern lights were utterly breathtaking", "serene", 10),
            ("Hiking through the autumn forest was magical", "peaceful", 8),
            ("The thunderstorm over the ocean was terrifying and beautiful", "fascinated", 7),
            ("A deer walked right up to our campsite at dawn", "surprised", 6),
            ("The cherry blossoms in spring never get old", "content", 7),
        ],
    }

    all_entries = []
    topic_labels = []
    topic_colours = {"Sports": "#FF5722", "Work": "#3F51B5",
                     "Food": "#8BC34A", "Nature": "#00BCD4"}

    for topic, mems in topics.items():
        for content, emotion, importance in mems:
            e = ve.add(content, emotion=emotion, importance=importance)
            all_entries.append(e)
            topic_labels.append(topic)

    vecs = np.array([e.vector for e in all_entries], dtype=np.float32)
    perp = min(8, len(all_entries) - 1)
    coords = TSNE(n_components=2, perplexity=perp, random_state=42,
                   init="pca").fit_transform(vecs)

    fig = dark_fig((14, 10))
    ax = dark_ax(fig, 111)
    for topic, colour in topic_colours.items():
        mask = [i for i, t in enumerate(topic_labels) if t == topic]
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=colour, s=140, label=topic,
                   edgecolors="white", linewidths=0.5, alpha=0.9)
        for j in mask:
            snippet = all_entries[j].content[:30] + "..."
            ax.annotate(snippet,
                        (coords[j, 0], coords[j, 1]),
                        fontsize=6, color="white", alpha=0.7,
                        textcoords="offset points", xytext=(6, 3))

    ax.set_title("TEST 2: Do Semantic Topics Cluster?  (Mixed Emotions Within Topics)",
                  color="white", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", facecolor="#1a1a2e",
              edgecolor="#444", labelcolor="white", fontsize=10)
    save_fig(fig, "02_semantic_topics")

    # Quantitative check
    topic_indices = {}
    for i, t in enumerate(topic_labels):
        topic_indices.setdefault(t, []).append(i)

    sim_matrix = vecs @ vecs.T
    intra_sims = []
    inter_sims = []
    for tname, indices in topic_indices.items():
        for i in indices:
            for j in indices:
                if i < j:
                    intra_sims.append(sim_matrix[i, j])
            for other, other_idx in topic_indices.items():
                if other != tname:
                    for j in other_idx:
                        if i < j:
                            inter_sims.append(sim_matrix[i, j])

    avg_intra = float(np.mean(intra_sims))
    avg_inter = float(np.mean(inter_sims))
    print(f"  Avg intra-topic similarity: {avg_intra:.4f}")
    print(f"  Avg inter-topic similarity: {avg_inter:.4f}")
    check(avg_intra > avg_inter,
          f"Intra-topic sim ({avg_intra:.3f}) > inter-topic sim ({avg_inter:.3f})")

    # Per-topic checks
    for tname, indices in topic_indices.items():
        t_sims = [sim_matrix[i, j] for i in indices for j in indices if i < j]
        t_avg = float(np.mean(t_sims)) if t_sims else 0
        check(t_avg > avg_inter,
              f"{tname} internal sim ({t_avg:.3f}) > inter-topic avg ({avg_inter:.3f})")

    return ve


# ══════════════════════════════════════════════════════════════
# TEST 3:  VIVIDNESS DECAY — FADED MEMORIES SHRINK
# Old memories should be small dots, fresh ones large dots.
# ══════════════════════════════════════════════════════════════

def test_vividness_decay():
    print("\n" + "="*60)
    print("  TEST 3: Vividness Decay Visualization")
    print("="*60)

    ve = VividEmbed()

    # Add memories at different "ages" via backdated timestamps
    now = datetime.now()
    memories = [
        ("Just heard the most incredible news",       "elated",    9,  0),     # now
        ("Had a great lunch with my friend today",     "happy",     7,  1),     # 1 day ago
        ("The workshop last week was enlightening",    "inspired",  8,  7),     # 1 week
        ("That road trip two weeks back was wild",     "excited",   7,  14),    # 2 weeks
        ("The concert a month ago was fantastic",      "joyful",    8,  30),    # 1 month
        ("That holiday last quarter was nice",         "content",   6,  90),    # 3 months
        ("The move to this city half a year ago",      "hopeful",   5,  180),   # 6 months
        ("My first day at this job a year ago",        "nervous",   7,  365),   # 1 year
    ]

    entries = []
    ages = []
    for content, emotion, importance, days_ago in memories:
        ts = (now - timedelta(days=days_ago)).isoformat()
        e = ve.add(content, emotion=emotion, importance=importance,
                   timestamp=ts)
        entries.append(e)
        ages.append(days_ago)

    # ── Plot: bubble chart where size = vividness ───────────
    vecs = np.array([e.vector for e in entries], dtype=np.float32)
    coords = PCA(n_components=2, random_state=42).fit_transform(vecs)

    vivids = [e.vividness for e in entries]
    max_viv = max(vivids) if vivids else 1

    fig = dark_fig((16, 8))
    gs = GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.3)

    # Left: bubble chart
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor("#16213e")
    sizes = [(v / max_viv) * 500 + 30 for v in vivids]  # scale to visible range
    colours = [emotion_colour(e.emotion) for e in entries]
    ax1.scatter(coords[:, 0], coords[:, 1],
                s=sizes, c=colours, edgecolors="white",
                linewidths=0.5, alpha=0.85, zorder=3)
    for i, e in enumerate(entries):
        lbl = f"{ages[i]}d: {e.content[:25]}..."
        ax1.annotate(lbl, (coords[i, 0], coords[i, 1]),
                     fontsize=7, color="white", alpha=0.8,
                     textcoords="offset points", xytext=(8, 5))
    ax1.set_title("Bubble Size = Vividness  (bigger = more vivid)",
                   color="white", fontsize=12, fontweight="bold")
    ax1.tick_params(colors="#666")
    for s in ax1.spines.values(): s.set_color("#333")

    # Right: vividness vs age scatter
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor("#16213e")
    ax2.scatter(ages, vivids, c=colours, s=120,
                edgecolors="white", linewidths=0.5, alpha=0.9)
    for i, e in enumerate(entries):
        ax2.annotate(f"{e.importance}imp, {e.stability:.0f}stab",
                     (ages[i], vivids[i]),
                     fontsize=7, color="white", alpha=0.7,
                     textcoords="offset points", xytext=(5, 5))

    # Theoretical decay curve
    days_range = np.linspace(0, 400, 200)
    for imp, stab, style in [(9, 3.0, "-"), (7, 3.0, "--"), (5, 3.0, ":")]:
        curve = imp * np.exp(-days_range / stab)
        ax2.plot(days_range, curve, color="#ffffff", alpha=0.3,
                 linestyle=style, linewidth=1,
                 label=f"imp={imp}, stab={stab}")

    ax2.set_xlabel("Age (days)", color="white", fontsize=11)
    ax2.set_ylabel("Vividness", color="white", fontsize=11)
    ax2.set_title("Vividness vs Age  (should decay exponentially)",
                   color="white", fontsize=12, fontweight="bold")
    ax2.legend(facecolor="#1a1a2e", edgecolor="#444",
               labelcolor="white", fontsize=8)
    ax2.tick_params(colors="#666")
    for s in ax2.spines.values(): s.set_color("#333")

    fig.suptitle("TEST 3: Faded Memories Should Be Small, Fresh Ones Large",
                  color="white", fontsize=15, fontweight="bold", y=1.02)
    save_fig(fig, "03_vividness_decay")

    # Quantitative checks
    check(entries[0].vividness > entries[-1].vividness,
          f"Fresh memory ({entries[0].vividness:.2f}) more vivid than 1-year old ({entries[-1].vividness:.6f})")
    check(entries[0].vividness > entries[4].vividness,
          f"Today's memory ({entries[0].vividness:.2f}) > 1-month old ({entries[4].vividness:.4f})")

    # Verify monotonic decay for same-importance memories
    for i in range(len(entries) - 1):
        if ages[i] < ages[i+1] and entries[i].importance >= entries[i+1].importance:
            check(entries[i].vividness >= entries[i+1].vividness,
                  f"Age {ages[i]}d viv ({entries[i].vividness:.3f}) >= age {ages[i+1]}d viv ({entries[i+1].vividness:.3f})")

    return ve


# ══════════════════════════════════════════════════════════════
# TEST 4:  MOOD CONGRUENCE — SAME QUERY, DIFFERENT RANKINGS
# The same query should return different top results under
# different moods.
# ══════════════════════════════════════════════════════════════

def test_mood_congruence():
    print("\n" + "="*60)
    print("  TEST 4: Mood Congruence Retrieval Shift")
    print("="*60)

    ve = VividEmbed()

    # Mix of memories — same importance, no age difference
    memories = [
        ("The beach holiday was absolutely wonderful", "happy", 7),
        ("I felt so overwhelmed trying to plan that trip", "anxious", 7),
        ("We had a peaceful evening watching the sunset", "serene", 7),
        ("The argument during the trip ruined everything", "angry", 7),
        ("I was grateful for the time with family", "grateful", 7),
        ("The sudden storm scared everyone on the boat", "afraid", 7),
        ("The kids building sandcastles was adorable", "warm", 7),
        ("I felt guilty about not inviting her", "guilty", 7),
    ]
    for content, emotion, importance in memories:
        ve.add(content, emotion=emotion, importance=importance)

    query = "that holiday experience"
    moods = ["happy", "anxious", "angry", "peaceful", "sad"]

    all_results = {}
    for mood in moods:
        results = ve.query(query, top_k=len(memories), mood=mood)
        all_results[mood] = results

    # ── Visualise: bar chart per mood showing score ranking ──
    fig = dark_fig((18, 10))
    n_moods = len(moods)
    for idx, mood in enumerate(moods):
        ax = dark_ax(fig, 2, 3, idx + 1)
        results = all_results[mood]
        contents = [r["content"][:35] + "..." for r in results]
        scores = [r["score"] for r in results]
        bar_colours = [emotion_colour(r["emotion"]) for r in results]
        y_pos = np.arange(len(contents))
        ax.barh(y_pos, scores, color=bar_colours, edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(contents, fontsize=6, color="white")
        ax.invert_yaxis()
        ax.set_xlabel("Score", color="white", fontsize=8)
        ax.set_title(f"Mood: {mood.upper()}",
                      color="white", fontsize=11, fontweight="bold")
        # Highlight the emotion tag
        for i, r in enumerate(results):
            ax.text(scores[i] + 0.002, i, f" [{r['emotion']}]",
                    fontsize=6, color="#ccc", va="center")

    # Use the 6th subplot as a legend/info panel
    ax_info = dark_ax(fig, 2, 3, 6)
    ax_info.axis("off")
    info_text = f'Query: "{query}"\n\nMemories: {len(memories)}\n'
    info_text += "\nColour key:\n"
    for cat, col in VALENCE_COLOURS.items():
        info_text += f"  {cat.replace('_', ' ').title()}\n"
    info_text += "\nIf mood congruence works,\npositive moods should boost\npositive memories and vice versa."
    ax_info.text(0.1, 0.9, info_text, transform=ax_info.transAxes,
                 fontsize=10, color="white", verticalalignment="top",
                 family="monospace")

    fig.suptitle("TEST 4: Same Query — Different Moods — Different Rankings?",
                  color="white", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "04_mood_congruence")

    # Quantitative: happy mood should rank happy memory higher than sad
    happy_results = all_results["happy"]
    sad_results = all_results["sad"]
    happy_top_emotions = [r["emotion"] for r in happy_results[:3]]
    sad_top_emotions = [r["emotion"] for r in sad_results[:3]]

    # Under happy mood, top result should have positive valence
    top_happy_pad = _emotion_to_pad(happy_results[0]["emotion"])
    check(float(top_happy_pad[0]) > 0,
          f"Happy mood: top result emotion '{happy_results[0]['emotion']}' has positive valence ({float(top_happy_pad[0]):.2f})")

    # Rankings should differ between opposite moods
    angry_results = all_results["angry"]
    happy_order = [r["uid"] for r in happy_results]
    angry_order = [r["uid"] for r in angry_results]
    check(happy_order != angry_order,
          "Happy vs Angry mood produce different ranking orders")

    # Under angry mood, negative memories should be ranked HIGHER than under happy mood
    # (mood weight is 10%, so it nudges rather than dominates)
    def avg_rank_neg(results):
        ranks = []
        for i, r in enumerate(results):
            pad = _emotion_to_pad(r["emotion"])
            if float(pad[0]) < 0:
                ranks.append(i)
        return np.mean(ranks) if ranks else len(results)
    check(avg_rank_neg(angry_results) < avg_rank_neg(happy_results),
          f"Angry mood ranks negative memories higher ({avg_rank_neg(angry_results):.1f}) than happy mood ({avg_rank_neg(happy_results):.1f})")

    # Peaceful mood should rank calm-positive memories higher than angry mood
    peaceful_results = all_results["peaceful"]
    def avg_rank_calm(results):
        ranks = []
        for i, r in enumerate(results):
            pad = _emotion_to_pad(r["emotion"])
            if float(pad[0]) > 0 and float(pad[1]) < 0.3:
                ranks.append(i)
        return np.mean(ranks) if ranks else len(results)
    check(avg_rank_calm(peaceful_results) <= avg_rank_calm(angry_results),
          f"Peaceful mood ranks calm memories higher ({avg_rank_calm(peaceful_results):.1f}) than angry mood ({avg_rank_calm(angry_results):.1f})")

    return ve


# ══════════════════════════════════════════════════════════════
# TEST 5:  IMPORTANCE MATTERS
# High-importance memories should score higher than trivial ones
# for equally relevant content.
# ══════════════════════════════════════════════════════════════

def test_importance_effect():
    print("\n" + "="*60)
    print("  TEST 5: Importance Effect on Retrieval")
    print("="*60)

    ve = VividEmbed()

    # Same topic, different importance levels
    ve.add("I had coffee this morning", emotion="neutral", importance=2)
    ve.add("I had a routine coffee meeting at work", emotion="neutral", importance=4)
    ve.add("The coffee shop date changed my entire life", emotion="loving", importance=9)
    ve.add("I discovered an incredible artisan coffee roaster", emotion="excited", importance=7)
    ve.add("Spilling coffee on my laptop was a disaster", emotion="stressed", importance=8)

    results = ve.query("coffee", top_k=5)

    fig = dark_fig((12, 7))
    ax = dark_ax(fig, 111)
    contents = [f"[imp={r['importance']}] {r['content'][:40]}..." for r in results]
    scores = [r["score"] for r in results]
    importances = [r["importance"] for r in results]
    bar_colours = plt.cm.YlOrRd(np.array(importances) / 10.0)

    y_pos = np.arange(len(contents))
    bars = ax.barh(y_pos, scores, color=bar_colours, edgecolor="white",
                   linewidth=0.3, alpha=0.9)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(contents, fontsize=8, color="white")
    ax.invert_yaxis()
    ax.set_xlabel("Score", color="white", fontsize=11)
    ax.set_title("TEST 5: Higher Importance = Higher Score?  (all about 'coffee')",
                  color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "05_importance_effect")

    # The highest importance coffee memory should be near the top
    check(results[0]["importance"] >= 7,
          f"Top result has importance {results[0]['importance']} (>= 7)")
    check(results[-1]["importance"] <= results[0]["importance"],
          f"Lowest-ranked ({results[-1]['importance']}) <= highest-ranked ({results[0]['importance']})")


# ══════════════════════════════════════════════════════════════
# TEST 6:  SEMANTIC RELEVANCE — RIGHT ANSWER FOR THE QUERY
# Does "exercise" find the running memory, not the cooking one?
# ══════════════════════════════════════════════════════════════

def test_semantic_retrieval():
    print("\n" + "="*60)
    print("  TEST 6: Semantic Retrieval Accuracy")
    print("="*60)

    ve = VividEmbed()

    ve.add("I ran five miles this morning in the park", emotion="proud", importance=7)
    ve.add("The cheesecake recipe turned out perfectly", emotion="satisfied", importance=7)
    ve.add("My Python code finally compiled with no errors", emotion="relieved", importance=7)
    ve.add("The guitar lesson was really challenging today", emotion="frustrated", importance=7)
    ve.add("I watched a documentary about space exploration", emotion="fascinated", importance=7)
    ve.add("The swimming pool was closed for maintenance", emotion="disappointed", importance=5)
    ve.add("We played basketball at the community center", emotion="excited", importance=7)
    ve.add("The yoga class stretched muscles I forgot I had", emotion="calm", importance=6)

    queries_expected = [
        ("exercise and fitness",  ["ran five miles", "swimming pool", "basketball", "yoga"]),
        ("cooking and baking",    ["cheesecake recipe"]),
        ("programming and code",  ["Python code"]),
        ("music practice",        ["guitar lesson"]),
        ("science and space",     ["documentary about space"]),
    ]

    fig = dark_fig((16, 12))
    for idx, (query, expected_snippets) in enumerate(queries_expected):
        ax = dark_ax(fig, 3, 2, idx + 1)
        results = ve.query(query, top_k=4)
        contents = [r["content"][:38] + "..." for r in results]
        scores = [r["score"] for r in results]

        # Colour: green if expected, grey otherwise
        bar_colours = []
        for r in results:
            found = any(snip.lower() in r["content"].lower() for snip in expected_snippets)
            bar_colours.append("#4CAF50" if found else "#666")

        y_pos = np.arange(len(contents))
        ax.barh(y_pos, scores, color=bar_colours, edgecolor="white",
                linewidth=0.3, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(contents, fontsize=7, color="white")
        ax.invert_yaxis()
        ax.set_title(f'"{query}"', color="white", fontsize=10, fontweight="bold")

        # Check top result contains expected content
        top_content = results[0]["content"].lower()
        hit = any(snip.lower() in top_content for snip in expected_snippets)
        check(hit, f'"{query}" -> top result matches expected')

    fig.suptitle("TEST 6: Semantic Retrieval  (green = expected match)",
                  color="white", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, "06_semantic_retrieval")


# ══════════════════════════════════════════════════════════════
# TEST 7:  CONTRADICTION DETECTION
# "I love X" vs "I hate X" — should flag as contradictions.
# ══════════════════════════════════════════════════════════════

def test_contradictions():
    print("\n" + "="*60)
    print("  TEST 7: Contradiction Detection")
    print("="*60)

    ve = VividEmbed()

    ve.add("I absolutely love my job and look forward to Mondays", emotion="happy", importance=8)
    ve.add("My commute to work is pleasant and relaxing", emotion="content", importance=6)
    ve.add("The team at work is supportive and collaborative", emotion="warm", importance=7)
    ve.add("I enjoy spending weekends hiking in the mountains", emotion="peaceful", importance=7)
    ve.add("My cat always cheers me up after a long day", emotion="warm", importance=6)

    # Query with contradictions
    contras = ve.find_contradictions(
        "I absolutely hate my job and dread every Monday",
        emotion="frustrated",
        threshold=0.50
    )

    fig = dark_fig((14, 7))
    ax = dark_ax(fig, 111)

    if contras:
        contents = [f"[{c['emotion']}] {c['content'][:45]}..." for c in contras]
        contra_scores = [c["contradiction"] for c in contras]
        sem_sims = [c["semantic_sim"] for c in contras]
        val_diffs = [c["valence_diff"] for c in contras]

        y_pos = np.arange(len(contents))
        bars = ax.barh(y_pos, contra_scores, color="#F44336",
                       edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(contents, fontsize=8, color="white")
        ax.invert_yaxis()

        for i, c in enumerate(contras):
            ax.text(contra_scores[i] + 0.01, i,
                    f"sem={sem_sims[i]:.2f} val_diff={val_diffs[i]:.2f}",
                    fontsize=8, color="#aaa", va="center")
    else:
        ax.text(0.5, 0.5, "No contradictions found (threshold may be too high)",
                transform=ax.transAxes, fontsize=14, color="white",
                ha="center", va="center")

    ax.set_xlabel("Contradiction Score", color="white", fontsize=11)
    ax.set_title('TEST 7: "I hate my job" vs stored "I love my job"',
                  color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "07_contradictions")

    # The job-loving memory should be flagged
    if contras:
        top_contra = contras[0]["content"].lower()
        check("love my job" in top_contra or "job" in top_contra,
              f"Top contradiction is about the job: '{contras[0]['content'][:50]}'")
        check(contras[0]["valence_diff"] > 0.6,
              f"Valence diff ({contras[0]['valence_diff']:.2f}) > 0.6")
    else:
        check(False, "Should detect at least one contradiction about the job")


# ══════════════════════════════════════════════════════════════
# TEST 8:  SIMILARITY HEATMAP — SELF-SIMILARITY STRUCTURE
# We should see clear diagonal blocks for related memories.
# ══════════════════════════════════════════════════════════════

def test_similarity_structure():
    print("\n" + "="*60)
    print("  TEST 8: Similarity Matrix Structure")
    print("="*60)

    ve = VividEmbed()

    # Add in blocks: positive, negative, practical
    positive = [
        ("The wedding was the happiest day ever", "joyful", 9),
        ("I'm so proud of what we built together", "proud", 8),
        ("The surprise gift made me cry happy tears", "loving", 8),
    ]
    negative = [
        ("The breakup devastated me for months", "sad", 8),
        ("I felt completely worthless after the rejection", "ashamed", 7),
        ("The loss of my grandmother haunts me still", "melancholy", 9),
    ]
    practical = [
        ("Need to file taxes before the deadline", "stressed", 4),
        ("Remember to pick up groceries after work", "neutral", 3),
        ("The plumber is coming Tuesday at 10am", "neutral", 3),
    ]

    groups = [("Positive", positive, "#4CAF50"),
              ("Negative", negative, "#F44336"),
              ("Practical", practical, "#9E9E9E")]

    all_entries = []
    labels = []
    for gname, mems, _ in groups:
        for content, emotion, importance in mems:
            e = ve.add(content, emotion=emotion, importance=importance)
            all_entries.append(e)
            labels.append(gname)

    vecs = np.array([e.vector for e in all_entries], dtype=np.float32)
    sim = vecs @ vecs.T

    fig = dark_fig((12, 10))
    ax = dark_ax(fig, 111)
    im = ax.imshow(sim, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    plt.colorbar(im, ax=ax, label="Cosine Similarity", shrink=0.8)

    tick_labels = [f"[{labels[i]}] {all_entries[i].content[:25]}..."
                   for i in range(len(all_entries))]
    ax.set_xticks(range(len(all_entries)))
    ax.set_yticks(range(len(all_entries)))
    ax.set_xticklabels(tick_labels, rotation=45, ha="right",
                        fontsize=7, color="white")
    ax.set_yticklabels(tick_labels, fontsize=7, color="white")

    # Draw group boundaries
    for start in [3, 6]:
        ax.axhline(y=start - 0.5, color="white", linewidth=1, alpha=0.5)
        ax.axvline(x=start - 0.5, color="white", linewidth=1, alpha=0.5)

    ax.set_title("TEST 8: Similarity Heatmap  (should see 3x3 diagonal blocks)",
                  color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig(fig, "08_similarity_heatmap")

    # Check diagonal blocks are brighter than off-diagonal
    pos_block = sim[0:3, 0:3]
    neg_block = sim[3:6, 3:6]
    prac_block = sim[6:9, 6:9]
    cross_pn = sim[0:3, 3:6]
    cross_pp = sim[0:3, 6:9]

    avg_pos = float(np.mean(pos_block[np.triu_indices(3, k=1)]))
    avg_neg = float(np.mean(neg_block[np.triu_indices(3, k=1)]))
    avg_cross = float(np.mean(cross_pn))

    check(avg_pos > avg_cross,
          f"Positive block ({avg_pos:.3f}) > pos-neg cross ({avg_cross:.3f})")
    check(avg_neg > avg_cross,
          f"Negative block ({avg_neg:.3f}) > pos-neg cross ({avg_cross:.3f})")


# ══════════════════════════════════════════════════════════════
# TEST 9:  PERSISTENCE — SAVE & LOAD ROUND TRIP
# Vectors must survive saving and reloading perfectly.
# ══════════════════════════════════════════════════════════════

def test_persistence():
    print("\n" + "="*60)
    print("  TEST 9: Persistence Round-Trip")
    print("="*60)

    tmp = tempfile.mkdtemp(prefix="vividembed_test_")
    try:
        ve1 = VividEmbed(persist_dir=tmp)
        ve1.add("Memory one about the sunrise", emotion="peaceful", importance=8)
        ve1.add("Memory two about the argument", emotion="angry", importance=6)
        ve1.add("Memory three about the concert", emotion="excited", importance=9)
        ve1.save()

        # Reload from disk
        ve2 = VividEmbed(persist_dir=tmp)

        check(ve2.size == 3, f"Reloaded {ve2.size} entries (expected 3)")

        # Compare vectors
        for e1, e2 in zip(ve1.entries(), ve2.entries()):
            check(e1.content == e2.content,
                  f"Content match: '{e1.content[:30]}'")
            check(e1.emotion == e2.emotion,
                  f"Emotion match: {e1.emotion}")
            check(e1.importance == e2.importance,
                  f"Importance match: {e1.importance}")
            vec_diff = float(np.max(np.abs(e1.vector - e2.vector)))
            check(vec_diff < 1e-6,
                  f"Vector match (max diff: {vec_diff:.2e})")

        # Query should produce identical results
        r1 = ve1.query("music and concerts", top_k=3)
        r2 = ve2.query("music and concerts", top_k=3)
        check(len(r1) == len(r2), f"Same number of results ({len(r1)})")
        if r1 and r2:
            check(r1[0]["uid"] == r2[0]["uid"],
                  f"Same top result: {r1[0]['content'][:30]}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ══════════════════════════════════════════════════════════════
# TEST 10:  PAD EMOTION SPACE — 3D SCATTER
# Do emotions land where they should in Pleasure-Arousal-Dominance?
# ══════════════════════════════════════════════════════════════

def test_pad_space():
    print("\n" + "="*60)
    print("  TEST 10: PAD Emotion Space")
    print("="*60)

    ve = VividEmbed()

    # Spread across PAD extremes
    emotion_set = [
        ("I'm having the best day of my life", "elated", 9),
        ("Everything is terrible and hopeless", "sad", 8),
        ("I want to scream at everyone", "furious", 8),
        ("I feel completely at peace with the world", "serene", 8),
        ("I'm terrified of what happens next", "panicked", 7),
        ("I feel powerful and unstoppable", "triumphant", 9),
        ("I feel utterly helpless and small", "helpless", 6),
        ("I'm quietly content with how things turned out", "content", 7),
        ("This is so boring I could cry", "bored", 4),
        ("I can't wait to start the adventure", "excited", 8),
    ]

    entries = []
    for content, emotion, importance in emotion_set:
        e = ve.add(content, emotion=emotion, importance=importance)
        entries.append(e)

    pads = np.array([_emotion_to_pad(e.emotion) for e in entries])

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor("#1a1a2e")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#16213e")

    sizes = np.array([e.importance for e in entries]) * 30
    colours = [emotion_colour(e.emotion) for e in entries]

    ax.scatter(pads[:, 0], pads[:, 1], pads[:, 2],
               c=colours, s=sizes, edgecolors="white",
               linewidths=0.5, alpha=0.85)

    for i, e in enumerate(entries):
        ax.text(pads[i, 0], pads[i, 1], pads[i, 2],
                f"  {e.emotion}", fontsize=8, color="white", alpha=0.9)

    # Mark axis meanings
    ax.set_xlabel("Pleasure  (-1=pain, +1=joy)", color="white", fontsize=10)
    ax.set_ylabel("Arousal  (-1=calm, +1=excited)", color="white", fontsize=10)
    ax.set_zlabel("Dominance  (-1=weak, +1=powerful)", color="white", fontsize=10)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.set_title("TEST 10: PAD Emotion Space\n(size=importance, position=emotion vector)",
                  color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#666")
    save_fig(fig, "10_pad_emotion_space")

    # Checks: elated should have high pleasure, furious should have low
    elated_pad = _emotion_to_pad("elated")
    furious_pad = _emotion_to_pad("furious")
    serene_pad = _emotion_to_pad("serene")
    panicked_pad = _emotion_to_pad("panicked")

    check(float(elated_pad[0]) > 0.5, f"Elated pleasure ({float(elated_pad[0]):.2f}) > 0.5")
    check(float(furious_pad[0]) < -0.5, f"Furious pleasure ({float(furious_pad[0]):.2f}) < -0.5")
    check(float(serene_pad[1]) < 0, f"Serene arousal ({float(serene_pad[1]):.2f}) < 0 (calm)")
    check(float(panicked_pad[1]) > 0.5, f"Panicked arousal ({float(panicked_pad[1]):.2f}) > 0.5 (high)")
    check(float(elated_pad[2]) > float(panicked_pad[2]),
          f"Elated dominance ({float(elated_pad[2]):.2f}) > panicked ({float(panicked_pad[2]):.2f})")


# ══════════════════════════════════════════════════════════════
# TEST 11:  VECTOR PROPERTIES
# Basic embedding properties: correct dimensionality, L2-normed,
# different texts produce different vectors.
# ══════════════════════════════════════════════════════════════

def test_vector_properties():
    print("\n" + "="*60)
    print("  TEST 11: Vector Properties")
    print("="*60)

    ve = VividEmbed()

    e1 = ve.add("The cat sat on the mat", emotion="content", importance=5)
    e2 = ve.add("Quantum computing will revolutionize cryptography", emotion="fascinated", importance=8)
    e3 = ve.add("The cat sat on the mat", emotion="sad", importance=5)  # same text, diff emotion

    check(e1.vector.shape == (EMBED_DIM,),
          f"Vector dim = {e1.vector.shape[0]} (expected {EMBED_DIM})")

    norm1 = float(np.linalg.norm(e1.vector))
    check(abs(norm1 - 1.0) < 0.01, f"L2-normalized (norm={norm1:.6f})")

    # Different texts → different vectors
    sim_12 = float(np.dot(e1.vector, e2.vector))
    check(sim_12 < 0.9, f"Different texts have sim={sim_12:.3f} < 0.9")

    # Same text, different emotion → slightly different vectors
    sim_13 = float(np.dot(e1.vector, e3.vector))
    check(sim_13 > 0.8, f"Same text, diff emotion: sim={sim_13:.3f} > 0.8 (text dominates)")
    check(sim_13 < 1.0, f"Same text, diff emotion: sim={sim_13:.3f} < 1.0 (emotion matters)")

    # Base embedding portion should be very similar for same text
    base_sim = float(np.dot(e1.vector[:_BASE_DIM], e3.vector[:_BASE_DIM]))
    check(base_sim > sim_13,
          f"Base-only sim ({base_sim:.4f}) > full sim ({sim_13:.4f}) for same text diff emotion")


# ══════════════════════════════════════════════════════════════
# TEST 12:  ENTITY & SOURCE FILTERING
# Filters should work correctly.
# ══════════════════════════════════════════════════════════════

def test_filters():
    print("\n" + "="*60)
    print("  TEST 12: Entity & Source Filtering")
    print("="*60)

    ve = VividEmbed()

    ve.add("Alex loves hiking in the mountains", emotion="happy", importance=7,
           source="social", entity="Alex")
    ve.add("Alex's new job starts Monday", emotion="excited", importance=6,
           source="social", entity="Alex")
    ve.add("I need to prepare for my presentation", emotion="anxious", importance=8,
           source="reflection", entity="")
    ve.add("Jordan mentioned they enjoy cooking", emotion="warm", importance=5,
           source="social", entity="Jordan")
    ve.add("Note: refactor the database layer", emotion="neutral", importance=6,
           source="task", entity="")

    # Entity filter
    alex_results = ve.query("activities", entity_filter="Alex")
    check(all(r["entity"].lower() == "alex" for r in alex_results),
          f"Entity filter: all results are about Alex ({len(alex_results)} found)")
    check(len(alex_results) == 2, f"Found {len(alex_results)} Alex memories (expected 2)")

    # Source filter
    social_results = ve.query("hobbies", source_filter="social")
    check(all(r["source"] == "social" for r in social_results),
          f"Source filter: all results are social ({len(social_results)} found)")

    # Min importance filter
    hi_results = ve.query("work", min_importance=7)
    check(all(r["importance"] >= 7 for r in hi_results),
          f"Min importance filter: all >= 7 ({len(hi_results)} results)")


# ══════════════════════════════════════════════════════════════
# TEST 13:  INDEX MANAGEMENT — add, remove, update, clear
# ══════════════════════════════════════════════════════════════

def test_index_management():
    print("\n" + "="*60)
    print("  TEST 13: Index Management")
    print("="*60)

    ve = VividEmbed()

    e1 = ve.add("First memory", importance=5)
    e2 = ve.add("Second memory", importance=5)
    e3 = ve.add("Third memory", importance=5)
    check(ve.size == 3, f"Size after 3 adds: {ve.size}")

    # Remove
    removed = ve.remove(e2.uid)
    check(removed, "remove() returned True")
    check(ve.size == 2, f"Size after remove: {ve.size}")
    check(ve.get(e2.uid) is None, "Removed entry is gone")

    # Remove non-existent
    check(not ve.remove("fake_uid"), "remove() returns False for unknown uid")

    # Update importance
    old_imp = ve.get(e1.uid).importance
    ve.update_importance(e1.uid, 10)
    check(ve.get(e1.uid).importance == 10,
          f"Importance updated: {old_imp} -> {ve.get(e1.uid).importance}")

    # Importance clamping
    ve.update_importance(e1.uid, 15)
    check(ve.get(e1.uid).importance == 10, "Importance clamped to 10")
    ve.update_importance(e1.uid, -5)
    check(ve.get(e1.uid).importance == 1, "Importance clamped to 1")

    # Clear
    ve.clear()
    check(ve.size == 0, f"Size after clear: {ve.size}")

    # Query on empty index returns empty
    results = ve.query("anything")
    check(len(results) == 0, "Query on empty index returns []")


# ══════════════════════════════════════════════════════════════
# TEST 14:  BATCH ADD & VividnessMem BRIDGE
# ══════════════════════════════════════════════════════════════

def test_batch_and_bridge():
    print("\n" + "="*60)
    print("  TEST 14: Batch Add & VividnessMem Bridge")
    print("="*60)

    ve = VividEmbed()

    items = [
        {"content": "Batch item one about dogs", "emotion": "happy", "importance": 7},
        {"content": "Batch item two about cats", "emotion": "warm", "importance": 6},
        {"content": "Batch item three about birds", "emotion": "curious", "importance": 5},
    ]
    result = ve.add_batch(items)
    check(len(result) == 3, f"add_batch returned {len(result)} entries")
    check(ve.size == 3, f"Index size = {ve.size}")

    # Empty batch
    result2 = ve.add_batch([])
    check(len(result2) == 0, "Empty batch returns []")

    # Bridge from VividnessMem format
    ve2 = VividEmbed()
    fake_memories = [
        {
            "content": "Scott loves boxing",
            "emotion": "excited",
            "importance": 8,
            "stability": 5.0,
            "source": "social",
            "entity": "Scott",
            "timestamp": datetime.now().isoformat(),
        },
        {
            "content": "Luna enjoys painting",
            "emotion": "peaceful",
            "importance": 7,
            "stability": 4.0,
            "source": "social",
            "entity": "Luna",
            "timestamp": datetime.now().isoformat(),
        },
    ]
    count = ve2.index_from_vividnessmem(fake_memories)
    check(count == 2, f"Bridge imported {count} memories")
    check(ve2.size == 2, f"Index size = {ve2.size}")


# ══════════════════════════════════════════════════════════════
# TEST 15:  QUERY BY EMOTION
# "Find memories that FEEL like X"
# ══════════════════════════════════════════════════════════════

def test_query_by_emotion():
    print("\n" + "="*60)
    print("  TEST 15: Query by Emotion")
    print("="*60)

    ve = VividEmbed()

    ve.add("The promotion was so deserved", emotion="proud", importance=8)
    ve.add("I sobbed watching the ending of that film", emotion="sad", importance=6)
    ve.add("The traffic jam made me snap", emotion="irritated", importance=5)
    ve.add("Floating in the lake at sunset was heavenly", emotion="serene", importance=8)
    ve.add("I was shaking during the earthquake", emotion="afraid", importance=9)
    ve.add("We laughed until we couldn't breathe", emotion="joyful", importance=7)

    # Query for sad-like memories
    sad_results = ve.query_by_emotion("melancholy", top_k=3)
    check(len(sad_results) > 0, f"Got {len(sad_results)} results for 'melancholy'")
    if sad_results:
        # Top result should have negative pleasure (sad-like)
        top_pad = _emotion_to_pad(sad_results[0]["emotion"])
        check(float(top_pad[0]) < 0,
              f"Top melancholy match: '{sad_results[0]['emotion']}' (pleasure={float(top_pad[0]):.2f})")

    # Query for calm-like memories
    calm_results = ve.query_by_emotion("peaceful", top_k=3)
    if calm_results:
        top_pad = _emotion_to_pad(calm_results[0]["emotion"])
        check(float(top_pad[0]) > 0 and float(top_pad[1]) < 0,
              f"Top peaceful match: '{calm_results[0]['emotion']}' is calm-positive")

    # Unknown emotion returns empty
    unknown = ve.query_by_emotion("xyzzy_not_real", top_k=3)
    check(len(unknown) == 0, "Unknown emotion returns empty list")


# ══════════════════════════════════════════════════════════════
# TEST 16:  EMOTION CLUSTERS ANALYTICS
# ══════════════════════════════════════════════════════════════

def test_emotion_clusters_api():
    print("\n" + "="*60)
    print("  TEST 16: Emotion Clusters API")
    print("="*60)

    ve = VividEmbed()
    for i in range(15):
        emotions = ["happy", "sad", "angry", "calm", "excited"]
        e = emotions[i % 5]
        ve.add(f"Memory number {i} feeling {e}", emotion=e, importance=5 + (i % 4))

    clusters = ve.emotion_clusters(n_clusters=3)
    check(len(clusters) == 3, f"Got {len(clusters)} clusters (expected 3)")
    total_entries = sum(c["count"] for c in clusters)
    check(total_entries == 15, f"Total entries across clusters = {total_entries} (expected 15)")
    for c in clusters:
        check("label" in c and "centroid" in c and "entries" in c,
              f"Cluster '{c['label']}' has required fields")

    # Fewer entries than clusters → single "all" cluster
    ve2 = VividEmbed()
    ve2.add("Only one memory", importance=5)
    c2 = ve2.emotion_clusters(n_clusters=5)
    check(len(c2) == 1, f"Fewer entries than clusters: got {len(c2)} cluster")


# ══════════════════════════════════════════════════════════════
# TEST 17:  VIVIDNESS DISTRIBUTION
# ══════════════════════════════════════════════════════════════

def test_vividness_distribution():
    print("\n" + "="*60)
    print("  TEST 17: Vividness Distribution API")
    print("="*60)

    ve = VividEmbed()

    # Empty
    dist = ve.vividness_distribution()
    check(dist["count"] == 0, "Empty index: count=0")

    # With entries
    ve.add("High importance fresh memory", importance=10)
    ve.add("Low importance fresh memory", importance=2)
    now = datetime.now()
    old_ts = (now - timedelta(days=100)).isoformat()
    ve.add("Old memory faded away", importance=5, timestamp=old_ts)

    dist = ve.vividness_distribution()
    check(dist["count"] == 3, f"Count = {dist['count']}")
    check("mean" in dist and "median" in dist and "std" in dist and
          "min" in dist and "max" in dist and "faded" in dist and "vivid" in dist,
          "Has all expected stats keys")
    check(dist["max"] > dist["min"], f"max ({dist['max']:.2f}) > min ({dist['min']:.4f})")
    print(f"  Distribution: {dist}")


# ══════════════════════════════════════════════════════════════
# TEST 18:  EDGE CASES
# ══════════════════════════════════════════════════════════════

def test_edge_cases():
    print("\n" + "="*60)
    print("  TEST 18: Edge Cases")
    print("="*60)

    ve = VividEmbed()

    # Unknown emotion defaults to neutral PAD
    e = ve.add("test memory", emotion="xyzzy_unknown", importance=5)
    pad = _emotion_to_pad("xyzzy_unknown")
    check(np.allclose(pad, 0), "Unknown emotion maps to zero PAD vector")
    check(e.vector.shape == (EMBED_DIM,), "Vector still correct dimension")

    # Empty string content
    e2 = ve.add("", emotion="happy", importance=5)
    check(e2.vector.shape == (EMBED_DIM,), "Empty content still produces valid vector")

    # Importance clamping at add time
    e3 = ve.add("over nine thousand", importance=9001)
    check(e3.importance == 10, f"Importance clamped to 10 (was 9001)")
    e4 = ve.add("below zero", importance=-5)
    check(e4.importance == 1, f"Importance clamped to 1 (was -5)")

    # Stability floor
    e5 = ve.add("zero stability", stability=-100)
    check(e5.stability >= 0.1, f"Stability floored to {e5.stability}")

    # Custom weights
    ve2 = VividEmbed(weights={"semantic": 1.0, "vividness": 0, "mood": 0, "recency": 0})
    ve2.add("pure semantic test", importance=5)
    r = ve2.query("semantic test", top_k=1)
    check(len(r) == 1, "Pure-semantic-weight query works")

    # Repr
    rep = repr(ve)
    check("VividEmbed" in rep and str(ve.size) in rep,
          f"repr: {rep}")

    # include_vector flag
    ve.add("vector test", importance=5)
    r_with = ve.query("vector test", top_k=1, include_vector=True)
    r_without = ve.query("vector test", top_k=1, include_vector=False)
    check("vector" in r_with[0], "include_vector=True includes vector")
    check("vector" not in r_without[0], "include_vector=False excludes vector")

    # Mood as raw tuple
    r_tuple = ve.query("anything", mood=(0.5, 0.3, 0.2), top_k=1)
    check(len(r_tuple) >= 1, "Mood as raw PAD tuple works")


# ══════════════════════════════════════════════════════════════
# TEST 19:  COMBINED VIEW — THE BIG PICTURE
# A final dashboard showing everything working together.
# ══════════════════════════════════════════════════════════════

def test_combined_dashboard():
    print("\n" + "="*60)
    print("  TEST 19: Combined Dashboard")
    print("="*60)

    ve = VividEmbed()
    now = datetime.now()

    # Rich set of memories simulating a real person
    real_memories = [
        # Fresh positive
        ("Got a promotion today, I'm over the moon!", "elated", 10, 0, "reflection"),
        ("Beautiful walk through the autumn park", "serene", 7, 0, "reflection"),
        # Recent mixed
        ("The argument with Sarah about the budget", "frustrated", 6, 2, "social"),
        ("Amazing Thai food at the new restaurant", "satisfied", 7, 3, "reflection"),
        ("Worried about the medical test results", "anxious", 8, 4, "reflection"),
        # Older positive
        ("Our wedding anniversary trip to Italy", "loving", 9, 30, "reflection"),
        ("Graduating from university was surreal", "proud", 9, 60, "reflection"),
        # Older negative
        ("Getting laid off from the startup", "devastated", 8, 45, "reflection"),
        ("The car accident that shook me up", "afraid", 8, 90, "reflection"),
        # Very old
        ("My first pet dog when I was young", "nostalgic", 7, 365, "reflection"),
        # Practical
        ("Dentist appointment next Thursday", "neutral", 3, 1, "task"),
        ("Fix the leaking bathroom faucet", "stressed", 4, 5, "task"),
    ]

    entries = []
    for content, emotion, importance, days_ago, source in real_memories:
        ts = (now - timedelta(days=days_ago)).isoformat()
        e = ve.add(content, emotion=emotion, importance=importance,
                   source=source, timestamp=ts)
        entries.append(e)

    vecs = np.array([e.vector for e in entries], dtype=np.float32)

    fig = plt.figure(figsize=(22, 18))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("TEST 19: VividEmbed — Full System Dashboard",
                  color="white", fontsize=18, fontweight="bold", y=0.98)
    gs = GridSpec(3, 3, hspace=0.35, wspace=0.3)

    # ── Panel 1: PCA Map coloured by emotion ────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#16213e")
    coords = PCA(n_components=2, random_state=42).fit_transform(vecs)
    colours = [emotion_colour(e.emotion) for e in entries]
    sizes = [(e.vividness / max(ev.vividness for ev in entries)) * 300 + 30
             for e in entries]
    ax1.scatter(coords[:, 0], coords[:, 1], c=colours, s=sizes,
                edgecolors="white", linewidths=0.5, alpha=0.85)
    for i, e in enumerate(entries):
        ax1.annotate(e.emotion, (coords[i, 0], coords[i, 1]),
                     fontsize=6, color="white", alpha=0.7,
                     textcoords="offset points", xytext=(5, 3))
    ax1.set_title("Memory Map (size=vividness)", color="white", fontsize=11)
    ax1.tick_params(colors="#666")
    for s in ax1.spines.values(): s.set_color("#333")

    # ── Panel 2: Decay curves ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#16213e")
    days_range = np.linspace(0, 60, 200)
    for e in entries:
        curve = e.importance * np.exp(-days_range / max(e.stability, 0.1))
        ax2.plot(days_range, curve, color=emotion_colour(e.emotion),
                 alpha=0.5, linewidth=1)
        ax2.scatter([e.age_days], [e.vividness],
                    color=emotion_colour(e.emotion),
                    s=40, zorder=5, edgecolors="white", linewidths=0.3)
    ax2.set_xlabel("Days", color="white", fontsize=9)
    ax2.set_ylabel("Vividness", color="white", fontsize=9)
    ax2.set_title("Decay Curves", color="white", fontsize=11)
    ax2.axhline(y=1.0, color="#F44336", linestyle="--", alpha=0.3)
    ax2.tick_params(colors="#666")
    for s in ax2.spines.values(): s.set_color("#333")

    # ── Panel 3: Similarity heatmap ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor("#16213e")
    sim = vecs @ vecs.T
    im = ax3.imshow(sim, cmap="RdYlGn", vmin=-0.1, vmax=1.0)
    plt.colorbar(im, ax=ax3, shrink=0.7)
    labels = [e.emotion[:6] for e in entries]
    ax3.set_xticks(range(len(entries)))
    ax3.set_yticks(range(len(entries)))
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=6, color="white")
    ax3.set_yticklabels(labels, fontsize=6, color="white")
    ax3.set_title("Similarity Matrix", color="white", fontsize=11)

    # ── Panel 4: Query "happy moments" ──────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_facecolor("#16213e")
    r_happy = ve.query("happy moments and celebrations", top_k=5, mood="happy")
    if r_happy:
        names = [f"[{r['emotion']}] {r['content'][:30]}..." for r in r_happy]
        scores = [r["score"] for r in r_happy]
        bcol = [emotion_colour(r["emotion"]) for r in r_happy]
        y_pos = np.arange(len(names))
        ax4.barh(y_pos, scores, color=bcol, edgecolor="white", linewidth=0.3)
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(names, fontsize=7, color="white")
        ax4.invert_yaxis()
    ax4.set_title('Query: "happy moments" (mood: happy)', color="white", fontsize=10)
    ax4.tick_params(colors="#666")
    for s in ax4.spines.values(): s.set_color("#333")

    # ── Panel 5: Query "worries and stress" ─────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_facecolor("#16213e")
    r_stress = ve.query("worries and difficult times", top_k=5, mood="anxious")
    if r_stress:
        names = [f"[{r['emotion']}] {r['content'][:30]}..." for r in r_stress]
        scores = [r["score"] for r in r_stress]
        bcol = [emotion_colour(r["emotion"]) for r in r_stress]
        y_pos = np.arange(len(names))
        ax5.barh(y_pos, scores, color=bcol, edgecolor="white", linewidth=0.3)
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(names, fontsize=7, color="white")
        ax5.invert_yaxis()
    ax5.set_title('Query: "worries" (mood: anxious)', color="white", fontsize=10)
    ax5.tick_params(colors="#666")
    for s in ax5.spines.values(): s.set_color("#333")

    # ── Panel 6: Vividness distribution histogram ───────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#16213e")
    vivs = [e.vividness for e in entries]
    ax6.hist(vivs, bins=10, color="#2196F3", edgecolor="white",
             linewidth=0.5, alpha=0.85)
    ax6.axvline(x=np.mean(vivs), color="#FF9800", linestyle="--",
                label=f"Mean: {np.mean(vivs):.2f}")
    ax6.set_xlabel("Vividness", color="white", fontsize=9)
    ax6.set_ylabel("Count", color="white", fontsize=9)
    ax6.set_title("Vividness Distribution", color="white", fontsize=11)
    ax6.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=8)
    ax6.tick_params(colors="#666")
    for s in ax6.spines.values(): s.set_color("#333")

    # ── Panel 7: Importance vs Score ────────────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_facecolor("#16213e")
    all_r = ve.query("life experiences", top_k=len(entries))
    if all_r:
        imps = [r["importance"] for r in all_r]
        scrs = [r["score"] for r in all_r]
        ax7.scatter(imps, scrs, c=[emotion_colour(r["emotion"]) for r in all_r],
                    s=100, edgecolors="white", linewidths=0.5)
        for r in all_r:
            ax7.annotate(r["emotion"][:5], (r["importance"], r["score"]),
                         fontsize=6, color="white", alpha=0.7,
                         textcoords="offset points", xytext=(3, 3))
    ax7.set_xlabel("Importance", color="white", fontsize=9)
    ax7.set_ylabel("Score", color="white", fontsize=9)
    ax7.set_title("Importance vs Score", color="white", fontsize=11)
    ax7.tick_params(colors="#666")
    for s in ax7.spines.values(): s.set_color("#333")

    # ── Panel 8: Age vs Score ───────────────────────────────
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_facecolor("#16213e")
    if all_r:
        age_d = [r["age_days"] for r in all_r]
        scrs = [r["score"] for r in all_r]
        ax8.scatter(age_d, scrs, c=[emotion_colour(r["emotion"]) for r in all_r],
                    s=100, edgecolors="white", linewidths=0.5)
    ax8.set_xlabel("Age (days)", color="white", fontsize=9)
    ax8.set_ylabel("Score", color="white", fontsize=9)
    ax8.set_title("Age vs Score (recency effect)", color="white", fontsize=11)
    ax8.tick_params(colors="#666")
    for s in ax8.spines.values(): s.set_color("#333")

    # ── Panel 9: Score components pie chart for top result ──
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_facecolor("#16213e")
    ax9.text(0.5, 0.5,
             f"Memories: {ve.size}\n"
             f"Dim: {EMBED_DIM}\n"
             f"Fresh (viv>5): {sum(1 for e in entries if e.vividness >= 5)}\n"
             f"Faded (viv<1): {sum(1 for e in entries if e.vividness < 1)}\n"
             f"Sources: {len(set(e.source for e in entries))}\n"
             f"Emotions: {len(set(e.emotion for e in entries))}\n",
             transform=ax9.transAxes, fontsize=12, color="white",
             ha="center", va="center", family="monospace")
    ax9.set_title("Index Stats", color="white", fontsize=11)
    ax9.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, "19_combined_dashboard")

    # Basic sanity
    check(ve.size == len(real_memories), f"All {len(real_memories)} memories indexed")
    check(len(r_happy) > 0, "Happy query returned results")
    check(len(r_stress) > 0, "Stress query returned results")


# ══════════════════════════════════════════════════════════════
# TEST 20:  CORE MEMORY
# Letta-style always-in-context memory blocks.
# ══════════════════════════════════════════════════════════════

def test_core_memory():
    print("\n" + "="*60)
    print("  TEST 20: Core Memory")
    print("="*60)

    core = CoreMemory(
        persona="Luna is a warm, witty AI companion who loves wordplay.",
        user="Scott is a developer who enjoys boxing and AI projects.",
        system="Always be supportive. Never break character.",
    )

    # Render includes all non-empty blocks
    rendered = core.render()
    check('type="persona"' in rendered, "Render includes persona block")
    check('type="user"' in rendered, "Render includes user block")
    check('type="system"' in rendered, "Render includes system block")
    check('type="scratch"' not in rendered, "Render excludes empty scratch")

    # Update block
    core.update_block("scratch", "Temp note: Scott mentioned a deadline Friday.")
    check("deadline Friday" in core.scratch, "Scratch updated")
    check('type="scratch"' in core.render(), "Scratch now in render")

    # Append to block
    core.append_to_block("user", "Scott also likes hiking.")
    check("hiking" in core.user, "Append added hiking")
    check("boxing" in core.user, "Original content preserved")

    # Invalid block name is ignored
    core.update_block("hacker", "should not work")
    check(not hasattr(core, "hacker"), "Invalid block ignored")

    # Serialization round-trip
    d = core.to_dict()
    core2 = CoreMemory.from_dict(d)
    check(core2.persona == core.persona, "Round-trip preserves persona")
    check(core2.user == core.user, "Round-trip preserves user")
    check(core2.scratch == core.scratch, "Round-trip preserves scratch")


# ══════════════════════════════════════════════════════════════
# TEST 21:  VIVID CORTEX — MOCK LLM TESTS
# Test the intelligence layer without a real LLM.
# ══════════════════════════════════════════════════════════════

def _mock_llm(messages: list[dict]) -> str:
    """A fake LLM that returns canned JSON responses based on the
    system prompt content, so we can test VividCortex without GPU."""
    system = messages[0]["content"] if messages else ""
    user_msg = messages[-1]["content"] if messages else ""

    # Query decomposition
    if "query optimizer" in system.lower():
        if "sarah" in user_msg.lower():
            return '["conversations with Sarah", "events involving Sarah"]'
        return f'["{user_msg}"]'

    # Memory extraction
    if "memory extraction" in system.lower():
        if "boxing" in user_msg.lower() or "training" in user_msg.lower():
            return json.dumps([{
                "content": "Scott trains boxing three times a week.",
                "emotion": "motivated",
                "importance": 7,
                "entity": "Scott",
                "source": "conversation",
            }])
        return "[]"

    # Memory edit operations
    if "memory management" in system.lower():
        # Look for a uid in the stored memories to promote
        import re
        uids = re.findall(r'uid=(\w+)', user_msg)
        if uids:
            return json.dumps([{"op": "PROMOTE", "uid": uids[0], "new_importance": 9}])
        return "[]"

    # Reflection
    if "reflection system" in system.lower():
        return json.dumps({
            "insights": ["The user has many memories about work stress."],
            "contradictions": [],
            "promotions": [],
            "demotions": [],
        })

    return "[]"


def test_cortex_query_decomposition():
    print("\n" + "="*60)
    print("  TEST 21: Cortex Query Decomposition")
    print("="*60)

    ve = VividEmbed()
    ve.add("Had a long chat with Sarah about the project", emotion="interested", importance=7, entity="Sarah")
    ve.add("Sarah and I argued about the budget", emotion="frustrated", importance=8, entity="Sarah")
    ve.add("Went for a run in the park", emotion="calm", importance=5)
    ve.add("The meeting with the client went well", emotion="relieved", importance=6)

    cortex = VividCortex(embed=ve, llm=_mock_llm)

    # Decomposed query: "that thing with Sarah" → sub-queries about Sarah
    results = cortex.query("that thing with Sarah", decompose=True)
    check(len(results) > 0, f"Decomposed query returned {len(results)} results")
    check(all("sarah" in r["entity"].lower() for r in results if r["entity"]),
          "Decomposed query found Sarah memories")

    # Direct query (no decomposition) for comparison
    direct = cortex.query("that thing with Sarah", decompose=False)
    check(len(direct) > 0, f"Direct query returned {len(direct)} results")

    # Decomposition should find at least as many relevant results
    sarah_decomposed = [r for r in results if "sarah" in r.get("entity", "").lower()]
    sarah_direct = [r for r in direct if "sarah" in r.get("entity", "").lower()]
    check(len(sarah_decomposed) >= len(sarah_direct),
          f"Decomposition found >= direct ({len(sarah_decomposed)} vs {len(sarah_direct)})")


def test_cortex_conversation_processing():
    print("\n" + "="*60)
    print("  TEST 22: Cortex Conversation Processing")
    print("="*60)

    ve = VividEmbed()
    cortex = VividCortex(embed=ve, llm=_mock_llm, auto_extract=True)

    # Process user turn mentioning boxing
    extracted = cortex.process_turn(
        "user", "I've been doing boxing training three times a week lately."
    )
    check(len(extracted) > 0, f"Extracted {len(extracted)} memories from conversation")
    if extracted:
        check("boxing" in extracted[0].content.lower(),
              f"Extracted memory about boxing: '{extracted[0].content[:50]}'")
        check(extracted[0].entity == "Scott", f"Entity = '{extracted[0].entity}'")

    # Process assistant turn (should NOT extract)
    extracted2 = cortex.process_turn(
        "assistant", "That's great! Boxing is an amazing workout."
    )
    check(len(extracted2) == 0, "No extraction from assistant turns")

    # Working memory should have both turns
    check(len(cortex.working_memory) == 2, f"Working memory has {len(cortex.working_memory)} turns")

    # Archival should have the extracted memory
    check(ve.size >= 1, f"Archival has {ve.size} entries")


def test_cortex_context_block():
    print("\n" + "="*60)
    print("  TEST 23: Cortex Context Block")
    print("="*60)

    ve = VividEmbed()
    ve.add("Scott's favourite food is pizza", emotion="happy", importance=7, entity="Scott")
    ve.add("Scott works as a developer", emotion="neutral", importance=6, entity="Scott")

    core = CoreMemory(
        persona="Luna is a friendly companion.",
        user="Scott likes boxing.",
    )
    cortex = VividCortex(embed=ve, llm=_mock_llm, core=core, auto_extract=False)

    # Add a user turn so there's context to retrieve from
    cortex.process_turn("user", "What do you know about me?")

    block = cortex.get_context_block()
    check('type="persona"' in block, "Context includes persona")
    check('type="user"' in block, "Context includes user block")
    check("<retrieved_memories>" in block, "Context includes retrieved memories")
    check("Scott" in block, "Context mentions Scott")


def test_cortex_edit_memories():
    print("\n" + "="*60)
    print("  TEST 24: Cortex Agentic Memory Edit")
    print("="*60)

    ve = VividEmbed()
    e1 = ve.add("Scott likes pizza", emotion="happy", importance=5, entity="Scott")

    cortex = VividCortex(embed=ve, llm=_mock_llm)

    # The mock LLM will promote the first uid it finds
    ops = cortex.edit_memories("Scott really loves pizza, it's his absolute favourite")
    check(len(ops) > 0, f"Edit returned {len(ops)} operations")
    if ops:
        check(ops[0]["op"] == "PROMOTE", f"Operation type: {ops[0]['op']}")


def test_cortex_reflect():
    print("\n" + "="*60)
    print("  TEST 25: Cortex Reflection")
    print("="*60)

    ve = VividEmbed()
    for i in range(5):
        ve.add(f"Work memory {i}: deadline pressure", emotion="stressed", importance=6)

    cortex = VividCortex(embed=ve, llm=_mock_llm)
    result = cortex.reflect()

    check("insights" in result, "Reflection has insights")
    check(isinstance(result["insights"], list), "Insights is a list")
    if result["insights"]:
        check("work" in result["insights"][0].lower() or "stress" in result["insights"][0].lower(),
              f"Insight about work/stress: '{result['insights'][0][:60]}'")


def test_cortex_persistence():
    print("\n" + "="*60)
    print("  TEST 26: Cortex Persistence")
    print("="*60)

    tmp = tempfile.mkdtemp(prefix="cortex_test_")
    try:
        ve1 = VividEmbed(persist_dir=tmp)
        ve1.add("Test memory for persistence", importance=7)

        core1 = CoreMemory(persona="Test persona", user="Test user info")
        cortex1 = VividCortex(embed=ve1, llm=_mock_llm, core=core1)
        cortex1.save()

        # Reload
        ve2 = VividEmbed(persist_dir=tmp)
        cortex2 = VividCortex(embed=ve2, llm=_mock_llm)
        cortex2.load_core()

        check(ve2.size == 1, f"Reloaded {ve2.size} archival entries")
        check(cortex2.core.persona == "Test persona", "Core persona restored")
        check(cortex2.core.user == "Test user info", "Core user restored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_cortex_repr():
    print("\n" + "="*60)
    print("  TEST 27: Cortex repr")
    print("="*60)

    ve = VividEmbed()
    ve.add("test", importance=5)
    core = CoreMemory(persona="x", user="y")
    cortex = VividCortex(embed=ve, llm=_mock_llm, core=core)
    rep = repr(cortex)
    check("VividCortex" in rep, f"repr: {rep}")
    check("archival=1" in rep, "repr shows archival count")
    check("core_blocks=2" in rep, "repr shows core block count")


def test_cortex_json_parsing():
    print("\n" + "="*60)
    print("  TEST 28: Cortex JSON Parsing Edge Cases")
    print("="*60)

    # Clean JSON
    check(VividCortex._parse_json_array('["a","b"]') == ["a", "b"],
          "Parse clean array")

    # With markdown fences
    check(VividCortex._parse_json_array('```json\n["x"]\n```') == ["x"],
          "Parse fenced array")

    # With surrounding text
    check(VividCortex._parse_json_array('Here is the result: ["y"]') == ["y"],
          "Parse array with surrounding text")

    # Invalid JSON
    check(VividCortex._parse_json_array('not json at all') == [],
          "Invalid JSON returns []")

    # Object parsing
    check(VividCortex._parse_json_object('{"a": 1}') == {"a": 1},
          "Parse clean object")

    # Fenced object
    check(VividCortex._parse_json_object('```\n{"b": 2}\n```') == {"b": 2},
          "Parse fenced object")

    # Invalid object
    check(VividCortex._parse_json_object('nope') == {},
          "Invalid JSON object returns {}")


# ══════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print()
    print("=" * 60)
    print("  VividEmbed Visual Verification Suite")
    print("=" * 60)
    mode = "INTERACTIVE (--show)" if INTERACTIVE else "HEADLESS (saving PNGs)"
    print(f"  Mode: {mode}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 60)

    test_emotion_clustering()
    test_semantic_grouping()
    test_vividness_decay()
    test_mood_congruence()
    test_importance_effect()
    test_semantic_retrieval()
    test_contradictions()
    test_similarity_structure()
    test_persistence()
    test_pad_space()
    test_vector_properties()
    test_filters()
    test_index_management()
    test_batch_and_bridge()
    test_query_by_emotion()
    test_emotion_clusters_api()
    test_vividness_distribution()
    test_edge_cases()
    test_combined_dashboard()

    # Tier 3 — VividCortex tests
    test_core_memory()
    test_cortex_query_decomposition()
    test_cortex_conversation_processing()
    test_cortex_context_block()
    test_cortex_edit_memories()
    test_cortex_reflect()
    test_cortex_persistence()
    test_cortex_repr()
    test_cortex_json_parsing()

    print()
    print("=" * 60)
    print(f"  RESULTS: {_pass}/{_total} passed, {_fail} failed")
    print("=" * 60)
    if _fail == 0:
        print("  ALL TESTS PASSED")
    else:
        print(f"  {_fail} FAILURES — check output above")
    print(f"\n  Visual report saved to: {OUT_DIR}")
    print(f"  PNGs: {list(OUT_DIR.glob('*.png'))}")
    print()

    if INTERACTIVE:
        plt.show()
