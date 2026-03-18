"""Smoke test: VividEmbed with the fine-tuned VividEmbedder model."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from VividEmbed import VividEmbed

MODEL = os.path.join(os.path.dirname(__file__), "vivid_model_output", "best")
print(f"Loading VividEmbedder from: {MODEL}")

ve = VividEmbed(model_name=MODEL)
print(f"  {ve}")
print(f"  _is_vivid_model = {ve._is_vivid_model}")
print(f"  _embed_dim = {ve._embed_dim}")

# Add some memories
ve.add("I had a wonderful birthday party with all my friends", emotion="happy", importance=8)
ve.add("My dog passed away last month and I miss him terribly", emotion="sad", importance=9)
ve.add("Got promoted to senior engineer at work today", emotion="proud", importance=7)
ve.add("I've been having terrible headaches from stress", emotion="stressed", importance=6)
ve.add("The sunset over the ocean was absolutely breathtaking", emotion="serene", importance=7)
ve.add("I failed my driving test again, so frustrating", emotion="frustrated", importance=5)
ve.add("My grandmother makes the best apple pie", emotion="nostalgic", importance=8)
ve.add("I love going for long runs in the morning", emotion="excited", importance=6)

print(f"\nIndexed {ve.size} memories ({ve._embed_dim}-d vectors)")

# Check vector dimensions
entry = ve.entries()[0]
print(f"  Vector shape: {entry.vector.shape}")
import numpy as np
print(f"  Vector norm: {np.linalg.norm(entry.vector):.4f}")

# Test mood-conditioned queries
print("\n--- Query: 'family memories' (mood=happy) ---")
for r in ve.query("family memories", mood="happy", top_k=3):
    print(f"  [{r['emotion']:>10}] {r['score']:.4f} | {r['content'][:60]}")

print("\n--- Query: 'family memories' (mood=sad) ---")
for r in ve.query("family memories", mood="sad", top_k=3):
    print(f"  [{r['emotion']:>10}] {r['score']:.4f} | {r['content'][:60]}")

print("\n--- Query: 'exercise and health' (mood=excited) ---")
for r in ve.query("exercise and health", mood="excited", top_k=3):
    print(f"  [{r['emotion']:>10}] {r['score']:.4f} | {r['content'][:60]}")

print("\n--- Query: 'work problems' (mood=stressed) ---")
for r in ve.query("work problems", mood="stressed", top_k=3):
    print(f"  [{r['emotion']:>10}] {r['score']:.4f} | {r['content'][:60]}")

# Test contradiction detection
print("\n--- Contradictions for 'I hate my job' ---")
contradictions = ve.find_contradictions("I hate my job", emotion="angry")
for c in contradictions[:3]:
    print(f"  score={c['contradiction']:.4f} | {c['content'][:60]}")

# Test batch add
print("\n--- Batch add ---")
batch = [
    {"content": "Learning piano is my new hobby", "emotion": "happy", "importance": 5},
    {"content": "The thunderstorm last night was terrifying", "emotion": "fearful", "importance": 6},
]
added = ve.add_batch(batch)
print(f"  Added {len(added)} memories, total: {ve.size}")

# Test persistence
import tempfile, shutil
tmpdir = tempfile.mkdtemp()
try:
    ve2 = VividEmbed(model_name=MODEL, persist_dir=tmpdir)
    for e in ve.entries():
        ve2.add(e.content, emotion=e.emotion, importance=e.importance)
    ve2.save()
    print(f"\nSaved to {tmpdir}")

    # Reload
    ve3 = VividEmbed(model_name=MODEL, persist_dir=tmpdir)
    print(f"  Reloaded: {ve3}")
    assert ve3.size == ve2.size, f"Size mismatch: {ve3.size} vs {ve2.size}"
    print(f"  PASS: Persistence round-trip ({ve3.size} entries)")
finally:
    shutil.rmtree(tmpdir, ignore_errors=True)

print("\n✓ All integration checks passed!")
