"""Quick smoke test for the training pipeline."""
import torch, time, sys
from sentence_transformers import SentenceTransformer

print("Loading model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Remove Normalize
keys = list(model._modules.keys())
last = model._modules[keys[-1]]
print(f"Last module: {type(last).__name__}")
del model._modules[keys[-1]]
print("Removed Normalize layer")

# Add test tokens
added = model.tokenizer.add_tokens(["[EMO:happy]", "[IMP:5]", "[MOOD:sad]", "[QUERY]"])
model[0].auto_model.resize_token_embeddings(len(model.tokenizer))
print(f"Added {added} tokens, vocab={len(model.tokenizer)}")

# Test forward pass
texts = ["[EMO:happy] [IMP:7] had a great day", "[EMO:sad] [IMP:3] feeling down"]
tok = model.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
print(f"Tokenized shapes: {tok['input_ids'].shape}")

t0 = time.time()
out = model(dict(tok))
dt = time.time() - t0
print(f"Forward keys: {list(out.keys())}")
emb = out["sentence_embedding"]
print(f"Embedding shape: {emb.shape}")
print(f"Norms: {torch.norm(emb, dim=1).tolist()}")
print(f"Forward took: {dt:.3f}s")

# Test backward
loss = emb.sum()
loss.backward()
print("Backward pass OK")

# Test a training step
import torch.nn.functional as F
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

for step in range(3):
    optimizer.zero_grad()
    out = model(dict(tok))
    e = out["sentence_embedding"]
    # Fake triplet loss
    cos = F.cosine_similarity(e[0:1], e[1:2])
    loss = (1.0 - cos).mean()
    loss.backward()
    optimizer.step()
    print(f"  Step {step}: loss={loss.item():.4f}")

print("\nSMOKE TEST PASSED")
