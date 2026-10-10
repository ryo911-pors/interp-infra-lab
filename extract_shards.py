import os
import time
import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

HOOK = "blocks.6.hook_resid_post"
CTX = 1024
DOCS_PER_SHARD = 100
N_SHARDS = 10
MAX_TOKENS_PER_DOC = 1024  

out_dir = os.path.join(os.environ["ACTS_DIR"], "gpt2_16")
os.makedirs(out_dir, exist_ok=True)

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
ds = load_dataset("NeelNanda/pile-10k", split="train")

def get_acts(chunk):
    with torch.no_grad():
        _, cache = model.run_with_cache(chunk, names_filter=HOOK, stop_at_layer=7)
    return cache[HOOK][0].cpu().numpy()

t0 = time.perf_counter()
n = 0
for shard_idx in range(N_SHARDS):
    path = os.path.join(out_dir, f"shard_{shard_idx:05d}.npy")
    if os.path.exists(path):
        print(f"skip {path}")
        continue

    first = shard_idx * DOCS_PER_SHARD
    buffer = []
    for doc in ds.select(range(first, first + DOCS_PER_SHARD)):
        tokens = model.to_tokens(doc["text"], truncate=False)[0]
        for start in range(0, min(len(tokens), MAX_TOKENS_PER_DOC), CTX):
            chunk = tokens[start:start + CTX].unsqueeze(0)
            buffer.append(get_acts(chunk))

    acts = np.concatenate(buffer)
    np.save(path, acts)
    n += len(acts)
    print(f"saved {path} {acts.shape}")

elapsed = time.perf_counter() - t0
print(f"{n} tokens in {elapsed:.2f}s = {n / elapsed:.0f} tokens/s")
