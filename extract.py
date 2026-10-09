import os
import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

HOOK = "blocks.6.hook_resid_post"
CTX = 1024
TARGET = 10_000

acts_dir = os.environ["ACTS_DIR"]
os.makedirs(acts_dir, exist_ok=True)

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
ds = load_dataset("NeelNanda/pile-10k", split="train")

collected = []
n = 0
for doc in ds:
    tokens = model.to_tokens(doc["text"], truncate=False)[0]
    for start in range(0, len(tokens), CTX):
        chunk = tokens[start:start+CTX].unsqueeze(0)
        with torch.no_grad():
            _, cache = model.run_with_cache(chunk, names_filter=HOOK)
        acts = cache[HOOK][0] #[tokens, 768]
        collected.append(acts.cpu().numpy()) #to cpu from GPU, and change to numpy
        n += acts.shape[0] #counting the tokens

        if n >= TARGET:
            break

    if n >= TARGET:
        break

all_acts = np.concatenate(collected)[:TARGET]
path = os.path.join(acts_dir, "layer6_10k.npy")
np.save(path, all_acts)
print(all_acts.shape, all_acts.dtype, os.path.getsize(path) / 1e6, "MB")