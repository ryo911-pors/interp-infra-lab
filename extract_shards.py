import os
import time
import numpy as np
import torch
import json
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer



CTX = 1024
DOCS_PER_SHARD = 100
N_SHARDS = 10
MAX_TOKENS_PER_DOC = 1024 
LAYER = 6

CONFIG = {                  
    "model": "gpt2-small",
    "loader": "transformers.AutoModel (no processing)",   
    "hook": f"h[{LAYER}] output",
    "dtype": "float32",
    "dataset": "NeelNanda/pile-10k",
    "ctx": CTX,
    "max_tokens_per_doc": MAX_TOKENS_PER_DOC,
    "docs_per_shard": DOCS_PER_SHARD,
}

out_dir = os.path.join(os.environ["ACTS_DIR"], "gpt2_16_hf")
os.makedirs(out_dir, exist_ok=True)
manifest_path = os.path.join(out_dir, "manifest.json")

def load_manifest():
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        if manifest["config"] != CONFIG:                       
            raise SystemExit(f"取り方が台帳と違います。別のフォルダを使ってください\n"
                             f"台帳: {manifest['config']}\n今回: {CONFIG}")
        return manifest
    return {"config": CONFIG, "shards": {}}

def save_manifest(manifest):
    tmp = manifest_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, manifest_path)

tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModel.from_pretrained("openai-community/gpt2").cuda().eval()
model.h = model.h[:LAYER + 1]

ds = load_dataset("NeelNanda/pile-10k", split="train")

saved = {}
def hook(module, inputs, output):                              
    saved["acts"] = output
model.h[LAYER].register_forward_hook(hook)

ds = load_dataset("NeelNanda/pile-10k", split="train")

def get_acts(chunk):                                     
    with torch.no_grad():
        model(chunk, use_cache=False)
    return saved["acts"][0].cpu().numpy()
    
manifest = load_manifest()

t0 = time.perf_counter()
n = 0
for shard_idx in range(N_SHARDS):
    name = f"shard_{shard_idx:05d}.npy"
    path = os.path.join(out_dir, name)
    if name in manifest["shards"]:
        print(f"skip {name}")
        continue

    first = shard_idx * DOCS_PER_SHARD
    buffer = []
    for doc in ds.select(range(first, first + DOCS_PER_SHARD)):
        ids = [tok.bos_token_id] + tok(doc["text"], truncation=True, max_length=MAX_TOKENS_PER_DOC - 1).input_ids
        tokens = torch.tensor(ids).cuda() 
        for start in range(0, min(len(tokens), MAX_TOKENS_PER_DOC), CTX):
            chunk = tokens[start:start + CTX].unsqueeze(0)
            buffer.append(get_acts(chunk))

    acts = np.concatenate(buffer)
    
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, acts)
    os.replace(tmp, path)

    
    manifest["shards"][name] = {                 # 2. そのあとで台帳に書き足す
            "docs": [first, first + DOCS_PER_SHARD - 1],
            "tokens": len(acts),
    }
    save_manifest(manifest)
    
    n += len(acts)
    print(f"saved {path} {acts.shape}")

elapsed = time.perf_counter() - t0
print(f"{n} tokens in {elapsed:.2f}s = {n / elapsed:.0f} tokens/s")
