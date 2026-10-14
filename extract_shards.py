import os
import time
import numpy as np
import torch
import json
import uuid
import io
import boto3
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer


MODEL = os.environ.get("MODEL", "openai-community/gpt2")
LAYER = int(os.environ.get("LAYER", "6"))
DTYPE = os.environ.get("DTYPE", "float32")
OUT_NAME = os.environ.get("OUT_NAME", "gpt2_16")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


CTX = 1024
DOCS_PER_SHARD = 100
N_SHARDS = 10
MAX_TOKENS_PER_DOC = 1024 

s3 = boto3.client("s3")


CONFIG = {                  
    "model": MODEL,
    "loader": "transformers.AutoModel (no processing)",   
    "hook": f"h[{LAYER}] output",
    "dtype": DTYPE,
    "stored_as" : "uint16 (bf16 bits)" if DTYPE == "bfloat16" else DTYPE,
    "device": DEVICE,
    "dataset": "NeelNanda/pile-10k",
    "ctx": CTX,
    "max_tokens_per_doc": MAX_TOKENS_PER_DOC,
    "docs_per_shard": DOCS_PER_SHARD,
    "index": "shard_XXXXX_index.npy = int32 [doc, pos] per row"
}

def join(a, b):                        # S3 は区切りが必ず "/"（Windows の os.path.join は "\" にしてしまう）
    if a.startswith("s3://"):
        return a.rstrip("/") + "/" + b
    return os.path.join(a, b)

out_dir = join(os.environ["ACTS_DIR"], OUT_NAME)
if not out_dir.startswith("s3://"):    # S3 にはフォルダが無いので、作るのは手元のときだけ
    os.makedirs(out_dir, exist_ok=True)
manifest_path = join(out_dir, "manifest.json")

def load_manifest():
    if exists(manifest_path):
        manifest = json.loads(read_bytes(manifest_path))
        if manifest["config"] != CONFIG:                       
            raise SystemExit(f"取り方が台帳と違います。別のフォルダを使ってください\n"
                             f"台帳: {manifest['config']}\n今回: {CONFIG}")
        return manifest
    return {"config": CONFIG, "shards": {}}

def split_s3(path):
    bucket, key = path[len("s3://"):].split("/", 1)
    return bucket, key

def write_bytes(path, data):
    if path.startswith("s3://"):
        bucket, key = split_s3(path)
        s3.put_object(Bucket=bucket, Key=key, Body=data)
    else:
        tmp = f"{path}.{uuid.uuid4().hex[:8]}.tmp"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, path)

def read_bytes(path):
    if path.startswith("s3://"):
        bucket, key = split_s3(path)
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    with open(path, "rb") as f:
        return f.read()


def exists(path):
    if path.startswith("s3://"):
        bucket, key = split_s3(path)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return True
        except s3.exceptions.ClientError:
            return False

    return os.path.exists(path)


def save_manifest(manifest):
    write_bytes(manifest_path, json.dumps(manifest, indent=2).encode())

def save_npy(path, arr):
    buf = io.BytesIO()
    np.save(buf, arr)
    write_bytes(path, buf.getvalue())

    
manifest = load_manifest()
save_manifest(manifest)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModel.from_pretrained(MODEL, dtype=getattr(torch, DTYPE)).to(DEVICE).eval()

if hasattr(model, "h"):
    model.h = model.h[:LAYER + 1]
    layer = model.h
else:
    model.layers = model.layers[:LAYER+1]
    layer = model.layers


saved = {}
def hook(module, inputs, output):                              
    saved["acts"] = output
layer[LAYER].register_forward_hook(hook)

ds = load_dataset("NeelNanda/pile-10k", split="train")

def get_acts(chunk):                                     
    with torch.no_grad():
        model(chunk, use_cache=False)
    acts = saved["acts"][0].cpu()
    if DTYPE == "bfloat16":
        return acts.view(torch.uint16).numpy()
    return acts.numpy()
    


t0 = time.perf_counter()
n = 0
if "JOB_COMPLETION_INDEX" in os.environ:
    shard_ids = [int(os.environ["JOB_COMPLETION_INDEX"])]
else:
    shard_ids = range(N_SHARDS)

for shard_idx in shard_ids:
    
    name = f"shard_{shard_idx:05d}.npy"
    path = join(out_dir, name)
    if exists(path.replace(".npy", ".done.json")):
        print(f"skip {name}")
        continue

    first = shard_idx * DOCS_PER_SHARD
    buffer = []
    index = []
    for doc_id, doc in enumerate(ds.select(range(first, first + DOCS_PER_SHARD)), start=first):
        ids = tok(doc["text"], truncation=True, max_length=MAX_TOKENS_PER_DOC).input_ids
        if ids[0] != tok.bos_token_id:
            ids = [tok.bos_token_id] + ids[:MAX_TOKENS_PER_DOC - 1]
        tokens = torch.tensor(ids).to(DEVICE)
        for start in range(0, len(tokens), CTX):
            chunk = tokens[start:start + CTX].unsqueeze(0)
            acts = get_acts(chunk)
            buffer.append(acts)
            pos = np.arange(start, start + len(acts))
            index.append(np.stack([np.full(len(acts), doc_id), pos], axis=1))

    acts = np.concatenate(buffer)
    idx = np.concatenate(index).astype(np.int32)
    print(f"uploading {name} ...", flush=True)
    save_npy(path, acts)
    save_npy(path.replace(".npy", "_index.npy"), idx)

    #with open(path.replace(".npy", ".done.json"), "w") as f:
        #json.dump({"docs" : [first, first + DOCS_PER_SHARD - 1], "tokens": len(acts)}, f)

    memo = {"docs": [first, first + DOCS_PER_SHARD - 1], "tokens": len(acts)}
    write_bytes(path.replace(".npy", ".done.json"), json.dumps(memo).encode())
    
    n += len(acts)
    print(f"saved {path} {acts.shape}")

elapsed = time.perf_counter() - t0
print(f"{n} tokens in {elapsed:.2f}s = {n / elapsed:.0f} tokens/s")
