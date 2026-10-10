import os
import time
import numpy as np
import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer

HOOK = "blocks.6.hook_resid_post"
CTX = 1024
TARGET = 10_00000

acts_dir = os.environ["ACTS_DIR"]
os.makedirs(acts_dir, exist_ok=True)

model = HookedTransformer.from_pretrained("gpt2-small", device="cuda")
ds = load_dataset("NeelNanda/pile-10k", split="train")

collected = []
t0 = time.perf_counter()
t_tok = t_model = t_copy = 0.0   # 作業ごとの合計時間
n = 0
for doc in ds:
    a = time.perf_counter()
    tokens = model.to_tokens(doc["text"], truncate=False)[0]   # 1. トークン化
    t_tok += time.perf_counter() - a

    for start in range(0, len(tokens), CTX):
        chunk = tokens[start:start + CTX].unsqueeze(0)

        a = time.perf_counter()
        with torch.no_grad():
            _, cache = model.run_with_cache(chunk, names_filter=HOOK, stop_at_layer=7)   # 2. モデル
        acts = cache[HOOK][0]
        torch.cuda.synchronize()   # GPU の計算が終わるまで待つ
        t_model += time.perf_counter() - a

        a = time.perf_counter()
        collected.append(acts.cpu().numpy())   # 3. GPU → CPU
        t_copy += time.perf_counter() - a

        n += acts.shape[0]
        if n >= TARGET:
            break
    if n >= TARGET:
        break

a = time.perf_counter()
all_acts = np.concatenate(collected)[:TARGET]
path = os.path.join(acts_dir, "layer6_10k.npy")
np.save(path, all_acts)   # 4. 保存
t_save = time.perf_counter() - a

elapsed = time.perf_counter() - t0
print(f"{n} tokens in {elapsed:.2f}s = {n / elapsed:.0f} tokens/s")
print(f"tokenize {t_tok:.2f}s / model {t_model:.2f}s / copy {t_copy:.2f}s / save {t_save:.2f}s")