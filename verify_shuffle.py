import os

import numpy as np

ACTS_DIR = os.environ["ACTS_DIR"]
IN_DIR = os.path.join(ACTS_DIR, "gpt2_16_kind_indexed")
OUT_DIR = os.path.join(ACTS_DIR, "gpt2_16_shuffled")
N = 10

orig_acts = [np.load(os.path.join(IN_DIR, f"shard_{i:05d}.npy")) for i in range(N)]
orig_idx = [np.load(os.path.join(IN_DIR, f"shard_{i:05d}_index.npy")) for i in range(N)]
shuf_acts = [np.load(os.path.join(OUT_DIR, f"shuffled_{i:05d}.npy")) for i in range(N)]
shuf_idx = [np.load(os.path.join(OUT_DIR, f"shuffled_{i:05d}_index.npy")) for i in range(N)]


total_orig = 0
for a in orig_acts:
  total_orig += len(a)

total_shuf = 0
for b in shuf_acts:
  total_shuf += len(b)

print(f"before shuffle {total_orig}")
print(f"after shuffle {total_shuf}")
print(f"same? {total_orig == total_shuf}")

print(f"unique docs in first 4096 rows (before) {len(np.unique(orig_idx[0][:4096, 0]))}")
print(f"unique docs in first 4096 rows (after) {len(np.unique(shuf_idx[0][:4096, 0]))}")

