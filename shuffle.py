import io
import os

import numpy as np

from storage import join, read_bytes, save_npy, write_bytes, exists

IN_NAME = os.environ.get("IN_NAME", "gpt2_16_kind_indexed")
OUT_NAME = os.environ.get("OUT_NAME", "gpt2_16_shuffled")
N_SHARDS = 10
N_BOXES = 10
SEED = 0

in_dir = join(os.environ["ACTS_DIR"], IN_NAME)
out_dir = join(os.environ["ACTS_DIR"], OUT_NAME)

def load_npy(path):
  return np.load(io.BytesIO(read_bytes(path)))

def box_dir(box):
  d = join(out_dir, f"stage1/box_{box:05d}")
  if not d.startswith("s3://"):
    os.makedirs(d, exist_ok=True)
  return d

def scatter(shard_idx):
  done = join(out_dir, f"stage1/from_{shard_idx:05d}.done.json")
  if exists(done):
    print(f"skip shard {shard_idx}")
    return

  acts = load_npy(join(in_dir, f"shard_{shard_idx:05d}.npy"))
  idx = load_npy(join(in_dir, f"shard_{shard_idx:05d}_index.npy"))

  rng = np.random.default_rng([SEED, shard_idx])
  box_of_row = rng.integers(0, N_BOXES, size=len(acts))

  for box in range(N_BOXES):
      mask = box_of_row == box
      d = box_dir(box)
      save_npy(join(d, f"from_{shard_idx:05d}.npy"), acts[mask])
      save_npy(join(d, f"from_{shard_idx:05d}_index.npy"), idx[mask])

  write_bytes(done, b"{}")
  print(f"scattered shard {shard_idx}: {len(acts)} row -> {N_BOXES} boxes")

if __name__ == "__main__":
    for shard_idx in range(N_SHARDS):
        scatter(shard_idx)

    