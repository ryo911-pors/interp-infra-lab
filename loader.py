import numpy as np


from storage import join, exists, load_npy

def load_batches(dir_path, batch_size, drop_bos=True):
  leftover = None
  box = 0
  while True:
    path = join(dir_path, f"shuffled_{box:05d}.npy")
    if not exists(path):
      break
    acts = load_npy(path)
    idx = load_npy(path.replace(".npy", "_index.npy"))
    if drop_bos:
      keep = idx[:, 1] > 0 
      acts = acts[keep]
      
    print(f"read box {box}: {acts.shape}")
    if leftover is not None:
      acts = np.concatenate([leftover, acts])
    while len(acts) >= batch_size:
      yield acts[:batch_size]
      acts = acts[batch_size:]
    leftover = acts
    box += 1
      

if __name__ == "__main__":
  n_batches = 0
  n_rows = 0

  for batch in load_batches(r"C:\data\acts\gpt2_16_shuffled", 4096):
    n_batches += 1
    n_rows += len(batch)
  print(f"{n_batches} batches, {n_rows} rows")