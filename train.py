import torch
import os
import io
import json
from loader import load_batches
from storage import join, write_bytes
from sae import SAE, loss_fn


IN_NAME = os.environ["IN_NAME"]
in_dir = join(os.environ["ACTS_DIR"], IN_NAME)
OUT_NAME = os.environ["OUT_NAME"]
out_dir = join(os.environ["ACTS_DIR"], OUT_NAME)
if not out_dir.startswith("s3://"):
    os.makedirs(out_dir, exist_ok=True)
device = "cuda"
l1_coeff = 10
epochs = 30
lr=3e-4
batch_size = 4096
d_sae = 12288

torch.manual_seed(0) #fix the random init so every run starts from the same weights
first = next(load_batches(in_dir, batch_size))
d_in = first.shape[1]
sae = SAE(d_in, d_sae).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=lr)
#fired = torch.zeros(12288, device=device) # how many times each feature fired in the last epoch(0 : dead)

step = 0

config = {
    "data": in_dir,
    "d_in": d_in,
    "d_sae": d_sae,
    "l1_coeff": l1_coeff,
    "lr": lr,
    "epochs": epochs,
    "batch_size": batch_size,
    "seed": 0,
    "drop_bos": True,
}
write_bytes(join(out_dir, "config.json"), json.dumps(config, indent=2).encode())

metrics = []

for epoch in range(epochs):
  print(f"epoch {epoch}")
  fired = torch.zeros(d_sae, device=device) # how many times each feature fired in the last epoch(0 : dead)
  for batch in load_batches(in_dir, batch_size):
    x = torch.from_numpy(batch).to(device)
    x_hat, f = sae(x)
    loss, recon, sparsity = loss_fn(x, x_hat, f, l1_coeff)

    opt.zero_grad()
    loss.backward()
    opt.step()
    with torch.no_grad():
      sae.W_dec.data /= sae.W_dec.data.norm(dim=1, keepdim=True)

    fired += (f > 0).float().sum(dim=0)

    if step % 200 == 0:
      l0 = (f>0).float().sum(dim=1).mean()
      print(f"step {step} recon {recon.item():.1f} L0 {l0.item():.1f}")
    step += 1

  print(f"never fired features {(fired == 0).sum().item()} / {d_sae}")

  l0 = (f > 0).float().sum(dim=1).mean().item()
  metrics.append({
      "epoch": epoch,
      "dead": (fired == 0).sum().item(),
      "l0": l0,
      "recon": recon.item(),
  })

write_bytes(join(out_dir, "metrics.json"), json.dumps(metrics, indent=2).encode())

buf = io.BytesIO()
torch.save(sae.state_dict(), buf)
write_bytes(join(out_dir, "sae.pt"), buf.getvalue())

#print("never fired features", (fired == 0).sum().item(), "/ 12288")

