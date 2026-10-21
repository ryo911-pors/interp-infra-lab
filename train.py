import torch

from loader import load_batches
from sae import SAE, loss_fn

DATA = r"C:\data\acts\gpt2_16_shuffled"
device = "cuda"
l1_coeff = 10
epochs = 30

torch.manual_seed(0) #fix the random init so every run starts from the same weights
sae = SAE(768, 12288).to(device) 
opt = torch.optim.Adam(sae.parameters(), lr=3e-4)
#fired = torch.zeros(12288, device=device) # how many times each feature fired in the last epoch(0 : dead)

step = 0

for epoch in range(epochs):
  print(f"epoch {epoch}")
  fired = torch.zeros(12288, device=device) # how many times each feature fired in the last epoch(0 : dead)
  for batch in load_batches(DATA, 4096):
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

  print("never fired features", (fired == 0).sum().item(), "/ 12288")

#print("never fired features", (fired == 0).sum().item(), "/ 12288")

