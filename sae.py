import torch
import torch.nn as nn
from loader import load_batches

class SAE(nn.Module):
  def __init__(self, d_in, d_sae):
    super().__init__()
    self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
    self.b_enc = nn.Parameter(torch.zeros(d_sae))
    self.W_dec = nn.Parameter(self.W_enc.data.T.clone())
    self.W_dec.data /= self.W_dec.data.norm(dim=1, keepdim=True)
    self.b_dec = nn.Parameter(torch.zeros(d_in))

  def forward(self, x):
    f = torch.relu(x @ self.W_enc + self.b_enc)
    x_hat = f @ self.W_dec + self.b_dec
    return x_hat, f

def loss_fn(x, x_hat, f, l1_coeff):
  recon = ((x_hat - x) ** 2).sum(dim=1).mean()
  sparsity = f.sum(dim=1).mean()
  return recon + l1_coeff * sparsity, recon, sparsity

if __name__ == "__main__":

  sae = SAE(768, 12288)
  batch = next(load_batches(r"C:\data\acts\gpt2_16_shuffled", 4096))
  x = torch.from_numpy(batch)

  x_hat, f = sae(x)
  loss, recon, sparsity = loss_fn(x, x_hat, f, l1_coeff=5)
  l0 =  (f>0).float().sum(dim=1).mean()

  norms = x.norm(dim=1)
  print("x の大きさ: 平均", norms.mean().item(), "最大", norms.max().item())
  print("recon:", recon.item(), "sparsity:", sparsity.item(), "L0:", l0.item())


