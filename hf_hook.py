import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

text = "<|endoftext|>My name is Ryo"

#normal GPT2-small
tok = AutoTokenizer.from_pretrained("openai-community/gpt2")
hf = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").cuda().eval()

saved = {}
def hook(module, inputs, output):
    saved["acts"] = output

hf.transformer.h[6].register_forward_hook(hook)
ids = tok(text, return_tensors="pt").input_ids.cuda()
with torch.no_grad():
    hf(ids)
hf_acts = saved["acts"]
print("HF:", hf_acts.shape)

tl = HookedTransformer.from_pretrained_no_processing("gpt2-small", device="cuda")
_, cache = tl.run_with_cache(ids, names_filter="blocks.6.hook_resid_post")
tl_acts = cache["blocks.6.hook_resid_post"]
print("TL:", tl_acts.shape)


print("biggest difference:", (hf_acts - tl_acts).abs().max().item())

