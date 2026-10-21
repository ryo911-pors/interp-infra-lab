
$env:ACTS_DIR = "s3://ryo-interp-acts-321604617840"
$env:IN_NAME  = "gpt2_16_eks"
$env:OUT_NAME = "gpt2_16_eks_shuffled"
uv run python shuffle.py
