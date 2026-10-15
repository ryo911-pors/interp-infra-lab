

terraform {
  backend "s3" {
    bucket       = "ryo-tfstate-321604617840"
    key          = "interp-infra-lab/eks-access.tfstate"
    region       = "us-east-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_iam_role" "extract" {
  name = "interp-extract"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = {Service = "pods.eks.amazonaws.com"}
      Action    = ["sts:AssumeRole", "sts:TagSession"]
    }]
  })
}

resource "aws_iam_role_policy" "extract" {
  name = "s3-acts"
  role = aws_iam_role.extract.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Action  = ["s3:PutObject", "s3:GetObject"]
      Resource = "arn:aws:s3:::ryo-interp-acts-321604617840/*"
    }]
  })
}

resource "aws_eks_pod_identity_association" "extract" {
  cluster_name    = "vllm-lab"
  namespace       = "default"
  service_account = "extract"
  role_arn        = aws_iam_role.extract.arn
}


