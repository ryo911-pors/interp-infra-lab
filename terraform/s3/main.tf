terraform {
  backend "s3" {
    bucket       = "ryo-tfstate-321604617840"
    key          = "interp-infra-lab/s3.tfstate"
    region       = "us-east-1"
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "acts" {
  bucket = "ryo-interp-acts-321604617840"
}