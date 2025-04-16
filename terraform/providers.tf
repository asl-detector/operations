terraform {
  required_version = "~> 1.11.3"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.42.0" # Latest as of April 2025
    }
  }

  backend "s3" {
    bucket       = "terraform-state-asl-foundation"
    key          = "operations/terraform.tfstate"
    region       = "us-west-2"
    encrypt      = true
    use_lockfile = true
  }
}

provider "aws" {
  region = "us-west-2"

  default_tags {
    tags = {
      terraform_managed = true
      account           = "operations"
    }
  }
}

provider "awscc" {
  region = var.aws_region
}

data "aws_caller_identity" "current" {}

output "account_id" {
  value = data.aws_caller_identity.current.account_id
}
