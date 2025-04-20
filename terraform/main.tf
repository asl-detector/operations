# Remote state access for organization structure
data "terraform_remote_state" "org_structure" {
  backend = "s3"
  config = {
    bucket = "terraform-state-asl-foundation"
    key    = "aws_organization_structure/terraform.tfstate"
    region = "us-west-2"
  }
}

locals {
  account_ids = data.terraform_remote_state.org_structure.outputs.account_ids
}

# Remote state access for data_org resources
data "terraform_remote_state" "data_org" {
  backend = "s3"
  config = {
    bucket = "terraform-state-asl-foundation"
    key    = "data_org/terraform.tfstate"
    region = "us-west-2"
  }
}

# Create IAM role to assume data_org role for cross-account access to clean data lake
resource "aws_iam_role" "data_lake_clean_access_role" {
  name = "${var.project_name}-${var.environment}-data-lake-access"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# Policy to allow SageMaker to assume the role in data_org for clean data lake
resource "aws_iam_role_policy" "assume_data_org_role_policy" {
  name = "assume-data-org-role-policy"
  role = aws_iam_role.data_lake_clean_access_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Resource = data.terraform_remote_state.data_org.outputs.data_lake_clean_access_role_arn
    }]
  })
}

# Create IAM role for cross-account access to external training data
resource "aws_iam_role" "extrn_data_access_role" {
  name = "${var.project_name}-${var.environment}-extrn-data-access"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
      Action = "sts:AssumeRole"
    }]
  })
}

# Policy to allow SageMaker to assume the role in data_org for external training data
resource "aws_iam_role_policy" "assume_extrn_data_org_role_policy" {
  name = "assume-extrn-data-org-role-policy"
  role = aws_iam_role.extrn_data_access_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Resource = data.terraform_remote_state.data_org.outputs.extrn_data_access_role_arn
    }]
  })
}

# S3 buckets
resource "aws_s3_bucket" "baseline_dataset" {
  bucket = "${var.project_name}-${var.environment}-baseline-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "Baseline Dataset"
    Environment = var.environment
    Project     = var.project_name
  }
}

resource "aws_s3_bucket" "monitoring_data" {
  bucket = "${var.project_name}-${var.environment}-monitoring-${random_string.bucket_suffix.result}"

  tags = {
    Name        = "Monitoring Data"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Add bucket policy to allow cross-account access to monitoring_data bucket
resource "aws_s3_bucket_policy" "monitoring_data_cross_account" {
  bucket = aws_s3_bucket.monitoring_data.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${local.account_ids.staging}:root",
            "arn:aws:iam::${local.account_ids.production}:root"
          ]
        }
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.monitoring_data.arn,
          "${aws_s3_bucket.monitoring_data.arn}/*"
        ]
      }
    ]
  })
}

# Create IAM role for cross-account access
resource "aws_iam_role" "monitoring_data_access_role" {
  name = "${var.project_name}-${var.environment}-monitoring-access"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          AWS = [
            "arn:aws:iam::${local.account_ids.staging}:root",
            "arn:aws:iam::${local.account_ids.production}:root"
          ]
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "monitoring_data_access_policy" {
  name = "s3-monitoring-access"
  role = aws_iam_role.monitoring_data_access_role.id
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.monitoring_data.arn,
          "${aws_s3_bucket.monitoring_data.arn}/*"
        ]
      }
    ]
  })
}

# Random suffix for globally unique S3 bucket names
resource "random_string" "bucket_suffix" {
  length  = 8
  special = false
  upper   = false
}

# IAM roles
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-${var.environment}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_role_policy" "sagemaker_ecr_access" {
  name = "ECRFullAccess"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetAuthorizationToken",
          "ecr:DescribeRepositories",
          "ecr:ListImages",
          "ecr:DescribeImages"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonS3FullAccess"
}

resource "aws_iam_role_policy" "sagemaker_s3_access" {
  name = "S3Access"
  role = aws_iam_role.sagemaker_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.baseline_dataset.arn,
          "${aws_s3_bucket.baseline_dataset.arn}/*",
          aws_s3_bucket.monitoring_data.arn,
          "${aws_s3_bucket.monitoring_data.arn}/*"
        ]
      }
    ]
  })
}

# Fix role-to-role attachment by using an inline policy instead
resource "aws_iam_role_policy" "sagemaker_data_lake_access" {
  name = "assume-data-lake-access-role"
  role = aws_iam_role.sagemaker_role.name
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      # Allow SageMaker to directly assume the data_org clean data lake role
      Resource = data.terraform_remote_state.data_org.outputs.data_lake_clean_access_role_arn
    }]
  })
}

# Add policy for external training data access
resource "aws_iam_role_policy" "sagemaker_extrn_data_access" {
  name = "assume-extrn-data-access-role"
  role = aws_iam_role.sagemaker_role.name
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = "sts:AssumeRole"
      Resource = data.terraform_remote_state.data_org.outputs.extrn_data_access_role_arn
    }]
  })
}

# EventBridge role for triggering retraining
resource "aws_iam_role" "eventbridge_role" {
  name = "${var.project_name}-${var.environment}-eventbridge-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "events.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Output important information
output "baseline_bucket" {
  value = aws_s3_bucket.baseline_dataset.bucket
}

output "monitoring_bucket" {
  value = aws_s3_bucket.monitoring_data.bucket
}

output "sagemaker_role_arn" {
  value = aws_iam_role.sagemaker_role.arn
}

output "monitoring_data_access_role_arn" {
  value = aws_iam_role.monitoring_data_access_role.arn
  description = "The ARN of the IAM role for cross-account access to the monitoring data bucket"
}
