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
