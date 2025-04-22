# Variables
variable "aws_region" {
  description = "AWS region for all resources"
  default     = "us-west-2" # Change to your preferred region
}

variable "project_name" {
  description = "Name of the project"
  default     = "asl-detection"
}

variable "environment" {
  description = "Environment (dev/prod)"
  default     = "dev"
}

variable "custom_xgboost_image" {
  description = "ECR URI for custom XGBoost container"
  default     = "223537960975.dkr.ecr.us-west-2.amazonaws.com/custom-xgboost-universal:1.7.4-universal"
}
