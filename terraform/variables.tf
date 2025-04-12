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
