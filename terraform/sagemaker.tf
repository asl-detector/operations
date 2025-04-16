# SageMaker Model Package Group for edge model versioning
resource "aws_sagemaker_model_package_group" "asl_model_group" {
  model_package_group_name = "${var.project_name}-${var.environment}-models"

  tags = {
    Name        = "ASL Detection Models"
    Environment = var.environment
    Project     = var.project_name
  }
}

# Keep the SageMaker Model for training and registry purposes
resource "aws_sagemaker_model" "asl_model" {
  name               = "${var.project_name}-${var.environment}-model"
  execution_role_arn = aws_iam_role.sagemaker_role.arn

  primary_container {
    image          = "433757028032.dkr.ecr.${var.aws_region}.amazonaws.com/xgboost:1"
    model_data_url = "s3://${aws_s3_bucket.baseline_dataset.bucket}/models/initial-model.tar.gz"

    environment = {
      SAGEMAKER_CONTAINER_LOG_LEVEL = "20"
      SAGEMAKER_PROGRAM             = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY    = "/opt/ml/model/code"
      SAGEMAKER_REGION              = var.aws_region
    }
  }

  depends_on = [aws_s3_object.initial_model]
}

# Output model package group name
output "model_package_group_name" {
  value = aws_sagemaker_model_package_group.asl_model_group.model_package_group_name
}
