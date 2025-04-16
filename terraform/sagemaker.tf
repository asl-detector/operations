# SageMaker Model
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

# SageMaker Endpoint Configuration
resource "aws_sagemaker_endpoint_configuration" "asl_endpoint_config" {
  name = "${var.project_name}-${var.environment}-endpoint-config"

  production_variants {
    variant_name           = "AllTraffic"
    model_name             = aws_sagemaker_model.asl_model.name
    initial_instance_count = 1
    instance_type          = "ml.m5.large"
  }

  data_capture_config {
    enable_capture              = true
    initial_sampling_percentage = 100
    destination_s3_uri          = "s3://${aws_s3_bucket.monitoring_data.bucket}/capture"

    capture_options {
      capture_mode = "Input"
    }

    capture_options {
      capture_mode = "Output"
    }

    capture_content_type_header {
      csv_content_types = ["text/csv"]
    }
  }
}

# SageMaker Endpoint
resource "aws_sagemaker_endpoint" "asl_endpoint" {
  name                 = "${var.project_name}-${var.environment}-endpoint"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.asl_endpoint_config.name
}

# Output endpoint name
output "sagemaker_endpoint_name" {
  value = aws_sagemaker_endpoint.asl_endpoint.name
}
