# Edge Model Data Quality Monitoring Job
resource "aws_sagemaker_data_quality_job_definition" "data_quality_job" {
  name     = "${var.project_name}-${var.environment}-data-quality-job"
  role_arn = aws_iam_role.sagemaker_role.arn

  data_quality_app_specification {
    image_uri = "763104351884.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-model-monitor-analyzer:latest"
  }

  data_quality_job_input {
    batch_transform_input {
      data_captured_destination_s3_uri = "s3://${aws_s3_bucket.monitoring_data.bucket}/batch-transform-capture"
      dataset_format {
        csv {}
      }
      local_path = "/opt/ml/processing/input/batch"
    }
  }

  data_quality_job_output_config {
    monitoring_outputs {
      s3_output {
        s3_uri     = "s3://${aws_s3_bucket.monitoring_data.bucket}/data-monitoring"
        local_path = "/opt/ml/processing/output"
      }
    }
  }

  data_quality_baseline_config {
    constraints_resource {
      s3_uri = "s3://${aws_s3_bucket.baseline_dataset.bucket}/constraints/data_constraints.json"
    }
    statistics_resource {
      s3_uri = "s3://${aws_s3_bucket.baseline_dataset.bucket}/statistics/data_statistics.json"
    }
  }

  job_resources {
    cluster_config {
      instance_count    = 1
      instance_type     = "ml.t3.medium" # Smaller instance type
      volume_size_in_gb = 20
    }
  }

  stopping_condition {
    max_runtime_in_seconds = 3600
  }
}

# Data Quality Monitoring Schedule (runs weekly)
resource "aws_sagemaker_monitoring_schedule" "data_monitoring_schedule" {
  name = "${var.project_name}-${var.environment}-data-monitoring-schedule"

  monitoring_schedule_config {
    monitoring_type = "DataQuality"

    monitoring_job_definition_name = aws_sagemaker_data_quality_job_definition.data_quality_job.name

    schedule_config {
      schedule_expression = "cron(0 12 ? * MON *)" # Weekly on Monday at noon
    }
  }
}

# CloudWatch metric for model evaluation results
resource "aws_cloudwatch_metric_alarm" "model_accuracy_alarm" {
  alarm_name          = "${var.project_name}-${var.environment}-accuracy-alarm"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ModelAccuracy"
  namespace           = "Custom/ModelMetrics"
  period              = 604800 # 7 days (in seconds)
  statistic           = "Average"
  threshold           = 0.8 # 80% accuracy threshold
  alarm_description   = "Alarm when model accuracy drops below 80%"

  alarm_actions = [aws_sns_topic.model_alerts.arn]
}

# SNS Topic for model alerts
resource "aws_sns_topic" "model_alerts" {
  name = "${var.project_name}-${var.environment}-model-alerts"
}

# Lambda function to publish model metrics to CloudWatch - use inline code instead of S3
resource "aws_lambda_function" "publish_model_metrics" {
  function_name = "${var.project_name}-${var.environment}-publish-metrics"
  role          = aws_iam_role.lambda_exec.arn
  handler       = "index.handler"
  runtime       = "python3.11"
  timeout       = 60

  filename         = "${path.module}/lambda/publish_metrics.zip"
  source_code_hash = filebase64sha256("${path.module}/lambda/publish_metrics.zip")

  environment {
    variables = {
      MODEL_BUCKET = aws_s3_bucket.monitoring_data.bucket
    }
  }

  # Add this lifecycle block to prevent issues if the zip file doesn't exist yet
  lifecycle {
    ignore_changes = [
      filename,
      source_code_hash,
    ]
  }
}

# IAM role for Lambda function
resource "aws_iam_role" "lambda_exec" {
  name = "${var.project_name}-${var.environment}-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
        Action = "sts:AssumeRole"
      }
    ]
  })
}

# Policy for Lambda to read S3 and publish CloudWatch metrics
resource "aws_iam_role_policy" "lambda_s3_cloudwatch" {
  name = "S3CloudWatchAccess"
  role = aws_iam_role.lambda_exec.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
      }
    ]
  })
}

# EventBridge rule to trigger Lambda when evaluation results are created
resource "aws_cloudwatch_event_rule" "evaluation_completed" {
  name        = "${var.project_name}-${var.environment}-evaluation-completed"
  description = "Detect when model evaluation is completed"

  event_pattern = jsonencode({
    "source" : ["aws.s3"],
    "detail-type" : ["Object Created"],
    "detail" : {
      "bucket" : {
        "name" : [aws_s3_bucket.monitoring_data.bucket]
      },
      "object" : {
        "key" : [{
          "prefix" : "evaluation/"
        }]
      }
    }
  })
}

resource "aws_cloudwatch_event_target" "invoke_metrics_lambda" {
  rule      = aws_cloudwatch_event_rule.evaluation_completed.name
  target_id = "InvokeMetricsLambda"
  arn       = aws_lambda_function.publish_model_metrics.arn
}

# Lambda permission to be invoked by EventBridge
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.publish_model_metrics.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.evaluation_completed.arn
}
