# Data Quality Job Definition
resource "aws_sagemaker_data_quality_job_definition" "data_quality_job" {
  name     = "${var.project_name}-${var.environment}-data-quality-job"
  role_arn = aws_iam_role.sagemaker_role.arn

  data_quality_app_specification {
    image_uri = "763104351884.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-model-monitor-analyzer:latest"
  }

  data_quality_job_input {
    endpoint_input {
      endpoint_name = aws_sagemaker_endpoint.asl_endpoint.name
      local_path    = "/opt/ml/processing/input/endpoint"
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
      instance_type     = "ml.m5.xlarge"
      volume_size_in_gb = 20
    }
  }

  stopping_condition {
    max_runtime_in_seconds = 3600
  }
}

# Data Quality Monitoring Schedule
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

# Model quality will be implemented via CloudWatch alarms instead
# Since aws_sagemaker_model_quality_job_definition is not supported
resource "aws_cloudwatch_metric_alarm" "model_accuracy_alarm" {
  alarm_name          = "${var.project_name}-${var.environment}-accuracy-alarm"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ModelAccuracy"
  namespace           = "AWS/SageMaker"
  period              = 604800 # 7 days (in seconds) instead of 86400 (1 day)
  statistic           = "Average"
  threshold           = 0.8 # 80% accuracy threshold
  alarm_description   = "Alarm when model accuracy drops below 80%"

  dimensions = {
    EndpointName = aws_sagemaker_endpoint.asl_endpoint.name
  }

  alarm_actions = [aws_sns_topic.model_alerts.arn]
}

# SNS Topic for model alerts
resource "aws_sns_topic" "model_alerts" {
  name = "${var.project_name}-${var.environment}-model-alerts"
}
