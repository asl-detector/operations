# SageMaker Pipeline

resource "aws_cloudformation_stack" "sagemaker_pipeline" {
  name = "${var.project_name}-${var.environment}-pipeline-stack"

  template_body = <<EOF
{
  "Resources": {
    "ModelRetrainingPipeline": {
      "Type": "AWS::SageMaker::Pipeline",
      "Properties": {
        "PipelineName": "${var.project_name}-${var.environment}-retraining-pipeline",
        "PipelineDefinition": {
          "PipelineDefinitionBody": {
            "Version": "2020-12-01",
            "Steps": [
              {
                "Name": "FeatureProcessing",
                "Type": "Processing",
                "Arguments": {
                  "ProcessingResources": {
                    "ClusterConfig": {
                      "InstanceCount": 1,
                      "InstanceType": "ml.m5.xlarge",
                      "VolumeSizeInGB": 30
                    }
                  },
                  "AppSpecification": {
                    "ImageUri": "683313688378.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:1.0-1",
                    "ContainerEntrypoint": ["python", "/opt/ml/processing/code/process.py"]
                  },
                  "ProcessingInputs": [
                    {
                      "InputName": "pose-data",
                      "S3Input": {
                        "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/pose-data",
                        "LocalPath": "/opt/ml/processing/input/pose",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    },
                    {
                      "InputName": "code",
                      "S3Input": {
                        "S3Uri": "s3://${aws_s3_bucket.baseline_dataset.bucket}/code",
                        "LocalPath": "/opt/ml/processing/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    }
                  ],
                  "ProcessingOutputs": [
                    {
                      "OutputName": "features",
                      "S3Output": {
                        "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/features",
                        "LocalPath": "/opt/ml/processing/output/features",
                        "S3UploadMode": "EndOfJob"
                      }
                    }
                  ],
                  "RoleArn": "${aws_iam_role.sagemaker_role.arn}"
                }
              },
              {
                "Name": "TrainModel",
                "Type": "Training",
                "DependsOn": ["FeatureProcessing"],
                "Arguments": {
                  "AlgorithmSpecification": {
                    "TrainingImage": "683313688378.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-xgboost:1.5-1",
                    "TrainingInputMode": "File"
                  },
                  "InputDataConfig": [
                    {
                      "ChannelName": "train",
                      "DataSource": {
                        "S3DataSource": {
                          "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/features/train",
                          "S3DataType": "S3Prefix"
                        }
                      }
                    },
                    {
                      "ChannelName": "validation",
                      "DataSource": {
                        "S3DataSource": {
                          "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/features/validation",
                          "S3DataType": "S3Prefix"
                        }
                      }
                    }
                  ],
                  "OutputDataConfig": {
                    "S3OutputPath": "s3://${aws_s3_bucket.monitoring_data.bucket}/models"
                  },
                  "ResourceConfig": {
                    "InstanceCount": 1,
                    "InstanceType": "ml.m5.xlarge",
                    "VolumeSizeInGB": 30
                  },
                  "StoppingCondition": {
                    "MaxRuntimeInSeconds": 3600
                  },
                  "RoleArn": "${aws_iam_role.sagemaker_role.arn}"
                }
              },
              {
                "Name": "ModelEvaluation",
                "Type": "Processing",
                "DependsOn": ["TrainModel"],
                "Arguments": {
                  "ProcessingResources": {
                    "ClusterConfig": {
                      "InstanceCount": 1,
                      "InstanceType": "ml.m5.large",
                      "VolumeSizeInGB": 20
                    }
                  },
                  "AppSpecification": {
                    "ImageUri": "683313688378.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:1.0-1",
                    "ContainerEntrypoint": ["python", "/opt/ml/processing/code/evaluate.py"]
                  },
                  "ProcessingInputs": [
                    {
                      "InputName": "test-data",
                      "S3Input": {
                        "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/features/test",
                        "LocalPath": "/opt/ml/processing/input/test",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    },
                    {
                      "InputName": "model",
                      "S3Input": {
                        "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/models",
                        "LocalPath": "/opt/ml/processing/input/model",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    },
                    {
                      "InputName": "code",
                      "S3Input": {
                        "S3Uri": "s3://${aws_s3_bucket.baseline_dataset.bucket}/code",
                        "LocalPath": "/opt/ml/processing/code",
                        "S3DataType": "S3Prefix",
                        "S3InputMode": "File"
                      }
                    }
                  ],
                  "ProcessingOutputs": [
                    {
                      "OutputName": "evaluation",
                      "S3Output": {
                        "S3Uri": "s3://${aws_s3_bucket.monitoring_data.bucket}/evaluation",
                        "LocalPath": "/opt/ml/processing/output/evaluation",
                        "S3UploadMode": "EndOfJob"
                      }
                    }
                  ],
                  "RoleArn": "${aws_iam_role.sagemaker_role.arn}"
                }
              },
              {
                "Name": "UpdateEndpoint",
                "Type": "RegisterModel",
                "DependsOn": ["ModelEvaluation"],
                "Arguments": {
                  "Model": {
                    "PrimaryContainer": {
                      "Image": "683313688378.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-xgboost:1.5-1",
                      "ModelDataUrl.$": "$.Steps.TrainModel.ModelArtifacts.S3ModelArtifacts",
                      "Environment": {
                        "SAGEMAKER_PROGRAM": "inference.py",
                        "SAGEMAKER_SUBMIT_DIRECTORY": "/opt/ml/model/code"
                      }
                    },
                    "ModelName.$": "Join('-', ['${var.project_name}', '${var.environment}', $.Steps.TrainModel.TrainingJobName])"
                  },
                  "EndpointName": "${var.project_name}-${var.environment}-endpoint",
                  "ContentTypes": ["text/csv"],
                  "ResponseTypes": ["text/csv"]
                }
              }
            ]
          }
        },
        "RoleArn": "${aws_iam_role.sagemaker_role.arn}"
      }
    }
  },
  "Outputs": {
    "PipelineName": {
      "Value": "${var.project_name}-${var.environment}-retraining-pipeline"
    }
  }
}
EOF

  capabilities = ["CAPABILITY_IAM"]
  depends_on = [
    aws_s3_object.process_script,
    aws_s3_object.evaluate_script,
    aws_s3_object.train_script
  ]
}

# EventBridge rule to trigger retraining based on monitoring alerts
resource "aws_cloudwatch_event_rule" "data_drift_detected" {
  name        = "${var.project_name}-${var.environment}-data-drift-rule"
  description = "Trigger model retraining when data drift is detected"

  event_pattern = jsonencode({
    "source" : ["aws.sagemaker"],
    "detail-type" : ["SageMaker Model Monitor Alert"],
    "detail" : {
      "MonitoringScheduleName" : [
        aws_sagemaker_monitoring_schedule.data_monitoring_schedule.name
      ],
      "Status" : ["Alert"]
    }
  })
}

resource "aws_iam_role_policy" "eventbridge_sagemaker_access" {
  name = "SageMakerAccess"
  role = aws_iam_role.eventbridge_role.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sagemaker:StartPipelineExecution"
        ]
        Resource = [
          "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.project_name}-${var.environment}-retraining-pipeline"
        ]
      }
    ]
  })
}

# Add an additional rule for CloudWatch alarms
resource "aws_cloudwatch_event_rule" "model_accuracy_alert" {
  name        = "${var.project_name}-${var.environment}-accuracy-alert-rule"
  description = "Trigger model retraining when model accuracy drops"

  event_pattern = jsonencode({
    "source" : ["aws.cloudwatch"],
    "detail-type" : ["CloudWatch Alarm State Change"],
    "resources" : [aws_cloudwatch_metric_alarm.model_accuracy_alarm.arn],
    "detail" : {
      "state" : {
        "value" : ["ALARM"]
      }
    }
  })
}

# Add an EventBridge target for the accuracy alarm
resource "aws_cloudwatch_event_target" "trigger_pipeline_accuracy" {
  rule      = aws_cloudwatch_event_rule.model_accuracy_alert.name
  target_id = "TriggerPipelineAccuracy"
  arn       = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.project_name}-${var.environment}-retraining-pipeline"
  role_arn  = aws_iam_role.eventbridge_role.arn

  sagemaker_pipeline_target {
    pipeline_parameter_list {
      name  = "TriggerSource"
      value = "AccuracyAlarm"
    }
  }
}


resource "aws_cloudwatch_event_target" "trigger_pipeline" {
  rule      = aws_cloudwatch_event_rule.data_drift_detected.name
  target_id = "TriggerPipeline"
  arn       = "arn:aws:sagemaker:${var.aws_region}:${data.aws_caller_identity.current.account_id}:pipeline/${var.project_name}-${var.environment}-retraining-pipeline"
  role_arn  = aws_iam_role.eventbridge_role.arn

  sagemaker_pipeline_target {
    pipeline_parameter_list {
      name  = "MonitoringScheduleName"
      value = aws_sagemaker_monitoring_schedule.data_schedule.monitoring_schedule_name
    }
  }
}

# Output pipeline ARN
output "retraining_pipeline_name" {
  value = "${var.project_name}-${var.environment}-retraining-pipeline"
}
