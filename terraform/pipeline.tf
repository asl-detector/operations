resource "awscc_sagemaker_pipeline" "model_retraining_pipeline" {
  pipeline_name         = "${var.project_name}-${var.environment}-retraining-pipeline"
  pipeline_display_name = "${var.project_name}-${var.environment}-retraining-pipeline"
  role_arn              = aws_iam_role.sagemaker_role.arn

  pipeline_definition = {
    pipeline_definition_body = jsonencode({
      "Version" : "2020-12-01",
      "Steps" : [
        {
          "Name" : "GenerateBaseline",
          "Type" : "Processing",
          "Arguments" : {
            "ProcessingResources" : {
              "ClusterConfig" : {
                "InstanceCount" : 1,
                "InstanceType" : "ml.m5.large",
                "VolumeSizeInGB" : 20
              }
            },
            "AppSpecification" : {
              "ImageUri" : "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1"
              "ContainerEntrypoint" : ["bash", "-c", "pip install -r /opt/ml/processing/code/requirements.txt && python /opt/ml/processing/code/generate_baseline.py --csv /opt/ml/processing/input/baseline-data/baseline.csv --output-dir /opt/ml/processing/output/baseline-outputs"]
            },
            "ProcessingInputs" : [
              {
                "InputName" : "baseline-data",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.baseline_dataset.bucket}/data",
                  "LocalPath" : "/opt/ml/processing/input/baseline-data",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              },
              {
                "InputName" : "code",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.baseline_dataset.bucket}/code",
                  "LocalPath" : "/opt/ml/processing/code",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              }
            ],
            "ProcessingOutputConfig" : {
              "Outputs" : [
                {
                  "OutputName" : "baseline-outputs",
                  "S3Output" : {
                    "S3Uri" : "s3://${aws_s3_bucket.baseline_dataset.bucket}",
                    "LocalPath" : "/opt/ml/processing/output/baseline-outputs",
                    "S3UploadMode" : "EndOfJob"
                  }
                }
              ]
            },
            "RoleArn" : "${aws_iam_role.sagemaker_role.arn}"
          }
        },
        {
          "Name" : "FeatureProcessing",
          "Type" : "Processing",
          "DependsOn" : ["GenerateBaseline"],
          "Arguments" : {
            "ProcessingResources" : {
              "ClusterConfig" : {
                "InstanceCount" : 1,
                "InstanceType" : "ml.m5.large",
                "VolumeSizeInGB" : 30
              }
            },
            "AppSpecification" : {
              "ImageUri" : "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1"
              "ContainerEntrypoint" : ["bash", "-c", "pip install -r /opt/ml/processing/code/requirements.txt && python /opt/ml/processing/code/process.py"]
            }
            "ProcessingInputs" : [
              {
                "InputName" : "pose-data",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/pose-data",
                  "LocalPath" : "/opt/ml/processing/input/pose",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              },
              {
                "InputName" : "baseline-data",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.baseline_dataset.bucket}/data",
                  "LocalPath" : "/opt/ml/processing/input/baseline",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              },
              {
                "InputName" : "code",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.baseline_dataset.bucket}/code",
                  "LocalPath" : "/opt/ml/processing/code",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              }
            ],
            "ProcessingOutputConfig" : {
              "Outputs" : [
                {
                  "OutputName" : "features",
                  "S3Output" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/features",
                    "LocalPath" : "/opt/ml/processing/output/features",
                    "S3UploadMode" : "EndOfJob"
                  }
                }
              ]
            },
            "RoleArn" : "${aws_iam_role.sagemaker_role.arn}"
          }
        },
        {
          "Name" : "TrainModel",
          "Type" : "Training",
          "DependsOn" : ["FeatureProcessing"],
          "Arguments" : {
            "AlgorithmSpecification" : {
              "TrainingImage" : "246618743249.dkr.ecr.us-west-2.amazonaws.com/sagemaker-xgboost:1.7-1",
              "TrainingInputMode" : "File",
              "EnableSageMakerMetricsTimeSeries" : true
            },
            "HyperParameters" : {
              "num-round" : "300",
              "max-depth" : "6",
              "eta" : "0.05",
              "gamma" : "0.1",
              "min-child-weight" : "2",
              "subsample" : "0.8",
              "colsample-bytree" : "0.8",
              "objective" : "binary:logistic",
              "eval-metric" : "auc",
              "sagemaker_program" : "train.py",
              "sagemaker_submit_directory" : "s3://asl-detection-dev-baseline-7jjirv6s/code/train-source.tar.gz"
            },
            "InputDataConfig" : [
              {
                "ChannelName" : "train",
                "DataSource" : {
                  "S3DataSource" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/features/train",
                    "S3DataType" : "S3Prefix"
                  }
                },
                "ContentType" : "text/csv",
                "CompressionType" : "None"
              },
              {
                "ChannelName" : "validation",
                "DataSource" : {
                  "S3DataSource" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/features/validation",
                    "S3DataType" : "S3Prefix"
                  }
                },
                "ContentType" : "text/csv",
                "CompressionType" : "None"
              }
            ],
            "OutputDataConfig" : {
              "S3OutputPath" : "s3://${aws_s3_bucket.monitoring_data.bucket}/models"
            },
            "ResourceConfig" : {
              "InstanceCount" : 1,
              "InstanceType" : "ml.m5.large",
              "VolumeSizeInGB" : 30
            },
            "StoppingCondition" : {
              "MaxRuntimeInSeconds" : 3600
            },
            "RoleArn" : "${aws_iam_role.sagemaker_role.arn}"
          }
        },
        {
          "Name" : "ModelEvaluation",
          "Type" : "Processing",
          "DependsOn" : ["TrainModel"],
          "Arguments" : {
            "ProcessingResources" : {
              "ClusterConfig" : {
                "InstanceCount" : 1,
                "InstanceType" : "ml.m5.large",
                "VolumeSizeInGB" : 20
              }
            },
            "AppSpecification" : {
              "ImageUri" : "246618743249.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-xgboost:1.7-1",
              "ContainerEntrypoint" : ["bash", "-c", "pip install xgboost==1.7.4 pandas scikit-learn matplotlib && python /opt/ml/processing/code/evaluate.py"]
            }
            "ProcessingInputs" : [
              {
                "InputName" : "test-data",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/features/test",
                  "LocalPath" : "/opt/ml/processing/input/test",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              },
              {
                "InputName" : "model",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/models",
                  "LocalPath" : "/opt/ml/processing/input/model",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              },
              {
                "InputName" : "code",
                "S3Input" : {
                  "S3Uri" : "s3://${aws_s3_bucket.baseline_dataset.bucket}/code",
                  "LocalPath" : "/opt/ml/processing/code",
                  "S3DataType" : "S3Prefix",
                  "S3InputMode" : "File"
                }
              }
            ],
            "ProcessingOutputConfig" : {
              "Outputs" : [
                {
                  "OutputName" : "evaluation",
                  "S3Output" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/evaluation",
                    "LocalPath" : "/opt/ml/processing/output/evaluation",
                    "S3UploadMode" : "EndOfJob"
                  }
                }
              ]
            },
            "RoleArn" : "${aws_iam_role.sagemaker_role.arn}"
          }
        },
      ]
    })
  }

  tags = [
    {
      key   = "environment"
      value = var.environment
    },
    {
      key   = "project"
      value = var.project_name
    }
  ]

  depends_on = [
    aws_s3_object.process_script,
    aws_s3_object.evaluate_script,
    aws_s3_object.custom_train_script,
    aws_s3_object.package_script,
    aws_s3_object.inference_script,
    aws_s3_object.debug_utils_script,
    aws_s3_object.generate_baseline_script,
    aws_sagemaker_model_package_group.asl_model_group
  ]
}

# EventBridge rules for triggering retraining
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

# EventBridge targets to trigger the pipeline
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
      value = aws_sagemaker_monitoring_schedule.data_monitoring_schedule.name
    }
  }
}

# Output pipeline ARN 
output "retraining_pipeline_name" {
  value = awscc_sagemaker_pipeline.model_retraining_pipeline.pipeline_name
}

# Output S3 location for packaged models
output "packaged_models_location" {
  value = "s3://${aws_s3_bucket.monitoring_data.bucket}/packaged-models/"
}
