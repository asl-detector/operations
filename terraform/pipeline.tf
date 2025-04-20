resource "awscc_sagemaker_pipeline" "model_retraining_pipeline" {
  pipeline_name         = "${var.project_name}-${var.environment}-retraining-pipeline"
  pipeline_display_name = "${var.project_name}-${var.environment}-retraining-pipeline"
  role_arn              = aws_iam_role.sagemaker_role.arn

  pipeline_definition = {
    pipeline_definition_body = jsonencode({
      "Version" : "2020-12-01",
      "Steps" : [
        {
          "Name" : "FeatureProcessing",
          "Type" : "Processing",
          "Arguments" : {
            "ProcessingResources" : {
              "ClusterConfig" : {
                "InstanceCount" : 1,
                "InstanceType" : "ml.m5.large",
                "VolumeSizeInGB" : 30
              }
            },
            "AppSpecification" : {
              "ImageUri" : "246618743249.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
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
              "TrainingImage" : "433757028032.dkr.ecr.${var.aws_region}.amazonaws.com/xgboost:1",
              "TrainingInputMode" : "File"
            },
            "HyperParameters" : {
              "num_round" : "300",
              "max_depth" : "6",
              "eta" : "0.05",
              "gamma" : "0.1",
              "min_child_weight" : "2",
              "subsample" : "0.8",
              "colsample_bytree" : "0.8",
              "objective" : "binary:logistic",
              "eval_metric" : "auc",
              "csv_weights" : "0",
              "csv_header" : "absent",
              "label_column" : "0"
            }
            "InputDataConfig" : [
              {
                "ChannelName" : "train",
                "DataSource" : {
                  "S3DataSource" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/features/train",
                    "S3DataType" : "S3Prefix"
                  }
                },
                "ContentType" : "text/csv"
              },
              {
                "ChannelName" : "validation",
                "DataSource" : {
                  "S3DataSource" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/features/validation",
                    "S3DataType" : "S3Prefix"
                  }
                },
                "ContentType" : "text/csv"
              }
            ]
            "OutputDataConfig" : {
              "S3OutputPath" : "s3://${aws_s3_bucket.monitoring_data.bucket}/models"
            },
            "ResourceConfig" : {
              "InstanceCount" : 1,
              "InstanceType" : "ml.m5.large",
              "VolumeSizeInGB" : 30
            }
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
              "ImageUri" : "246618743249.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
              "ContainerEntrypoint" : ["bash", "-c", "pip install -r /opt/ml/processing/code/requirements.txt && python /opt/ml/processing/code/evaluate.py"]
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
        {
          "Name" : "PackageModel",
          "Type" : "Processing",
          "DependsOn" : ["ModelEvaluation"],
          "Arguments" : {
            "ProcessingResources" : {
              "ClusterConfig" : {
                "InstanceCount" : 1,
                "InstanceType" : "ml.m5.large",
                "VolumeSizeInGB" : 20
              }
            },
            "AppSpecification" : {
              "ImageUri" : "246618743249.dkr.ecr.${var.aws_region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3",
              "ContainerEntrypoint" : ["bash", "-c", "pip install -r /opt/ml/processing/code/requirements.txt && python /opt/ml/processing/code/package_model.py --model /opt/ml/processing/input/model --output-dir /opt/ml/processing/output/packaged"]
            }
            "ProcessingInputs" : [
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
                  "OutputName" : "packaged-model",
                  "S3Output" : {
                    "S3Uri" : "s3://${aws_s3_bucket.monitoring_data.bucket}/packaged-models",
                    "LocalPath" : "/opt/ml/processing/output/packaged",
                    "S3UploadMode" : "EndOfJob"
                  }
                }
              ]
            },
            "RoleArn" : "${aws_iam_role.sagemaker_role.arn}"
          }
        },
        {
          "Name" : "RegisterModel",
          "Type" : "RegisterModel",
          "DependsOn" : ["PackageModel"],
          "Arguments" : {
            "ModelPackageGroupName" : "${var.project_name}-${var.environment}-models",
            "ModelPackageDescription" : "ASL detection model for edge deployment",
            "ModelApprovalStatus" : "Approved",
            "ModelMetrics" : {
              "ModelQuality" : {
                "Statistics" : {
                  "ContentType" : "application/json",
                  "S3Uri" : {
                    "Get" : "Steps.ModelEvaluation.ProcessingOutputConfig.Outputs['evaluation'].S3Output.S3Uri"
                  }
                }
              }
            },
            "InferenceSpecification" : {
              "Containers" : [
                {
                  "Image" : "433757028032.dkr.ecr.${var.aws_region}.amazonaws.com/xgboost:1",
                  "ModelDataUrl" : "s3://${aws_s3_bucket.monitoring_data.bucket}/packaged-models/initial-model.tar.gz"
                }
              ],
              "SupportedContentTypes" : ["text/csv"],
              "SupportedResponseMIMETypes" : ["text/csv"]
            }
          }
        }
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
    aws_s3_object.train_script,
    aws_s3_object.package_script,
    aws_s3_object.inference_script,
    aws_s3_object.debug_utils_script,
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
