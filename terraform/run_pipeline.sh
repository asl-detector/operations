#!/bin/bash

# Set AWS region
AWS_REGION="us-west-2"

# Get values from terraform output
MONITORING_BUCKET=$(terraform output -raw monitoring_bucket)
export MONITORING_BUCKET
PIPELINE_NAME=$(terraform output -raw retraining_pipeline_name)
export PIPELINE_NAME
MODEL_GROUP=$(terraform output -raw model_package_group_name)
export MODEL_GROUP

echo "Starting pipeline execution..."
# Start pipeline execution and capture the output
EXECUTION_RESPONSE=$(aws sagemaker start-pipeline-execution \
  --pipeline-name $PIPELINE_NAME \
  --region $AWS_REGION)

# Extract the execution ID from the ARN using string manipulation
EXECUTION_ARN=$(echo $EXECUTION_RESPONSE | jq -r '.PipelineExecutionArn')
EXECUTION_ID=$(echo $EXECUTION_ARN | cut -d'/' -f4)

echo "Pipeline execution started with ID: $EXECUTION_ID"

# Update main.py with the new execution ARN - compatible with both macOS and Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
  # macOS version
  sed -i '' "s|execution_arn = \".*\"|execution_arn = \"$EXECUTION_ARN\"|" main.py
else
  # Linux version
  sed -i "s|execution_arn = \".*\"|execution_arn = \"$EXECUTION_ARN\"|" main.py
fi

echo "Updated main.py with new execution ARN"

# Describe the pipeline execution
echo "Describing pipeline execution..."
aws sagemaker describe-pipeline-execution \
  --pipeline-execution-arn $EXECUTION_ARN \
  --region $AWS_REGION

# Run the Python script
echo "Running main.py..."
python main.py
