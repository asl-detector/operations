import boto3

client = boto3.client("sagemaker")

# Use your PipelineExecutionArn from the error message
execution_arn = "arn:aws:sagemaker:us-west-2:223537960975:pipeline/asl-detection-dev-retraining-pipeline/execution/ze02f7hybrnr"

# List the steps for this execution
response = client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)

# Look for steps with status 'Failed'
for step in response["PipelineExecutionSteps"]:
    if step["StepStatus"] == "Failed":
        print(f"Failed step: {step['StepName']}")
        print(f"Failure reason: {step.get('FailureReason', 'No reason provided')}")
