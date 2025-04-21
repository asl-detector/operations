import boto3
import time
import json
from datetime import datetime


def monitor_pipeline_execution(execution_arn, interval=15):
    client = boto3.client("sagemaker")

    print(f"Starting to monitor pipeline execution: {execution_arn}")
    print(f"Checking every {interval} seconds...")
    print("=" * 80)

    # Keep track of previous step statuses
    previous_statuses = {}

    while True:
        # Get current execution status
        execution = client.describe_pipeline_execution(
            PipelineExecutionArn=execution_arn
        )

        # Check if the overall execution is complete
        current_status = execution["PipelineExecutionStatus"]
        if current_status in ["Failed", "Succeeded", "Stopped"]:
            print(
                f"\n=== Pipeline execution {current_status} at {datetime.now().strftime('%H:%M:%S')} ==="
            )
            break

        # Get details about individual steps
        response = client.list_pipeline_execution_steps(
            PipelineExecutionArn=execution_arn
        )

        # Check for status changes
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        status_changed = False

        for step in response["PipelineExecutionSteps"]:
            step_name = step["StepName"]
            step_status = step["StepStatus"]

            # Check if this is a new status for this step
            if (
                step_name not in previous_statuses
                or previous_statuses[step_name] != step_status
            ):
                status_changed = True
                previous_statuses[step_name] = step_status

                # Print status change
                print(f"[{current_time}] Step '{step_name}': {step_status}")

                # If failed, print the reason
                if step_status == "Failed":
                    reason = step.get("FailureReason", "No reason provided")
                    print(f"  Failure reason: {reason}")

        # If no statuses changed, just print a simple update
        if not status_changed:
            print(
                f"[{current_time}] No status changes. Overall status: {current_status}"
            )

        # Wait for the next check
        time.sleep(interval)

    # Final report
    print("\nFinal status of all steps:")
    response = client.list_pipeline_execution_steps(PipelineExecutionArn=execution_arn)
    for step in response["PipelineExecutionSteps"]:
        status = step["StepStatus"]
        print(f"- {step['StepName']}: {status}")
        if status == "Failed":
            reason = step.get("FailureReason", "No reason provided")
            print(f"  Failure reason: {reason}")


# Replace with your execution ARN
execution_arn = "arn:aws:sagemaker:us-west-2:223537960975:pipeline/asl-detection-dev-retraining-pipeline/execution/5a5rrlh0jpfg"

# Start monitoring
try:
    monitor_pipeline_execution(execution_arn)
except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
except Exception as e:
    print(f"\nError occurred: {str(e)}")
