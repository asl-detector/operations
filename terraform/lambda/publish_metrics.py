import json
import boto3
import os
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3 = boto3.client("s3")
cloudwatch = boto3.client("cloudwatch")


def handler(event, context):
    """
    Reads model evaluation results from S3 and publishes them as CloudWatch metrics
    """
    try:
        bucket = event["detail"]["bucket"]["name"]
        key = event["detail"]["object"]["key"]

        logger.info(f"Processing evaluation file: s3://{bucket}/{key}")

        # Only process evaluation.json files
        if not key.endswith("evaluation.json"):
            logger.info(f"Skipping non-evaluation file: {key}")
            return

        # Get the evaluation file
        response = s3.get_object(Bucket=bucket, Key=key)
        evaluation_data = json.loads(response["Body"].read().decode("utf-8"))

        # Extract metrics
        accuracy = evaluation_data.get("accuracy", 0)
        auc = evaluation_data.get("auc", 0)

        # Put metrics to CloudWatch
        cloudwatch.put_metric_data(
            Namespace="Custom/ModelMetrics",
            MetricData=[
                {"MetricName": "ModelAccuracy", "Value": accuracy, "Unit": "None"},
                {"MetricName": "ModelAUC", "Value": auc, "Unit": "None"},
            ],
        )

        logger.info(f"Published metrics: Accuracy={accuracy}, AUC={auc}")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Metrics published successfully",
                    "accuracy": accuracy,
                    "auc": auc,
                }
            ),
        }

    except Exception as e:
        logger.error(f"Error processing evaluation file: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(
                {"message": f"Error processing evaluation file: {str(e)}"}
            ),
        }
