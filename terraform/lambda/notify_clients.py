import json
import boto3
import os

sagemaker = boto3.client("sagemaker")
sns = boto3.client("sns")


def handler(event, context):
    """Notifies edge clients of new model version"""
    print(f"Processing event: {json.dumps(event)}")

    # Get environment variables
    model_bucket = os.environ.get("MODEL_BUCKET")
    model_package_group = os.environ.get("MODEL_PACKAGE_GROUP")
    sns_topic_arn = os.environ.get("SNS_TOPIC_ARN")

    try:
        # Get latest model package
        response = sagemaker.list_model_packages(
            ModelPackageGroupName=model_package_group,
            SortBy="CreationTime",
            SortOrder="Descending",
            MaxResults=1,
        )

        if not response["ModelPackageSummaryList"]:
            return {
                "statusCode": 404,
                "body": json.dumps("No model packages found in group"),
            }

        model_package_arn = response["ModelPackageSummaryList"][0]["ModelPackageArn"]
        model_package = sagemaker.describe_model_package(
            ModelPackageName=model_package_arn
        )

        # Send notification via SNS
        notification = {
            "messageType": "MODEL_UPDATE",
            "modelPackageArn": model_package_arn,
            "modelVersion": model_package["ModelPackageVersion"],
            "creationTime": model_package["CreationTime"].isoformat(),
            "downloadUrl": f"s3://{model_bucket}/packaged-models/latest/model.tar.gz",
            "versionInfoUrl": f"s3://{model_bucket}/packaged-models/latest/version.json",
        }

        # Publish to SNS topic
        sns.publish(
            TopicArn=sns_topic_arn,
            Message=json.dumps(notification),
            Subject="New ASL Detection Model Available",
            MessageAttributes={
                "modelVersion": {
                    "DataType": "String",
                    "StringValue": str(model_package["ModelPackageVersion"]),
                }
            },
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Notification sent to edge clients",
                    "modelPackageArn": model_package_arn,
                    "notificationDetails": notification,
                }
            ),
        }

    except Exception as e:
        print(f"Error sending notification: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error sending notification: {str(e)}"),
        }
