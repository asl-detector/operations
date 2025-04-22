import boto3
import os
import json


def handler(event, context):
    """Copy model from operations account to artifacts account"""
    print(f"Processing event: {json.dumps(event)}")

    # Get environment variables
    source_bucket = os.environ.get("SOURCE_BUCKET")
    target_bucket = os.environ.get("TARGET_BUCKET")
    role_arn = os.environ.get("ROLE_ARN")

    # Source S3 client in operations account
    s3_source = boto3.client("s3")

    try:
        # Assume role in artifacts account
        sts_client = boto3.client("sts")
        assumed_role = sts_client.assume_role(
            RoleArn=role_arn, RoleSessionName="ModelTransferSession"
        )

        # Create S3 client with temporary credentials
        credentials = assumed_role["Credentials"]
        s3_target = boto3.client(
            "s3",
            aws_access_key_id=credentials["AccessKeyId"],
            aws_secret_access_key=credentials["SecretAccessKey"],
            aws_session_token=credentials["SessionToken"],
        )

        # Get the key of the model file from the event if provided
        if "Records" in event and len(event["Records"]) > 0:
            s3_event = event["Records"][0]["s3"]
            source_key = s3_event["object"]["key"]
        else:
            # Default to the latest model
            source_key = "packaged-models/model.tar.gz"

        # Update this section to accept JSON files as well
        if not (source_key.endswith(".tar.gz") or source_key.endswith(".json")):
            print(f"Skipping non-model file: {source_key}")
            return {
                "statusCode": 200,
                "body": json.dumps({"message": "Skipped non-model file"}),
            }

        # Define target key - update based on file extension
        if source_key.endswith(".json"):
            target_key = "models/asl-detection-model.json"
        else:
            target_key = "models/asl-detection-model.tar.gz"

        print(
            f"Copying {source_key} from {source_bucket} to {target_bucket}/{target_key}"
        )

        # Copy the object
        copy_source = {"Bucket": source_bucket, "Key": source_key}
        s3_target.copy_object(
            CopySource=copy_source, Bucket=target_bucket, Key=target_key
        )

        # Create version metadata
        version_info = {
            "modelVersion": datetime.datetime.now().strftime("%Y%m%d%H%M%S"),
            "sourceUrl": f"s3://{source_bucket}/{source_key}",
            "creationTime": datetime.datetime.now().isoformat(),
        }

        s3_target.put_object(
            Bucket=target_bucket,
            Key="models/version.json",
            Body=json.dumps(version_info, indent=2),
            ContentType="application/json",
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Model copied successfully",
                    "source": f"s3://{source_bucket}/{source_key}",
                    "destination": f"s3://{target_bucket}/{target_key}",
                }
            ),
        }

    except Exception as e:
        print(f"Error copying model: {str(e)}")
        return {"statusCode": 500, "body": json.dumps(f"Error copying model: {str(e)}")}
