import boto3
import os
import json
import datetime


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

        # Determine source key
        source_key = None

        # Check if triggered by S3 event
        if "Records" in event and len(event["Records"]) > 0:
            s3_event = event["Records"][0]["s3"]
            source_key = s3_event["object"]["key"]
            print(f"Using source key from event: {source_key}")

        # If not from event or not a valid model file, search for a model
        if not source_key or not (
            source_key.endswith(".tar.gz") or source_key.endswith(".json")
        ):
            # Try to find standardized model from evaluation first
            print("Searching for standardized model...")
            try:
                eval_response = s3_source.list_objects_v2(
                    Bucket=source_bucket,
                    Prefix="evaluation/standardized/",
                )

                if "Contents" in eval_response:
                    model_files = [
                        obj
                        for obj in eval_response["Contents"]
                        if obj["Key"].endswith(".json")
                    ]
                    if model_files:
                        source_key = sorted(
                            model_files, key=lambda x: x["LastModified"], reverse=True
                        )[0]["Key"]
                        print(f"Found standardized model: {source_key}")
            except Exception as e:
                print(f"Error searching for standardized model: {e}")

            # If still no model found, try direct model from training
            if not source_key:
                print("Searching for direct training output...")
                try:
                    # Try to get directly from the URI provided
                    source_key = "models/pipelines-4dcua81pcark-TrainModel-SvOWwwZnZS/output/model.tar.gz"

                    # Check if this specific model exists
                    try:
                        s3_source.head_object(Bucket=source_bucket, Key=source_key)
                        print(f"Found specific model at {source_key}")
                    except:
                        print(
                            f"Specific model not found, searching for latest model..."
                        )
                        source_key = None

                    # If specific model not found, look for latest
                    if not source_key:
                        model_response = s3_source.list_objects_v2(
                            Bucket=source_bucket,
                            Prefix="models/",
                        )

                        if "Contents" in model_response:
                            model_archives = [
                                obj
                                for obj in model_response["Contents"]
                                if obj["Key"].endswith(".tar.gz")
                                and "TrainModel" in obj["Key"]
                            ]
                            if model_archives:
                                source_key = sorted(
                                    model_archives,
                                    key=lambda x: x["LastModified"],
                                    reverse=True,
                                )[0]["Key"]
                                print(f"Found latest model archive: {source_key}")
                except Exception as e:
                    print(f"Error searching for training output: {e}")

        if not source_key:
            return {
                "statusCode": 404,
                "body": json.dumps("No model file found to copy"),
            }

        # Define target key based on file extension
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
