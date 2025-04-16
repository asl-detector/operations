import json
import os
import boto3
import urllib.parse

sagemaker = boto3.client("sagemaker")
s3 = boto3.client("s3")


def handler(event, context):
    """Updates the 'latest' model version in S3 for edge clients to download"""
    print(f"Processing event: {json.dumps(event)}")

    # Get environment variables
    model_bucket = os.environ.get("MODEL_BUCKET")
    model_package_group = os.environ.get("MODEL_PACKAGE_GROUP")

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

        # Get the S3 URI for the model data
        model_data_url = model_package["InferenceSpecification"]["Containers"][0][
            "ModelDataUrl"
        ]

        # Parse S3 URI to get bucket and key
        parsed_url = urllib.parse.urlparse(model_data_url)
        source_bucket = parsed_url.netloc
        source_key = parsed_url.path.lstrip("/")

        # Create a "latest" version of the model
        latest_key = "packaged-models/latest/model.tar.gz"

        print(
            f"Copying model from {source_bucket}/{source_key} to {model_bucket}/{latest_key}"
        )

        # Copy the model to the latest location
        s3.copy_object(
            Bucket=model_bucket,
            Key=latest_key,
            CopySource=f"{source_bucket}/{source_key}",
        )

        # Create version metadata file
        version_info = {
            "modelPackageArn": model_package_arn,
            "creationTime": model_package["CreationTime"].isoformat(),
            "modelVersion": model_package["ModelPackageVersion"],
            "sourceUrl": model_data_url,
        }

        s3.put_object(
            Bucket=model_bucket,
            Key="packaged-models/latest/version.json",
            Body=json.dumps(version_info, indent=2),
            ContentType="application/json",
        )

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": "Latest model updated successfully",
                    "modelPackageArn": model_package_arn,
                    "latestUrl": f"s3://{model_bucket}/{latest_key}",
                }
            ),
        }

    except Exception as e:
        print(f"Error updating latest model: {str(e)}")
        return {
            "statusCode": 500,
            "body": json.dumps(f"Error updating latest model: {str(e)}"),
        }
