# Upload initial files to baseline dataset bucket

resource "aws_s3_object" "process_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/process.py"
  source = "../code/process.py"
  etag   = filemd5("../code/process.py")
}

resource "aws_s3_object" "evaluate_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/evaluate.py"
  source = "../code/evaluate.py"
  etag   = filemd5("../code/evaluate.py")
}

resource "aws_s3_object" "train_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/train.py"
  source = "../code/train.py"
  etag   = filemd5("../code/train.py")
}

resource "aws_s3_object" "inference_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/inference.py"
  source = "../code/inference.py"
  etag   = filemd5("../code/inference.py")
}

resource "aws_s3_object" "data_constraints" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "constraints/data_constraints.json"
  source = "../model/constraints/data_constraints.json"
  etag   = filemd5("../model/constraints/data_constraints.json")
}

resource "aws_s3_object" "quality_constraints" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "constraints/quality_constraints.json"
  source = "../model/constraints/quality_constraints.json"
  etag   = filemd5("../model/constraints/quality_constraints.json")
}

resource "aws_s3_object" "data_statistics" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "statistics/data_statistics.json"
  source = "../model/statistics/data_statistics.json"
  etag   = filemd5("../model/statistics/data_statistics.json")
}

resource "aws_s3_object" "quality_statistics" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "statistics/quality_statistics.json"
  source = "../model/statistics/quality_statistics.json"
  etag   = filemd5("../model/statistics/quality_statistics.json")
}

resource "aws_s3_object" "initial_model" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "models/initial-model.tar.gz"
  source = "../model/initial-model.tar.gz"
  etag   = filemd5("../model/initial-model.tar.gz")
}

# Create empty ground truth labels file
resource "aws_s3_object" "ground_truth" {
  bucket  = aws_s3_bucket.baseline_dataset.bucket
  key     = "ground-truth/labels.csv"
  content = "timestamp,prediction,label\n"
}

resource "aws_s3_object" "package_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/package_model.py"
  source = "../code/package_model.py"
  etag   = filemd5("../code/package_model.py")
}

# Create proper directory structure for input data
resource "aws_s3_object" "asl_directory" {
  bucket       = aws_s3_bucket.monitoring_data.bucket
  key          = "pose-data/asl/"
  content      = ""
  content_type = "application/x-directory"
}

resource "aws_s3_object" "no_asl_directory" {
  bucket       = aws_s3_bucket.monitoring_data.bucket
  key          = "pose-data/no_asl/"
  content      = ""
  content_type = "application/x-directory"
}

resource "aws_s3_object" "test_positive" {
  bucket = aws_s3_bucket.monitoring_data.bucket
  key    = "pose-data/asl/ash_landmark.json"
  source = "./test_data/ash_landmark.json"
  etag   = filemd5("./test_data/ash_landmark.json")
}

resource "aws_s3_object" "test_negative" {
  bucket = aws_s3_bucket.monitoring_data.bucket
  key    = "pose-data/no_asl/JAKE_landmark.json"
  source = "./test_data/JAKE_landmark.json"
  etag   = filemd5("./test_data/JAKE_landmark.json")
}

resource "aws_s3_object" "debug_utils_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/debug_utils.py"
  source = "../code/debug_utils.py"
  etag   = filemd5("../code/debug_utils.py")
}

resource "aws_s3_object" "requirements_file" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/requirements.txt"
  source = "../code/requirements.txt"
  etag   = filemd5("../code/requirements.txt")
}

# resource "aws_s3_object" "baseline_dataset" {
#   bucket = aws_s3_bucket.baseline_dataset.bucket
#   key    = "data/baseline.csv"
#   source = "../baseline.csv"
#   etag   = filemd5("../baseline.csv")
# }

resource "aws_s3_object" "generate_baseline_script" {
  bucket = aws_s3_bucket.baseline_dataset.bucket
  key    = "code/generate_baseline.py"
  source = "../code/generate_baseline.py"
  etag   = filemd5("../code/generate_baseline.py")
}
