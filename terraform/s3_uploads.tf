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
