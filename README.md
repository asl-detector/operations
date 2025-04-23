# Operations

This directory contains ML operations (MLOps) infrastructure and code for the ASL Dataset project, including ML pipelines, monitoring, and deployment resources.

## Components

- `code/` - ML pipeline code:
  - `train.py` - Model training script
  - `evaluate.py` - Model evaluation script
  - `inference.py` - Model inference script
  - `process.py` - Data processing script
  - `debug_utils.py` - Debugging utilities
  - `generate_baseline.py` - Generate baseline statistics for monitoring
  - `package_model.py` - Package models for deployment
  - `requirements.txt` - Python dependencies

- `model/` - Initial model artifacts:
  - `initial-model.tar.gz` - Base model for training
  - `constraints/` - Model constraints for monitoring
  - `statistics/` - Baseline statistics

- `terraform/` - Infrastructure as code:
  - `pipeline.tf` - ML pipeline configuration
  - `monitoring.tf` - Model monitoring setup
  - `sagemaker.tf` - SageMaker resources configuration
  - `s3_uploads.tf` - Configuration for S3 asset uploads
  - `lambda/` - Lambda functions for pipeline operations
  - `test_data/` - Test data for pipeline validation

## Usage

This module sets up the operational infrastructure for training, monitoring, and serving ASL translation models.

```bash
cd terraform
terraform init
terraform plan
terraform apply
```

To run the pipeline manually:

```bash
cd terraform
./run_pipeline.sh
```
