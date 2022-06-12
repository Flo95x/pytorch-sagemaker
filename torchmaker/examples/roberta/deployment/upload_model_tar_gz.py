import boto3
from torchmaker.config import S3_MODEL_DATA_BUCKET, S3_ROBERTA_MODEL_DATA_KEY

s3 = boto3.resource('s3')
s3.Bucket(S3_MODEL_DATA_BUCKET).upload_file("model.tar.gz", S3_ROBERTA_MODEL_DATA_KEY)