import boto3
# import sys
# from pathlib import Path
# # add project root to pythonpath
# sys.path.append(str(Path(__file__).parent.parent.parent.parent))
#print(sys.path, "\n", __file__, os.getcwd())
from torchmaker.config import S3_MODEL_DATA_BUCKET, S3_CIFAR_MODEL_DATA_KEY

s3 = boto3.resource('s3')
s3.Bucket(S3_MODEL_DATA_BUCKET).upload_file("model.tar.gz", S3_CIFAR_MODEL_DATA_KEY)