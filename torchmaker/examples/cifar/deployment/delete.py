from sagemaker.pytorch.model import PyTorchPredictor
from torchmaker.config import AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME

predictor = PyTorchPredictor(endpoint_name=AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME)

predictor.delete_endpoint()