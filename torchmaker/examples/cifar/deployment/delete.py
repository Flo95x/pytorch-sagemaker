from sagemaker.pytorch.model import PyTorchPredictor
from torchmaker.config import SAGEMAKER_CIFAR_ENDPOINT_NAME

predictor = PyTorchPredictor(endpoint_name=SAGEMAKER_CIFAR_ENDPOINT_NAME)

predictor.delete_endpoint()