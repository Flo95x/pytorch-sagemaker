from sagemaker.pytorch.model import PyTorchPredictor
from sagemaker.huggingface.model import HuggingFacePredictor
from torchmaker.config import SAGEMAKER_ROBERTA_ENDPOINT_NAME

predictor = PyTorchPredictor(endpoint_name=SAGEMAKER_ROBERTA_ENDPOINT_NAME)

predictor.delete_endpoint()