import numpy as np
import torchvision

from sagemaker.pytorch.model import PyTorchPredictor

from torchmaker.config import SAGEMAKER_ROBERTA_ENDPOINT_NAME
from torchmaker.functions.profiling_fns import log_time
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = PyTorchPredictor(endpoint_name=SAGEMAKER_ROBERTA_ENDPOINT_NAME,
                             serializer=JSONSerializer(),
                             deserializer=JSONDeserializer())

aws_input_data = {"inputs": ["Hey this is a test sentence","Flo"]}
with log_time("Roberta prediction"):
    predictions = predictor.predict(aws_input_data)

print("Shape of prediction response", np.array(predictions).shape)





