# import sys
# from pathlib import Path
# # add project root to pythonpath
# sys.path.append(str(Path(__file__).parent.parent.parent.parent))
#print(sys.path, "\n", __file__, os.getcwd())
from torchmaker.config import SAGEMAKER_ROLE, AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME, AWS_SAGEMAKER_INSTANCE_TYPE, S3_CIFAR_MODEL_DATA_PATH, AWS_SAGEMAKER_SERVERLESS
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

model = PyTorchModel(
    entry_point="inference.py",
    #source_dir="code",
    role=SAGEMAKER_ROLE,
    model_data=S3_CIFAR_MODEL_DATA_PATH,
    framework_version="1.5.0",
    py_version="py3",
)

# set local_mode to False if you want to deploy on a remote
# SageMaker instance

local_mode = False

if local_mode:
    instance_type = "local"
else:
    instance_type = AWS_SAGEMAKER_INSTANCE_TYPE

if AWS_SAGEMAKER_SERVERLESS:
    from sagemaker.serverless import ServerlessInferenceConfig

    serverless_config = ServerlessInferenceConfig(memory_size_in_mb=4096,
                                                  max_concurrency=1)  # max_concurrency=3 3 was not possible with high torch version

    predictor = model.deploy(
        endpoint_name=AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        serverless_inference_config=serverless_config)
else:
    predictor = model.deploy(
        endpoint_name=AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME,
        initial_instance_count=1,
        instance_type=instance_type,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )


import numpy as np

dummy_data = {"inputs": np.random.rand(16, 3, 32, 32).tolist()}
predictions = predictor.predict(dummy_data)

