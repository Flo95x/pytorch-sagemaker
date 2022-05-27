from torchmaker import config
from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

model = PyTorchModel(
    entry_point="inference.py",
    #source_dir="code",
    role=config.SAGEMAKER_ROLE,
    model_data=config.S3_CIFAR_MODEL_DATA_PATH,
    framework_version="1.5.0",
    py_version="py3",
)

if config.AWS_SAGEMAKER_SERVERLESS:
    from sagemaker.serverless import ServerlessInferenceConfig

    serverless_config = ServerlessInferenceConfig(memory_size_in_mb=4096,
                                                  max_concurrency=1)  # max_concurrency=3 3 was not possible with high torch version

    predictor = model.deploy(
        endpoint_name=config.AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        serverless_inference_config=serverless_config)
else:
    predictor = model.deploy(
        endpoint_name=config.AWS_SAGEMAKER_CIFAR_ENDPOINT_NAME,
        initial_instance_count=1,
        instance_type=config.AWS_SAGEMAKER_INSTANCE_TYPE,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )


import numpy as np

dummy_data = {"inputs": np.random.rand(16, 3, 32, 32).tolist()}
predictions = predictor.predict(dummy_data)

