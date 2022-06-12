from torchmaker import config
from sagemaker.pytorch import PyTorchModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

model = PyTorchModel(
    entry_point="inference.py",
    role=config.SAGEMAKER_ROLE,
    model_data=config.S3_ROBERTA_MODEL_DATA_PATH,
    framework_version="1.5.0",
    py_version="py3",
)

if False: #config.AWS_SAGEMAKER_SERVERLESS:
    from sagemaker.serverless import ServerlessInferenceConfig

    serverless_config = ServerlessInferenceConfig(memory_size_in_mb=2048,
                                                  max_concurrency=1)  # max_concurrency=3 3 was not possible with high torch version

    predictor = model.deploy(
        endpoint_name=config.AWS_SAGEMAKER_ROBERTA_ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        serverless_inference_config=serverless_config)
else:
    predictor = model.deploy(
        endpoint_name=config.AWS_SAGEMAKER_ROBERTA_ENDPOINT_NAME,
        initial_instance_count=1,
        instance_type="ml.c5.large", # "ml.m5.xlarge", # config.AWS_SAGEMAKER_INSTANCE_TYPE,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )


