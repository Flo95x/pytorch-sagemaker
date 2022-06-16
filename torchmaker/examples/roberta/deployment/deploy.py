from torchmaker import config
from sagemaker.pytorch import PyTorchModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

model = HuggingFaceModel(
   model_data=config.S3_ROBERTA_MODEL_DATA_PATH,
   role=config.SAGEMAKER_ROLE,                    # iam role with permissions to create an Endpoint
   transformers_version="4.12",  # transformers version used
   pytorch_version="1.9",        # pytorch version used
   py_version='py38',            # python version used
)

if config.SAGEMAKER_ROBERTA_SERVERLESS:
    from sagemaker.serverless import ServerlessInferenceConfig

    serverless_config = ServerlessInferenceConfig(memory_size_in_mb=3072,
                                                  max_concurrency=5)  # max_concurrency=3 3 was not possible with high torch version

    predictor = model.deploy(
        endpoint_name=config.SAGEMAKER_ROBERTA_ENDPOINT_NAME,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
        serverless_inference_config=serverless_config)
else:
    predictor = model.deploy(
        endpoint_name=config.SAGEMAKER_ROBERTA_ENDPOINT_NAME,
        initial_instance_count=1,
        instance_type="ml.c5.large", # "ml.m5.xlarge", # config.AWS_SAGEMAKER_INSTANCE_TYPE,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )


