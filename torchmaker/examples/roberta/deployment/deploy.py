from torchmaker import config
from sagemaker.pytorch import PyTorchModel
from sagemaker.huggingface import HuggingFaceModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# model = PyTorchModel(
#     entry_point="inference.py",
#     source_dir="code",
#     role=config.SAGEMAKER_ROLE,
#     model_data=config.S3_ROBERTA_MODEL_DATA_PATH,
#     framework_version="1.5.0",
#    #transformers_version="4.17", # transformers version used
#    #pytorch_version="1.10", # pytorch version used
#     py_version="py3",
# )

#question-answering
model = HuggingFaceModel(
   env={"HF_TASK": "feature-extraction"}, #"HF_MODEL_ID":'T-Systems-onsite/cross-en-de-roberta-sentence-transformer',
   #source_dir="code",
   model_data=config.S3_ROBERTA_MODEL_DATA_PATH,  # path to your trained SageMaker model
   role=config.SAGEMAKER_ROLE,                                            # IAM role with permissions to create an endpoint
   #transformers_version="4.6",                           # Transformers version used
   #pytorch_version="1.7",                                # PyTorch version used
   #py_version='py36',                                    # Python version used
    transformers_version="4.17",  # transformers version used
    pytorch_version="1.10",  # pytorch version used
    py_version='py38',  # Python version used

)

if False: #config.AWS_SAGEMAKER_SERVERLESS:
    from sagemaker.serverless import ServerlessInferenceConfig

    serverless_config = ServerlessInferenceConfig(memory_size_in_mb=4096,
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
        instance_type="ml.m5.xlarge", # config.AWS_SAGEMAKER_INSTANCE_TYPE,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer(),
    )


