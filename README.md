# pytorch-sagemaker
Project showing examples of pytorch model training and sagemaker deployment

## Requirements
Install project as package `torchmaker` with `pip3 install -e .`.

Create `config.py` with help of `config.example.py`. Required changes are **SAGEMAKER_ROLE** and **S3_MODEL_DATA_BUCKET**.

## Available Models

| Model   | Trainable | Deployable |
|---------|-----------|------------|
| Cifar   | yes       | yes        |
| Roberta | no        | yes        |
|         |           |            |

## Example Usage
Pick one

