# pytorch-sagemaker
Project showing examples of pytorch model training and sagemaker deployment

## Requirements
Install project as package `torchmaker` with `cd pytorch-sagemaker && pip3 install -e .`.

Create `config.py` with help of `config.example.py`. Required changes are **SAGEMAKER_ROLE** and **S3_MODEL_DATA_BUCKET**.

## Available Models

| Model    | Trainable | Deployable |  Explanation       |
|----------|-----------|------------|-------------|
| CIFAR-10 | yes       | yes        | CNN for image classification |
| Roberta  | no        | yes        | BERT transformer model for text embedding generation en/de |


## Example Usage
Pick one of the available example models in dir `torchmaker/examples`. 
Run one of the following .sh files, depending on whether you want to **train**, **deploy** or **train and deploy** a model:
* `train.sh`
* `deploy.sh`
* `train_and_deploy.sh`

Checkout `deployment/test.py` to see example sagemaker inference scripts, for testing if model can be called as intended.
Feel free to customize the models like you want.

If you want to **delete** AWS Sagemaker endpoint again call:
* `delete.sh`




