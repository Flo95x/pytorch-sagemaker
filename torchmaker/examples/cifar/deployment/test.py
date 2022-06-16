import numpy as np
import torchvision

from sagemaker.pytorch.model import PyTorchPredictor

from torchmaker.config import SAGEMAKER_CIFAR_ENDPOINT_NAME
from torchmaker.examples.cifar.dataloader import get_train_test_loader, classes, batch_size_default
from torchmaker.functions.profiling_fns import log_time
from torchmaker.functions.image_fns import imshow
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

predictor = PyTorchPredictor(endpoint_name=SAGEMAKER_CIFAR_ENDPOINT_NAME,
                             serializer=JSONSerializer(),
                             deserializer=JSONDeserializer())

trainloader, testloader = get_train_test_loader()

# get some random training images
dataiter = iter(testloader)
images, labels = dataiter.next()

print(type(images), images.shape)
image_list = images.tolist()
aws_input_data = {"inputs": image_list}
with log_time("Cifar prediction"):
    predictions = predictor.predict(aws_input_data)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'pred: {classes[np.argmax(predictions[j])]:5s}, true: {classes[labels[j]]:5s}' for j in range(batch_size_default)))





