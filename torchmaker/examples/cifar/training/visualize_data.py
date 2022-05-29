
import torchvision

from torchmaker.examples.cifar.dataloader import get_train_test_loader, batch_size_default, classes
from torchmaker.functions.image_fns import imshow

trainloader, testloader = get_train_test_loader(batch_size=batch_size_default)


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size_default)))