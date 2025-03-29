import torch
from torchvision import transforms

from core.dataset import *

# Download MNIST dataset if it doesnt already exist locally
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data, test_data = download_mnist("./data", transforms)
print(f"Train Data: {train_data}")
print(f"Test Data: {test_data}")
