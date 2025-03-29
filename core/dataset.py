import torch
from torchvision import datasets, transforms
import os

def download_mnist(dataset_directory: str, transforms) -> tuple:
    if not os.path.exists(dataset_directory):
        os.mkdir(dataset_directory)

    train_data_path = f"{dataset_directory}/MNIST/raw/train-images-idx3-ubyte"
    test_data_path = f"{dataset_directory}/MNIST/raw/t10k-images-idx3-ubyte"

    train_data, test_data = None, None

    if not os.path.exists(train_data_path) or not os.path.exists(test_data_path):
        print(f"MNIST dataset not found in '{dataset_directory}', downloading...")
        train_data = datasets.MNIST(root=dataset_directory, train=True, download=True, transform=transforms)
        test_data = datasets.MNIST(root=dataset_directory, train=False, download=True, transform=transforms)
        print("Done")
    else:
        print("MNISTdataset found locally, loading...")
        train_data = datasets.MNIST(root=dataset_directory, train=True, download=False, transform=transforms)
        test_data = datasets.MNIST(root=dataset_directory, train=False, download=False, transform=transforms)
        print("Done")

    assert train_data is not None, "Training data is null, something went wrong :("
    assert test_data is not None, "Testing data is null, something went wrong :("

    return (train_data, test_data)
