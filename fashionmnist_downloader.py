# File: fashion_mnist_downloader.py
import os
import gzip
import numpy as np
import urllib.request

def download_fashion_mnist(data_dir='./data'):
    """
    Downloads and extracts the FashionMNIST dataset.

    Args:
        data_dir (str): Directory to save the dataset.

    Returns:
        Tuple of NumPy arrays: (train_images, train_labels, test_images, test_labels)
    """
    # URLs for the dataset files
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    # Ensure the data directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Download and extract files
    data = {}
    for key, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        with gzip.open(filepath, 'rb') as f:
            if 'images' in key:
                data[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28, 28)
            else:
                data[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    return data["train_images"], data["train_labels"], data["test_images"], data["test_labels"]
