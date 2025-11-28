import numpy as np
import urllib.request
import gzip
import os

def download_mnist():
    """
    Download MNIST dataset from the official source
    """
    base_url = "https://github.com/golbin/TensorFlow-MNIST/raw/master/mnist/data/"
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Downloading MNIST dataset...")
    for name, filename in files.items():
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        else:
            print(f"{filename} already exists")
    print("Download complete!")

def load_mnist_images(filename):
    """
    Load MNIST images from gz file
    """
    with gzip.open(filename, 'rb') as f:
        # Read magic number and dimensions
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        rows = int.from_bytes(f.read(4), 'big')
        cols = int.from_bytes(f.read(4), 'big')
        
        # Read image data
        data = np.frombuffer(f.read(), dtype=np.uint8)
        data = data.reshape(num_images, rows * cols)
    
    return data

def load_mnist_labels(filename):
    """
    Load MNIST labels from gz file
    """
    with gzip.open(filename, 'rb') as f:
        # Read magic number and number of labels
        magic = int.from_bytes(f.read(4), 'big')
        num_labels = int.from_bytes(f.read(4), 'big')
        
        # Read label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    
    return labels

def load_data():
    """
    Load and prepare MNIST data in the same format as Kaggle
    Returns: X_train, Y_train, X_dev, Y_dev (already normalized and transposed)
    """
    # Download if needed
    download_mnist()
    
    # Load training data
    print("Loading data...")
    train_images = load_mnist_images('data/train-images-idx3-ubyte.gz')
    train_labels = load_mnist_labels('data/train-labels-idx1-ubyte.gz')
    
    # Combine images and labels (like Kaggle CSV format)
    data = np.column_stack([train_labels, train_images])
    
    # Shuffle
    np.random.shuffle(data)
    
    # Split into dev and train sets
    data_dev = data[0:1000].T
    Y_dev = data_dev[0]
    X_dev = data_dev[1:] / 255.0  # Normalize
    
    data_train = data[1000:].T
    Y_train = data_train[0]
    X_train = data_train[1:] / 255.0  # Normalize
    
    print(f"Training set: {X_train.shape[1]} examples")
    print(f"Dev set: {X_dev.shape[1]} examples")
    print("Data loaded successfully!")
    
    return X_train, Y_train, X_dev, Y_dev

def load_test_data():
    """
    Load test data for final evaluation
    """
    download_mnist()
    
    test_images = load_mnist_images('data/t10k-images-idx3-ubyte.gz')
    test_labels = load_mnist_labels('data/t10k-labels-idx1-ubyte.gz')
    
    X_test = test_images.T / 255.0
    Y_test = test_labels
    
    return X_test, Y_test
