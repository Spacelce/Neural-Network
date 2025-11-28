# MNIST Neural Network from Scratch

A simple 2-layer neural network built from scratch using only NumPy to classify handwritten digits from the MNIST dataset.

## Model Architecture

```bash
Input Layer:    784 neurons (28x28 pixel images)
Hidden Layer:   10 neurons (ReLU activation)
Output Layer:   10 neurons (Softmax activation)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib

### Installation

1. Clone this repository:

```bash
git clone https://github.com/yourusername/neural-network.git
cd neural-network
```

2. Create a virtual environment (recommended):

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Usage

Run the training script:

```bash
python3 neural_network.py
```

The script will:

1. Download the MNIST dataset automatically (if not already present)
2. Train the neural network for 500 iterations
3. Display accuracy every 10 iterations
4. Save prediction visualizations

## Acknowledgments

Model based on Samson Zhang's video

MNIST dataset by Yann LeCun

GitHub mirror for MNIST data

