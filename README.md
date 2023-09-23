# keras-lightning
A lightweight, Keras-inspired wrapper for PyTorch Lightning.

## Features
- **No Checkpointing by Default**: Keeps your HD clean.
- **GPU Integration**: Automatically utilizes GPU if available.
- **Flexible Trainer Arguments**: Pass Trainer arguments directly through the `fit` function using kwargs.

## Installation
```
pip install keras-lightning
```

## Quick Start: MNIST Example
```python
import torch
from torch import nn
from keras_lightning import KLModel, SparseCategoricalAccuracy

# Loading the MNIST dataset
x_train, x_test, y_train, y_test = load_mnist() # Replace this with your MNIST data loading function 

# Defining the model
model = KLModel(nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
))

# Compiling the model
model.compile(
    optimizer=torch.optim.Adam(model.parameters(), 1e-3),
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=10),
    loss=nn.functional.cross_entropy,
    metrics={'acc': SparseCategoricalAccuracy()}
)

# Training the model
model.fit(
    x_train, y_train, 
    x_test=x_test, y_test=y_test,
    epochs=20, 
    batch_size=32,
)
```

## Contributions
We welcome contributions! Feel free to open an issue or submit a pull request
