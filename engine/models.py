import torch
import torch.nn as nn
from torch.optim import Adam
from abc import ABC, abstractmethod
from .hp_config import ModelConfig  # Ensure we import your new config contract

class BaseModel(nn.Module, ABC):
    """The abstract base class for all mathematical models."""
    def __init__(self, learning_rate: float):
        super().__init__()
        self.learning_rate = learning_rate

    @abstractmethod
    def forward(self, X): pass

    @abstractmethod
    def loss(self, y_hat, y): pass

    def training_step(self, batch):
        X, y = batch[0], batch[1]
        y_hat = self(X)
        return self.loss(y_hat, y)

    def validation_step(self, batch):
        X, y = batch[0], batch[1]
        y_hat = self(X)
        return self.loss(y_hat, y)

    @abstractmethod
    def configure_optimizers(self): pass


class KMnistCNN(BaseModel):
    """A CNN optimized for 28x28 grayscale images, now using ModelConfig."""
    def __init__(self, config: ModelConfig):
        super().__init__(learning_rate=config.learning_rate)
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, config.hidden_units), # Uses config.hidden_units
            nn.ReLU(),
            nn.Linear(config.hidden_units, 10) 
        )

    def forward(self, X): return self.net(X)
    def loss(self, y_hat, y): return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)


class KMnistMLP(BaseModel):
    """A high-speed Multi-Layer Perceptron using ModelConfig."""
    def __init__(self, config: ModelConfig):
        super().__init__(learning_rate=config.learning_rate)
        self.config = config
        self.loss_fn = nn.CrossEntropyLoss()
        
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, self.config.hidden_units), # Uses config.hidden_units
            nn.ReLU(),
            nn.Dropout(p=self.config.dropout_rate),       # Uses config.dropout_rate
            nn.Linear(self.config.hidden_units, self.config.hidden_units // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_units // 2, 10)
        )

    def forward(self, X): return self.net(X)
    def loss(self, y_hat, y): return self.loss_fn(y_hat, y)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.config.weight_decay)