from dataclasses import dataclass

@dataclass
class ModelHPConfig:
    """Hyperparameters structure"""
    learning_rate: float
    batch_size: int
    dropout_rate: float
    hidden_units: int
    weight_decay: float
    
    # We can even set a default value for things that rarely change
    max_epochs: int = 5