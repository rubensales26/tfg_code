from engine.hp_config import ModelHPConfig
from engine.models import BaseModel, KMnistCNN, KMnistMLP
from engine.data import KMnistDataModule
from engine.trainer import StandardTrainer

def _general_objective(sampled_hyperparams: dict, model_class: type[BaseModel], budget: int = 10) -> float:
    """
    Notice the new 'budget' parameter! It defaults to 10 for standard tuning,
    but SHA will override it dynamically (e.g., 1 epoch, 3 epochs, 9 epochs).
    """
    config = ModelHPConfig(
        learning_rate=sampled_hyperparams["learning_rate"],
        hidden_units=int(sampled_hyperparams["hidden_units"]),
        dropout_rate=sampled_hyperparams["dropout_rate"],
        batch_size=int(sampled_hyperparams["batch_size"]),
        weight_decay=1e-4  
    )
    
    data = KMnistDataModule(batch_size=config.batch_size)
    model = model_class(config) 
    
    # We pass the dynamically requested budget here!
    trainer = StandardTrainer(max_epochs=budget, verbose=False) 
    result = trainer.fit(model, data)
    
    return result['final_val_loss']

# Update the public wrappers to pass the budget through
def cnn_objective(sampled_hyperparams: dict, budget: int = 10) -> float:
    return _general_objective(sampled_hyperparams, KMnistCNN, budget)

def mlp_objective(sampled_hyperparams: dict, budget: int = 10) -> float:
    return _general_objective(sampled_hyperparams, KMnistMLP, budget)