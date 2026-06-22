from engine.hp_config import ModelHPConfig
from engine.models import KMnistCNN
from engine.data import KMnistDataModule
from engine.trainer import StandardTrainer

def main():
    print("=== Single Model Training & Evaluation ===")
    
    # 1. Define the specific configuration you want to visualize
    config = ModelHPConfig(
        learning_rate=0.001,
        batch_size=128,
        dropout_rate=0.5,
        hidden_units=128,
        weight_decay=1e-4,
        max_epochs=10
    )
    
    # 2. Set up the pipeline
    data = KMnistDataModule(batch_size=config.batch_size)
    model = KMnistCNN(config)
    
    # 3. Train with verbosity on so you can see the progress in the console
    trainer = StandardTrainer(max_epochs=config.max_epochs, verbose=True)
    result = trainer.fit(model, data)
    
    print(f"\nTraining Complete! Final Validation Accuracy: {result['final_val_acc']*100:.2f}%")
    
    # 4. Generate and save the professional training graph
    print("Generating training history plot...")
    trainer.plot_history()

if __name__ == "__main__":
    main()