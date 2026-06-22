import torch
from abc import ABC, abstractmethod
from .models import BaseModel
from .data import DataModule


class BaseTrainer(ABC):
    """The abstract contract for any training engine in the library."""
    
    def __init__(self, max_epochs: int, verbose: bool = True):
        self.max_epochs = max_epochs
        self.verbose = verbose
        
        # Track training histories cleanly across epochs
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        
        # Hardware Detection: Handled at the base level so all subclasses inherit it!
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

    @abstractmethod
    def fit(self, model: BaseModel, data: DataModule) -> dict:
        """
        Executes the optimization sequence.
        MUST return a dictionary containing 'final_val_loss' and 'final_val_acc'.
        """
        pass

    def plot_history(self):
        """Generates a professional plot of loss and accuracy."""
        if not self.history['train_loss']:
            print("Warning: No training history found. Train the model first!")
            return
    
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss Plot
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy Plot
        ax2.plot(epochs, self.history['val_acc'], label='Val Acc', color='green')
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        # Save the graph for your thesis!
        plt.savefig('training_results.png')
        plt.show()


class StandardTrainer(BaseTrainer):
    """A universal trainer for any BaseModel architecture."""

    def fit(self, model: BaseModel, data: DataModule) -> dict:
        # Transfer model parameters to the designated accelerator device
        model = model.to(self.device)
        
        train_loader = data.train_dataloader()
        val_loader = data.val_dataloader()
        optimizer = model.configure_optimizers()

        if self.verbose:
            print(f"Initializing CNN training on {self.device} for {self.max_epochs} epochs...")

        for epoch in range(self.max_epochs):
            # --- 1. Training Step ---
            model.train()  
            train_loss_sum = 0
            
            for batch in train_loader:
                X, y = batch[0].to(self.device), batch[1].to(self.device)
                
                optimizer.zero_grad()
                loss = model.training_step((X, y))
                loss.backward()
                optimizer.step()
                
                train_loss_sum += loss.item()

            avg_train_loss = train_loss_sum / len(train_loader)

            # --- 2. Validation Step ---
            model.eval()  
            val_loss_sum = 0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    X, y = batch[0].to(self.device), batch[1].to(self.device)
                    
                    # Track validation loss
                    loss = model.validation_step((X, y))
                    val_loss_sum += loss.item()
                    
                    # Compute categorical prediction accuracy
                    logits = model(X)
                    predictions = torch.argmax(logits, dim=1)
                    correct_predictions += (predictions == y).sum().item()
                    total_samples += y.size(0)

            avg_val_loss = val_loss_sum / max(len(val_loader), 1)
            val_accuracy = correct_predictions / total_samples

            # --- 3. Logging & Bookkeeping ---
            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_accuracy)
            
            if self.verbose:
                print(f"Epoch {epoch+1:02d}/{self.max_epochs} | "
                      f"Train Loss: {avg_train_loss:.4f} | "
                      f"Val Loss: {avg_val_loss:.4f} | "
                      f"Val Acc: {val_accuracy * 100:.2f}%")

        # 4. HPO Interface Output
        return {
            'final_val_loss': self.history['val_loss'][-1],
            'final_val_acc': self.history['val_acc'][-1]
        }