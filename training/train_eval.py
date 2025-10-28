import torch
import numpy as np
from tqdm.auto import tqdm

class TrainingManager:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)
            
    def train_loop(self, train_dataloader):
        train_loss, train_accuracy = 0, 0
        progress_bar = tqdm(range(len(train_dataloader)))
        
        self.model.train()
        for batch in train_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # update progress bar
            progress_bar.update(1)
            
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            train_loss += loss.item() * len(batch["labels"])
            train_accuracy += torch.sum(predictions == batch["labels"])
            
        return train_loss, train_accuracy
    
    def eval_loop(self, eval_dataloader):
        eval_loss, eval_accuracy = 0, 0
        all_logits, all_preds = [], []
        
        self.model.eval()
        for batch in eval_dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            eval_loss += loss.item() * len(batch["labels"])
            eval_accuracy += torch.sum(predictions == batch["labels"])

            # Collect logits and predictions for further analysis
            all_logits.append(logits.cpu().numpy())
            all_preds = np.concatenate((all_preds, predictions.cpu().numpy()))
            
        all_logits = np.concatenate(all_logits)
        return eval_loss, eval_accuracy, all_logits, all_preds