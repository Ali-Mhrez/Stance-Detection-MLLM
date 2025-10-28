import torch
import numpy as np
from tqdm.auto import tqdm
from sklearn.metrics import f1_score

class TrainingManager:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.model.to(device)

    def train_epoch(self, dataloader):
        total_loss, accuracy = 0, 0
        progress_bar = tqdm(range(len(dataloader)))
        
        self.model.train()
        for batch in dataloader:
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
            
            total_loss += loss.item()
            accuracy += (outputs.logits.argmax(dim=1) == batch["labels"]).sum().item()
        return total_loss / len(dataloader), accuracy / len(dataloader.dataset)
    
    def evaluate(self, dataloader):
        total_loss, accuracy = 0, 0
        all_preds = []
        
        self.model.eval()
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
                loss = outputs.loss
                
            total_loss += loss.item()
            predictions = outputs.logits.argmax(dim=1)
            accuracy += (predictions == batch["labels"]).sum().item()
            all_preds = np.concatenate((all_preds, predictions.cpu().numpy()))
            
        f1scores= f1_score(dataloader.dataset.labels, all_preds, average=None)
        mf1score = f1_score(dataloader.dataset.labels, all_preds, average='macro')
        return total_loss / len(dataloader), accuracy / (len(dataloader.dataset)), f1scores, mf1score