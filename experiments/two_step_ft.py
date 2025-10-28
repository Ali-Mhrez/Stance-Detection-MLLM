import os
import torch
import argparse
from torch.optim import AdamW
from torch.utils.data import DataLoader
from ..data.custom_dataset import CustomDataset
from ..utility.utils import write_results, set_seed
from ..training.training_manager import TrainingManager
from ..data.dataset_reader import AraStanceReader, UnifiedFCReader, stance_to_int
from transformers import AutoTokenizer, AutoModelForSequenceClassification

if __name__ == "__main__":
    
    lr = 2e-5
    no_reps = 5
    no_epochs = 4
    batch_size = 32
    sequence_length = 512
    seeds = [42, 43, 44, 45, 46]
    
    parser = argparse.ArgumentParser(description="Run fine tuning experiments.")
    parser.add_argument('--first-dataset', type=str, choices=['arastance', 'unifiedfc'], required=True, help='First fine-tuning dataset.')
    parser.add_argument('--second-dataset', type=str, choices=['arastance', 'unifiedfc'], required=True, help='Second fine-tuning dataset.')
    parser.add_argument('--checkpoint', type=str, choices=['mbert', 'distilmbert', 'xlmroberta', 'mdeberta'],
                        required=True, help='Pre-trained model checkpoint to use.')
    args = parser.parse_args()
    first_dataset = args.first_dataset
    second_dataset = args.second_dataset
    assert first_dataset != second_dataset, "First and second datasets must be different."
    if args.checkpoint == 'mbert':
        checkpoint = 'bert-base-multilingual-cased'
    elif args.checkpoint == 'distilmbert':
        checkpoint = 'distilbert-base-multilingual-cased'
    elif args.checkpoint == 'xlmroberta':
        checkpoint = 'xlm-roberta-base'
    elif args.checkpoint == 'mdeberta':
        checkpoint = 'microsoft/mdeberta-v3-base'
        batch_size = 16  # Decrease batch size for memory constraints
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "raw_data")
    
    print("Loading data and labels...")
    unifiedfc_raw_train = UnifiedFCReader(os.path.join(data_path, "unifiedfc-main/"), folds=['Fold_1', 'Fold_2', 'Fold_3'])
    unifiedfc_raw_val = UnifiedFCReader(os.path.join(data_path, "unifiedfc-main/"), folds=['Fold_4'])
    unifiedfc_raw_test = UnifiedFCReader(os.path.join(data_path, "unifiedfc-main/"), folds=['Fold_5'])

    arastance_raw_train = AraStanceReader(os.path.join(data_path, 'arastance-main/train.jsonl'))
    arastance_raw_val = AraStanceReader(os.path.join(data_path, 'arastance-main/dev.jsonl'))
    arastance_raw_test = AraStanceReader(os.path.join(data_path, 'arastance-main/test.jsonl'))
    print("Data and labels loaded successfully.")
     
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    def tokenize(instance):
        return tokenizer(instance[0], instance[1], truncation=True, padding='max_length', max_length=sequence_length)
    
    print("Tokenizing data...")
    unifiedfc_train_labels = [stance_to_int[stance] for stance in unifiedfc_raw_train.stances]
    unifiedfc_val_labels = [stance_to_int[stance] for stance in unifiedfc_raw_val.stances]
    unifiedfc_test_labels = [stance_to_int[stance] for stance in unifiedfc_raw_test.stances]
    
    unifiedfc_train_claims = list(map(unifiedfc_raw_train.claims.__getitem__, unifiedfc_raw_train.article_claim))
    unifiedfc_val_claims = list(map(unifiedfc_raw_val.claims.__getitem__, unifiedfc_raw_val.article_claim))
    unifiedfc_test_claims = list(map(unifiedfc_raw_test.claims.__getitem__, unifiedfc_raw_test.article_claim))
    
    unifiedfc_train_encodings = tokenize((unifiedfc_train_claims, unifiedfc_raw_train.articles))
    unifiedfc_val_encodings = tokenize((unifiedfc_val_claims, unifiedfc_raw_val.articles))
    unifiedfc_test_encodings = tokenize((unifiedfc_test_claims, unifiedfc_raw_test.articles))
    
    unifiedfc_train_dataset = CustomDataset(unifiedfc_train_encodings, unifiedfc_train_labels)
    unifiedfc_val_dataset = CustomDataset(unifiedfc_val_encodings, unifiedfc_val_labels)
    unifiedfc_test_dataset = CustomDataset(unifiedfc_test_encodings, unifiedfc_test_labels)
    
    arastance_train_labels = [stance_to_int[stance] for stance in arastance_raw_train.stances]
    arastance_val_labels = [stance_to_int[stance] for stance in arastance_raw_val.stances]
    arastance_test_labels = [stance_to_int[stance] for stance in arastance_raw_test.stances]
    
    arastance_train_claims = list(map(arastance_raw_train.claims.__getitem__, arastance_raw_train.article_claim))
    arastance_val_claims = list(map(arastance_raw_val.claims.__getitem__, arastance_raw_val.article_claim))
    arastance_test_claims = list(map(arastance_raw_test.claims.__getitem__, arastance_raw_test.article_claim))
    
    arastance_train_encodings = tokenize((arastance_train_claims, arastance_raw_train.articles))
    arastance_val_encodings = tokenize((arastance_val_claims, arastance_raw_val.articles))
    arastance_test_encodings = tokenize((arastance_test_claims, arastance_raw_test.articles))
    
    arastance_train_dataset = CustomDataset(arastance_train_encodings, arastance_train_labels)
    arastance_val_dataset = CustomDataset(arastance_val_encodings, arastance_val_labels)
    arastance_test_dataset = CustomDataset(arastance_test_encodings, arastance_test_labels)
    print("Tokenization completed.")
    
    print("Starting model fine-tuning...")
    for rep in range(no_reps):
        
        print(f"\n--- Repetition {rep+1} of {no_reps} ---")
        set_seed(seeds[rep])
        
        print("Creating DataLoaders...")
        unifiedfc_train_dataloader = DataLoader(unifiedfc_train_dataset, batch_size=batch_size, shuffle=True)
        unifiedfc_dev_dataloader = DataLoader(unifiedfc_val_dataset, batch_size=batch_size, shuffle=False)
        unifiedfc_test_dataloader = DataLoader(unifiedfc_test_dataset, batch_size=batch_size, shuffle=False)
        
        arastance_train_dataloader = DataLoader(arastance_train_dataset, batch_size=batch_size, shuffle=True)
        arastance_dev_dataloader = DataLoader(arastance_val_dataset, batch_size=batch_size, shuffle=False)
        arastance_test_dataloader = DataLoader(arastance_test_dataset, batch_size=batch_size, shuffle=False)
        print("DataLoaders created successfully.")
        
        print("Initializing model, optimizer, and loss function...")
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        if checkpoint == 'microsoft/mdeberta-v3-base':
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4, torch_dtype="bfloat16")
        else:
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=4, torch_dtype="auto")
        optimizer = AdamW(model.parameters(), lr=lr)
        manager = TrainingManager(model, optimizer, device)
        print("Model, optimizer, and loss function initialized successfully.")
        
        first_train_dataloader = unifiedfc_train_dataloader if first_dataset == 'unifiedfc' else arastance_train_dataloader
        first_dev_dataloader = unifiedfc_dev_dataloader if first_dataset == 'unifiedfc' else arastance_dev_dataloader
        second_train_dataloader = arastance_train_dataloader if second_dataset == 'arastance' else unifiedfc_train_dataloader
        second_dev_dataloader = arastance_dev_dataloader if second_dataset == 'arastance' else unifiedfc_dev_dataloader
                        
        first_eval_accuracies, first_eval_losses = [], []
        second_eval_accuracies, second_eval_losses = [], []
        for epoch in range(no_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            # --- Training Phase ---
            first_train_loss, first_train_accuracy = manager.train_epoch(first_train_dataloader)

            # --- Evaluation Phase ---
            first_eval_loss, first_eval_accuracy, first_eval_f1scores, first_eval_mf1score = manager.evaluate(first_dev_dataloader)
            first_eval_accuracies.append(first_eval_accuracy)
            first_eval_losses.append(first_eval_loss)

            print(f"Train_loss: {first_train_loss:.3f}, Train_acc: {first_train_accuracy:.3f},",
                f"Val_loss: {first_eval_loss:.3f}, Val_accuracy: {first_eval_accuracy:.3f}",
                "\n-------------------------------")
            
        for epoch in range(no_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # --- Training Phase ---
            second_train_loss, second_train_accuracy = manager.train_epoch(second_train_dataloader)

            # --- Evaluation Phase ---
            second_eval_loss, second_eval_accuracy, second_eval_f1scores, second_eval_mf1score = manager.evaluate(second_dev_dataloader)
            second_eval_accuracies.append(second_eval_accuracy)
            second_eval_losses.append(second_eval_loss)

            print(f"Train_loss: {second_train_loss:.3f}, Train_acc: {second_train_accuracy:.3f},",
                f"Val_loss: {second_eval_loss:.3f}, Val_accuracy: {second_eval_accuracy:.3f}",
                "\n-------------------------------")
                
        unifiedfc_test_loss, unifiedfc_test_accuracy, unifiedfc_test_f1scores, unifiedfc_test_mf1score = manager.evaluate(unifiedfc_test_dataloader)
        arastance_test_loss, arastance_test_accuracy, arastance_test_f1scores, arastance_test_mf1score = manager.evaluate(arastance_test_dataloader)
        
        results_path = os.path.join(os.path.dirname(__file__), "..", "results")
        if not os.path.exists(results_path):
            os.makedirs(results_path)
            
        unifiedfc_eval_losses = first_eval_losses if first_dataset == 'unifiedfc' else second_eval_losses
        unifiedfc_eval_accuracies = first_eval_accuracies if first_dataset == 'unifiedfc' else second_eval_accuracies
        unifiedfc_eval_f1scores = first_eval_f1scores if first_dataset == 'unifiedfc' else second_eval_f1scores
        unifiedfc_eval_mf1score = first_eval_mf1score if first_dataset ==   'unifiedfc' else second_eval_mf1score
        
        arastance_eval_losses = first_eval_losses if first_dataset == 'arastance' else second_eval_losses
        arastance_eval_accuracies = first_eval_accuracies if first_dataset == 'arastance' else second_eval_accuracies
        arastance_eval_f1scores = first_eval_f1scores if first_dataset == 'arastance' else second_eval_f1scores
        arastance_eval_mf1score = first_eval_mf1score if first_dataset == 'arastance' else second_eval_mf1score
            
        write_results(
            filename=os.path.join(results_path, f"unifiedfc_results.csv"),
            run_id=rep+1,
            eval_loss=unifiedfc_eval_losses[-1],
            eval_acc=unifiedfc_eval_accuracies[-1],
            eval_f1=unifiedfc_eval_f1scores,
            eval_mf1=unifiedfc_eval_mf1score,
            test_loss=unifiedfc_test_loss,
            test_acc=unifiedfc_test_accuracy,
            test_f1=unifiedfc_test_f1scores,
            test_mf1=unifiedfc_test_mf1score
        )
        
        write_results(
            filename=os.path.join(results_path, f"arastance_results.csv"),
            run_id=rep+1,
            eval_loss=arastance_eval_losses[-1],
            eval_acc=arastance_eval_accuracies[-1],
            eval_f1=arastance_eval_f1scores,
            eval_mf1=arastance_eval_mf1score,
            test_loss=arastance_test_loss,
            test_acc=arastance_test_accuracy,
            test_f1=arastance_test_f1scores,
            test_mf1=arastance_test_mf1score
        )