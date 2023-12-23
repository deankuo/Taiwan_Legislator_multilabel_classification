import argparse
import json
import logging
import os
import copy
import numpy as np
import pandas as pd
import warnings
import torch
from torch.utils.data import Dataset, DataLoader
from torch import cuda
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import transformers
from transformers import (
    BertTokenizer,
    LongformerForSequenceClassification,
)

from transformers.utils.versions import require_version

logging.basicConfig(level=logging.ERROR)
device = "cuda" if cuda.is_available() else "cpu"
warnings.simplefilter('ignore')
require_version("torch", "To fix: pip install -r requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--train_file", 
        type=str, 
        default=None, 
        help="A json file containing the training data.")
    parser.add_argument(
        "--validation_file", 
        type=str, 
        default=None, 
        help="A json file containing the validation data.")
    parser.add_argument(
        "--test_file", 
        type=str, 
        default=None, 
        help="A json file containing the test data.")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_length` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--num_label", 
        type=int, 
        default=4, 
        help="Total number of label.")
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use.")
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=10, 
        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--predict", 
        action="store_true", 
        help="Whether to do prediction or not.")
    parser.add_argument(
        "--predict_file", 
        type=str, 
        help="Path of prediction output.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_warmup_steps", 
        type=int, 
        default=0, 
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None, 
        help="Where to store the final model.")
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training.")
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--ignore_mismatched_sizes", 
        action="store_true", 
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.")
    args = parser.parse_args()

    return args

class MultiLabelDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.instruction
        self.targets = self.data.output
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Hamming Score
def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
    acc_list = []
    for i in range(y_true.shape[0]):
        set_true = set( np.where(y_true[i])[0] )
        set_pred = set( np.where(y_pred[i])[0] )
        tmp_a = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp_a = 1
        else:
            tmp_a = len(set_true.intersection(set_pred))/\
                    float(len(set_true.union(set_pred)) )
        acc_list.append(tmp_a)
    return np.mean(acc_list)

# Loss Function
def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

# Test only use the last 512 tokens
def truncate_texts(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: x[-512: ] if isinstance(x, str) else x)
    return df

def main():
    args = parse_args()
    
    # Model
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = LongformerForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_label,
        ignore_mismatched_sizes=args.ignore_mismatched_sizes,
        problem_type='multi_label_classification',
    )
    
    # Dataset
    if args.train_file is not None:
        train_set = pd.read_json(args.train_file, orient='records', lines=True)
        print("TRAIN Dataset: {}".format(train_set))
    if args.validation_file is not None:
        validation_set = pd.read_json(args.validation_file, orient='records', lines=True)
        print("VALID Dataset: {}".format(validation_set))
    if args.predict is not None:
        if args.test_file is None:
            print("Must include a test file in order to do prediction.")
        else:
            test_set = pd.read_json(args.test_file, orient='records', lines=True)
            print("VALID Dataset: {}".format(test_set))
    
    training_set = MultiLabelDataset(train_set, tokenizer, args.max_length)
    validation_set = MultiLabelDataset(validation_set, tokenizer, args.max_length)
    
    train_params = {'batch_size': args.per_device_train_batch_size,
                'shuffle': True,
                'num_workers': 0
                }

    valid_params = {'batch_size': args.per_device_eval_batch_size,
                    'shuffle': False,
                    'num_workers': 0
                }

    # DataLoader
    training_loader = DataLoader(training_set, **train_params)
    validation_loader = DataLoader(validation_set, **valid_params)
    
    model.to(device)
    # Optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    
    # Validation function
    def validation(testing_loader):
        model.eval()
        fin_targets=[]
        fin_outputs=[]
        with torch.no_grad():
            for _, data in tqdm(enumerate(testing_loader, 0)):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                targets = data['targets'].to(device, dtype = torch.float)
                outputs = model(ids, mask)
                fin_targets.extend(targets.cpu().detach().numpy().tolist())
                fin_outputs.extend(torch.sigmoid(outputs.logits).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
    
    # Training record
    training_record = {'best_score': -float('inf'),
                       'hamming_loss': [],
                       'hamming_score': [],
                       'accuracy': [],
                       'precision': [],
                       'recall': []}
    
    def train(epoch, gradient_accumulation_steps=args.gradient_accumulation_steps, train_batch_size=args.per_device_train_batch_size):
        print(f"\n")
        print("*" * 10, "Start training Epoch", str(epoch + 1), "*" * 10)

        best_model = False
        best_model_dict = None
        model.train()
        optimizer.zero_grad()
        total_loss = 0

        for step, data in tqdm(enumerate(training_loader, 0), total=round((train_set.shape[0] / train_batch_size))):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            logits = outputs.logits

            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            loss.backward()

            # Update gradient per gradient_accumulation_steps 
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        avg_train_loss = total_loss / len(training_loader)
        print(f'Epoch {epoch + 1}: Average Training Loss: {avg_train_loss}')

        # Evaluate
        print(f"**********Validation**********")
        outputs, targets = validation(validation_loader)

        final_outputs = np.array(outputs) >= 0.5 # Should also considered training

        print(" " * 10,"Answer", " " * 15, "Prediction")
        for i in range(10):
            print(targets[i], final_outputs[i])

        # Calculate evaluation metrics
        accuracy = accuracy_score(targets, final_outputs)
        precision = precision_score(targets, final_outputs, average='micro')
        recall = recall_score(targets, final_outputs, average='micro')
        val_hamming_loss = metrics.hamming_loss(targets, final_outputs)
        val_hamming_score = hamming_score(np.array(targets), np.array(final_outputs))

        training_record['hamming_loss'].append(val_hamming_loss)
        training_record['hamming_score'].append(val_hamming_score)
        training_record['accuracy'].append(accuracy)
        training_record['precision'].append(precision)
        training_record['recall'].append(recall)

        # Print the evaluation metrics
        print(f"Hamming Score = {val_hamming_score}")
        print(f"Hamming Loss = {val_hamming_loss}")
        print(f'Epoch {epoch + 1}: Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
        print(f"\n")

        # Save the best model
        if val_hamming_score > training_record['best_score']:
            best_model = True
            training_record['best_score'] = val_hamming_score

        if best_model:
            best_model = False
            print(f"***********Save this model!!!***********")
            best_model_dict = copy.deepcopy(model.state_dict())
            torch.save(best_model_dict, 'PyTorch_model.bin')
    
    for epoch in range(args.num_train_epochs):
        train(epoch)
        
    if args.predict and args.test_file is not None:
        def predict(test_file, tokenizer, model, device):
            test_df = pd.read_json(test_file, lines=True)
            test_df = truncate_texts(test_df, 'instruction')
            test_df['input_ids'] = test_df['instruction'].apply(lambda x: tokenizer.encode(x, truncation=True, max_length=512, add_special_tokens=True))

            predictions = []
            predictions_ = []
            model.to(device)
            model.eval()

            with torch.no_grad():
                for _, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
                    input_ids = torch.tensor(row['input_ids']).unsqueeze(0).to(device)
                    outputs = model(input_ids)
                    logits = outputs.logits
                    preds = torch.sigmoid(logits).cpu().numpy() >= 0.5
                    preds_ = torch.sigmoid(logits).cpu().numpy() >= 0.6
                    predictions.append(preds.astype(int).flatten().tolist())
                    predictions_.append(preds_.astype(int).flatten().tolist())

            test_df['prediction'] = predictions
            test_df['prediction_0.6'] = predictions_
            test_df.drop(columns=['input_ids'], inplace=True)
            return test_df
        
        predictions = predict(args.test_file, tokenizer, model, device)

    # Save prediction
    print("***** Predict results *****")
    json_str = predictions.to_json(orient='records', force_ascii=False)

    with open(args.predict_file, 'w', encoding='utf-8') as file:
        file.write(json_str)
    
    # Save model
    if args.output_dir is not None:
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        print("No output directory")
    
    # Save training record
    with open(os.path.join(args.output_dir), "training_record.json") as f:
        json.dump(training_record, f, indent=4)
    
if __name__ == "__main__":
    main()
