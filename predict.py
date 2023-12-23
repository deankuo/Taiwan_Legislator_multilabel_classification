import argparse
import logging
import re
import numpy as np
import pandas as pd
import warnings
import torch
from torch import cuda
from tqdm import tqdm
import datasets
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (
    BertTokenizer,
    LongformerTokenizer,
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
        "--reference_file", 
        type=str, 
        default=None, 
        help="A json file containing the training data.")
    parser.add_argument(
        "--predict_file", 
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
        type="dean22029/Taiwan_Legislator_multilabel_classification",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--num_label", 
        type=int, 
        default=4, 
        help="Total number of label.")
    parser.add_argument(
        "--output_dir", 
        type=str,
        default=None, 
        help="Where to store the final model.")
    parser.add_argument(
        "--ignore_mismatched_sizes", 
        action="store_true", 
        help="Whether or not to enable to load a pretrained model whose head dimensions are different.")
    args = parser.parse_args()

    return args

# Hamming score
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

# Truncate
def truncate_texts(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: x[-512:] if isinstance(x, str) else x)
    return df

# Punctuation
def remove_punctuation(text):
    punctuation = r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+'
    return re.sub(punctuation, ' ', text)

def main():
    args = parse_args()
    device = "cuda" if cuda.is_available() else "cpu"
    
    # Model
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = LongformerForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=args.num_label,
        ignore_mismatched_sizes=False,
    )
    
    model.to(device)
    
    # Predict
    def predict(test_file, tokenizer, model, device):
        test_df = pd.read_json(test_file, lines=True)
        test_df = truncate_texts(test_df, 'instruction')
        test_df['instruction'] = test_df['instruction'].apply(remove_punctuation)
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

    predictions = predict(args.predict_file, tokenizer, model, device)
    
    def evaluate_predictions(answer_file, predict_file):
        answer_df = pd.read_json(answer_file, lines=True)

        answer_df = answer_df.sort_values(by='id').reset_index(drop=True)
        pred_df = predict_file.sort_values(by='id').reset_index(drop=True)

        y_true = np.array(answer_df['output'].tolist())
        y_pred = np.array(pred_df['prediction'].tolist())

        # Metrics
        val_hamming_score = hamming_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        f1 = f1_score(y_true, y_pred, average='micro')

        print(f'Hamming Score: {val_hamming_score:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
    
    evaluate_predictions(args.reference_file, predictions)
    
    def calculate_class_wise_metrics(answer_file, pred_file):
        answer_df = pd.read_json(answer_file, lines=True)
        answer_df = answer_df.sort_values(by='id').reset_index(drop=True)
        pred_df = pred_file.sort_values(by='id').reset_index(drop=True)

        y_true = np.array(answer_df['output'].tolist())
        y_pred = np.array(pred_df['prediction'].tolist())

        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': []
        }

        for i in range(y_true.shape[1]):
            accuracy = accuracy_score(y_true[:, i], y_pred[:, i])
            precision = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)

            metrics['accuracy'].append(accuracy)
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['f1_score'].append(f1)

        return metrics

    class_wise_metrics = calculate_class_wise_metrics(args.reference_file, predictions)

    print("Label Metrics: ")
    for metric, values in class_wise_metrics.items():
        print(f"{metric}: {values}")


if __name__ == "__main__":
    main()
