import argparse
import logging
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--data_file", 
        type=str, 
        default=None, 
        help="A json file containing the training data.")
    parser.add_argument(
        "--record_files", 
        type=list, 
        default=['Data/training_record.json', 'Data/training_record_base.json', 'Data/training_record_final_model.json', 'Data/training_record_final_backward_model.json'], 
        help="A list of json file containing the record data.")
    parser.add_argument(
        "--model_name_or_path",
        type="dean22029/Taiwan_Legislator_multilabel_classification",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
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
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    # All dataset overview
    df = pd.read_excel(args.data_file)
    
    def plot_category_distribution(df):
        category_counts = df['output'].apply(lambda x: pd.Series(x)).sum()
        print(f"Total Data: {len(df)}")

        category_labels = {0: 'Information', 1: 'Justification', 2: 'Change', 3: 'Sanction'}
        category_counts.index = [category_labels.get(x, x) for x in category_counts.index]
        category_ratios = category_counts / category_counts.sum()

        # Color
        morandi_colors = ['#b2a29d', '#857e7b', '#a6998a', '#7a6f66']
        colors = ['skyblue', 'salmon', 'limegreen', 'gold']


        plt.style.use('ggplot')
        category_ratios.plot(kind='bar', color=morandi_colors, fontsize=12)
        plt.title('Category Distribution', fontsize=14)
        plt.xlabel('Category', fontsize=12)
        plt.ylabel('Ratio', fontsize=10)

        y_max = category_ratios.max()
        plt.yticks(np.linspace(0, y_max, num=5))
        plt.xticks(rotation=30)
        plt.tight_layout()

        plt.savefig(os.path.join(args.output_dir, "Category_distribution.png"))
        return category_ratios
    
    def plot_text_length_distribution(df):
        text_lengths = df['instruction'].str.len()
        
        plt.figure(figsize=(10, 6))
        max_length = text_lengths.quantile(0.99) # Cut the outliers
        plt.hist(text_lengths[text_lengths <= max_length], bins=100, color='#a6998a', alpha=0.7, edgecolor='black')

        plt.grid(color='grey', linestyle='-', linewidth=0.25, alpha=0.5)

        plt.title('Text Length Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Text Length', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        plt.xticks(fontsize=10, rotation=30)
        plt.yticks(fontsize=10)

        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "Text_length_distribution.png"))

    def plot_training_record(json_files, key):
        plt.figure(figsize=(12, 7))

        colors = ['#708090', '#556b2f', '#2f4f4f', '#bc8f8f', '#b2a29d', '#857e7b', '#a6998a', '#7a6f66']
        # It depends on how many combination exist
        labels = [
            'Max Length 1024',
            'Max 1024, Tokenizer BERT',
            'Max Length 512',
            'Max Length 512, Backward Truncation'] 

        for i, file_path in enumerate(json_files):
            with open(file_path, 'r') as file:
                data = json.load(file)

                if key in data:
                    values = data[key]
                    if isinstance(values, list):
                        plt.plot(values, label=labels[i], color=colors[i % len(colors)])
                    else:
                        print(f"Key '{key}' in file {file_path} is not a list.")
                else:
                    print(f"Key '{key}' not found in file {file_path}.")

        title = ' '.join(key.split('_')).title()
        plt.title(f'{title}', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel(f'{title}', fontsize=14)

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

        plt.tight_layout()
        plt.show()
    
    # Matrices plot
    def plot_matrices_evaluation():
        accuracy = [0.8957, 0.8934, 0.8942, 0.9550]
        precision = [0.8534, 0.8939, 0.8583, 0.8165]
        recall = [0.8571, 0.9084, 0.9229, 0.7063]
        f1_score = [0.8553, 0.9011, 0.8894, 0.7574]

        labels = ['Information', 'Justification', 'Change', 'Sanction']
        x = np.arange(len(labels))
        width = 0.1
        fig, ax = plt.subplots(figsize=(12, 6))

        rects1 = ax.bar(x - width*1.5, accuracy, width, label='Accuracy', color='#1f77b4')
        rects2 = ax.bar(x - width/2, precision, width, label='Precision', color='#ff7f0e')
        rects3 = ax.bar(x + width/2, recall, width, label='Recall', color='#2ca02c')
        rects4 = ax.bar(x + width*1.5, f1_score, width, label='F1 Score', color='#d62728')

        ax.set_ylabel('Score')
        ax.set_title('Evaluation Metrics by Label')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=True, ncol=4)

        plt.savefig(os.path.join(args.output_dir, "Matrices_Evaluation.png"))

    
    # Category plot
    category_ratios = plot_category_distribution(df)
    
    plot_text_length_distribution(df)
    
    # Training records
    training_record = args.record_files
    plot_training_record(training_record, 'hamming_score')
    plot_training_record(training_record, 'hamming_loss')
    
    # Labels' evaluation
    plot_matrices_evaluation()

    
if __name__ == "__main__":
    main()