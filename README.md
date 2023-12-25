# Multi-label Model for Legislators’ Questioning Classification 
This repository is dedicated to the 2023 ADL final project (Group 7), collaboratively developed by Peng-Ting Kuo (R10322029) and Yu-Hung Sun (R10343017).

## Detailed instructions are provided for replicating the training process and subsequent evaluation.
### First, download the dataset (Due to the confidential nature of the dataset, we are unable to provide a download link.)

### Subsequently, execute predict.py to generate the results.
```
!bash run.sh data/test.json data/test_answer.json
```

## Comprehensive step-by-step guide for training the model.
Please note that our model was trained on Google Colab, which may lead to slight variations when run on local machines. 

### Step 1 Training
Use `!python3 train.py` for training
```
!python3 train.py \
    --model_name_or_path ValkyriaLenneth/longformer_zh \
    --tokenizer_name_or_path tokenizer_name_or_path \
    --train_file data/train.json \
    --validation_file data/validation.json \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --max_length 512 \
    --learning_rate 2e-4 \
    --num_label 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 10 \
    --output_dir Your/Model/Path
```

Upon completion of the training, the best model will be saved to a specified path in the given Google Drive path.

### Step 2 Prediction and Evaluation
Use `!python3 predict.py` in jupyter notebook or `bash run.sh` for generating prediction result.
```
!python3 predict.py \
    --predict_file data/test.json \
    --reference_file data/test_answer.json
```

### Step 3 Plot
For visualizing data distribution and training logs used in our report and presentations, use `!python3 plot.py` in a Jupyter Notebook or employ the provided bash script.  
```
!bash plot.sh Your/Data/Path Your/Record/Path Output/Path
```

## Result
"We experimented with various pre-trained models and large language models (LLMs) during the training process. The following are the outcomes of these experiments:
| Model | Hamming Score | Accuracy | Precision | Recall | F1 Score |
|:-------:|:---------------:|:----------:|:-----------:|:--------:|:----------:|
| Longformer           | **83.89** | **72.51** | 86.77 | **88.65** | **87.70**  |
| Longformer + no punc | 82.15 | 71.88 | **87.95** | 85.56 | 86.74 |
| RoBERTa              | 73.55 | 60.70 | 81.28 | 77.40 | 79.34 |
| RoBERTa-large        | 74.28 | 61.81 | 70.56 | 83.97 | 77.27 |
|TAIWAN-LLM            | 67.78 | 56.79 | 57.13 | 45.97 | 51.55 |


  
## Demo
We have hosted a demo interface on HuggingFace. You can experiment with this model at <https://huggingface.co/spaces/dean22029/multi-lable_classification>.  
To launch this interface, run `app.py`.
