# Multi-label Model for Legislators’ Questioning Classification 
This repository is dedicated to the 2023 ADL final project (Group 7), collaboratively developed by Peng-Ting Kuo (R10322029) and Yu-Hung Sun (R10343017).  
The Graph folder contains all of the graphs we used in the report and presentation.  
`run.sh` and `plot.sh` is for one-step replication.  

## Detailed instructions are provided for replicating the training process and subsequent evaluation.
### First, download the dataset (Due to the confidential nature of the dataset, we are unable to provide a download link.)
```
Path/to/Your/Dataset
```

### Subsequently, execute predict.py to generate the results and the evaluation.
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
|TAIWAN-LLM + instruction tuning   | 67.78 | 56.79 | 54.66 | 52.97 | 51.55 |
|TAIWAN-LLM + few_shot| 43.64 | 39.04 | 49.86 | 46.22 | 35.71 |
|TAIWAN-LLM + zero_shot| 42.29 | 37.96 | 50.03 | 45.97 | 34.09 |

## Additional Information
To gain a detailed understanding of our methodology, we invite you to explore our Jupyter notebook files. These notebooks provide a comprehensive guide to the steps and processes we followed:  
  1. Longformer Pre-Trained Model Experiment: For insights into our work with the Longformer pre-trained model, refer to the notebook `ADL_Final_Project_Longformer.ipynb`, authored by Peng-Ting Kuo. This notebook offers an in-depth look at our experiment and findings.  
  2. Taiwan-LLAMA Related Experiment: If you are keen on learning about our experiment related to Taiwan-LLAMA, please view the notebook `4class_TWLLM.ipynb`, crafted by Yu-Hung Sun. It provides a thorough exploration of our approach and results in this report.  

## Demo
We have hosted a demo interface on HuggingFace. You can experiment with this model at [HuggingFace Spaces](https://huggingface.co/spaces/dean22029/multi-lable_classification).  
To launch this interface, run `app.py`.
