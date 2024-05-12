# Fusion Module

## For fine-tuing
### dataset
The top 10 queries and their corresponding candidate cases in the training dataset have a total of 1K of data.

### model
Qwen1.5-7B-Chat

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0
- mixed_precision_training: Native AMP

### Framework versions

- PEFT 0.10.0
- Transformers 4.39.3
- Pytorch 2.2.0+cu118
- Datasets 2.18.0
- Tokenizers 0.15.2

## checkepoint
We provide a fine-tuned checkpoint(https://drive.google.com/drive/folders/1vknPVt4CnUAFOCkLRbDOKP6OdTs15BOh?usp=sharing).



