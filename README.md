# Implementation of LR^2
This is the official implementation of the paper "LR^2: a Model-Agnostic Logical Reasoning Framework for Legal
Case Retrieval" based on PyTorch.

Here we mainly provide the implementation of LR^2-G, for LR^2-D and the neural module, you can refer to "https://github.com/ke-01/NS-LCR".

## Get LR^2 results in 3 steps: 
1. Get the results of the neural model through training. 

2. Get the results of the law-level module and case-level module. 

3. Use the fusion module to get the final results.

## Dataset
The Dataset details is shown in dataset file

## For law-level and case-level
You can get the law-level and case-level results with the following instructions.

```bash
python Law-level/law_main.py  
python Law-level/law_main.py  --data_type elam
python Case-level/case_main.py
python Case-level/case_main.py --data_type elam
```

## For fusion
### Get fine-tuned LLM
Refer to Fusion folder.

### Fusion
You can get the LR^2-G results with the following instructions.

```bash
python Fuison/fusion_main.py --data_type Lecard --base_model_path ./base_model_res/Lecard_bert_res.json
python Fuison/fusion_main.py --data_type Lecard --base_model_path ./base_model_res/Lecard_bertpli_res.json
python Fuison/fusion_main.py --data_type Lecard --base_model_path ./base_model_res/Lecard_lawformer_res.json
python Fuison/fusion_main.py --data_type Lecard --base_model_path ./base_model_res/Lecard_shaobert_res.json
python Fuison/fusion_main.py --data_type ELAM --base_model_path ./base_model_res/elam_bert_res.json
python Fuison/fusion_main.py --data_type ELAM --base_model_path ./base_model_res/elam_bertpli_res.json
python Fuison/fusion_main.py --data_type ELAM --base_model_path ./base_model_res/elam_lawformer_res.json
python Fuison/fusion_main.py --data_type ELAM --base_model_path ./base_model_res/elam_shaobert_res.json
```

## Evaluate explanation
You can evaluate explanations with the following instructions.

```bash
python eva_exps/eva_exps.py --data_type Lecard --eva_type logic
python eva_exps/eva_exps.py --data_type Lecard --eva_type rel
python eva_exps/eva_exps.py --data_type Lecard --eva_type flu
python eva_exps/eva_exps.py --data_type Lecard --eva_type complete
python eva_exps/eva_exps.py --data_type ELAM --eva_type logic
python eva_exps/eva_exps.py --data_type ELAM --eva_type rel
python eva_exps/eva_exps.py --data_type ELAM --eva_type flu
python eva_exps/eva_exps.py --data_type ELAM --eva_type complete
```

## Environments
We conducted the experiments based on the following environments:
* CUDA Version: 11.4
* torch version: 2.2.0
* OS: Ubuntu 18.04.5 LTS
* GPU: NVIDIA Geforce RTX A6000
* CPU: Intel(R) Xeon(R) Silver 4214 CPU @ 2.20GHz