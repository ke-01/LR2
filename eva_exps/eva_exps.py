import torch
import json
import os
from dataset import SimilarLawTestDataSet
from transformers import AutoModel, AutoTokenizer
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
import argparse
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoModelForCausalLM
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
os.environ["CUDA_VISIBLE_DEVICES"] = '3'



parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--query_path', type=str,  default='../Lecard-main/query/summary_query_all.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../Lecard-main/candidates/eva_exps_laws_law', help='candidates_path')
parser.add_argument('--law_exp_path', type=str,  default='./Lecard_law_exps.json', help='law_exp_path')
parser.add_argument('--case_exp_path', type=str,  default='./Lecard_case_exps.json', help='case_exp_path')
parser.add_argument('--save_path', type=str,  default='./Lecard_fusion_bert_qwen_res.json', help='save path')
parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')
parser.add_argument('--eva_type', type=str, default="flu",choices= ['flu','rel','complete','logic'], help='dataset choice')
parser.add_argument('--gpus', type=str, default="0", help='dataset choice')

args = parser.parse_args()

device=torch.device('cuda:'+args.gpus) 

# elam test
if args.data_type == 'elam':
    args.candidates_path = '../elam_data/elam_candidates/eva_exps_laws_law'
    args.query_path = '../elam_data/elam_summary_query_all.json'
    args.law_exp_path = './elam_case_exps.json'
    args.case_exp_path = './elam_law_exps.json'

if args.eva_type=='flu':
    with open("template_flu.txt", "r") as file:
        template = file.read()
elif args.eva_type=='rel':
    with open("template_rel.txt", "r") as file:
        template = file.read()
elif args.eva_type=='logic':
    with open("template_logic.txt", "r") as file:
        template = file.read()
elif args.eva_type=='complete':
    with open("template_complete.txt", "r") as file:
        template = file.read()

model_path = "../DISC-Law"

def get_model():
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
    )
    model.generation_config = GenerationConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, use_fast=False, trust_remote_code=True,
    )
    return tokenizer, model

tokenizer, model = get_model()

dataset = SimilarLawTestDataSet(args.candidates_path, args.query_path, tokenizer,  args.eva_type,args.law_exp_path,args.case_exp_path,template)
test_data_loader = DataLoader(dataset, batch_size=1,collate_fn=select_collate_test)

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch


def test():
    with torch.no_grad():
        batch_iterator = tqdm(test_data_loader, desc='testing...',ncols=100)
        ans_dict = {}
        
        for step, batch in enumerate(batch_iterator):
            batch = _move_to_device(batch, device)
            query_id = batch['q_id']
            doc_id = batch['d_id']

            x=batch['x'].to(device)
            logits = model(
                input_ids=x.input_ids,
                ).logits[:,-1].flatten()
            candidate_logits = [logits[tokenizer(label).input_ids[-1]] for label in ["0", "1", "2", "3","4", "5"]]

            candidate_logits = torch.tensor(candidate_logits).to(torch.float32)
            probs = (
                torch.nn.functional.softmax(
                    candidate_logits,
                    dim=0,
                ).detach().cpu().numpy()
            )

            weights = [0, 1, 2, 3, 4, 5]  # weights
            
            score=sum(weight * value for weight, value in zip(weights, probs))
            
            
            if query_id not in ans_dict.keys():
                ans_dict[query_id] = {}
            ans_dict[query_id][doc_id] = score.item()

        with open(args.save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))

    print("test finish")

if __name__ == '__main__':
    test()