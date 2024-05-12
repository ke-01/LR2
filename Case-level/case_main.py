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
import time
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--query_path', type=str,  default='../Lecard-main/query/test_summary_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../Lecard-main/candidates/test_summary_laws_law', help='candidates_path')
parser.add_argument('--save_path', type=str,  default='./Lecard_case_level.json', help='save path')
parser.add_argument('--exp_save_path', type=str,  default='./Lecard_case_exps.json', help='save path')

parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')
parser.add_argument('--gpus', type=str, default="0", help='dataset choice')

args = parser.parse_args()

device=torch.device('cuda:'+args.gpus) 

# elam test
if args.data_type == 'elam':
    args.candidates_path = '../elam_data/elam_candidates/eva_exps_laws_law'
    args.query_path = '../elam_data/elam_summary_query_all.json'
    args.save_path = './elam_case_level.json'
    args.exp_save_path = './elam_case_eva_exp.json'
    
    print('test elam')

mode_name_or_path = "Qwen/Qwen1.5-7B-Chat"

def get_model():
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16,  device_map="auto").eval()
  
    return tokenizer, model

tokenizer, model = get_model()
yes_id = tokenizer.encode("是", add_special_tokens=False)[0]
no_id = tokenizer.encode("否", add_special_tokens=False)[0]

dataset = SimilarLawTestDataSet(args.candidates_path, args.query_path, tokenizer,  args.data_type)
test_data_loader = DataLoader(dataset, batch_size=1,collate_fn=select_collate_test)

def _move_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
    return batch
model.to(device)


def test():
    # model.eval()
    with torch.no_grad():
        batch_iterator = tqdm(test_data_loader, desc='testing...',ncols=100)
        ans_dict = {}
        exp_dict = {}
        start_time = time.time()
        for step, batch in enumerate(batch_iterator):
            batch = _move_to_device(batch, device)
            query_id = batch['q_id']
            doc_id = batch['d_id']

            x=batch['x'].to(device)
            
            output = model.generate(
                x.input_ids,
                return_dict_in_generate=True,
                output_scores=True, 
                output_logits=True,
                max_new_tokens=512
            )
        
            logits=output['logits'][0]
            generated_ids=output['sequences']
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(x.input_ids, generated_ids)
            ]
            response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            
            yes_and_no_logits = logits[:, [yes_id, no_id]]
            yes_and_no_logits = torch.softmax(yes_and_no_logits, dim=1)
            score = yes_and_no_logits[:, 0]
            
            if query_id not in ans_dict.keys():
                ans_dict[query_id] = {}
                exp_dict[query_id] ={}
            ans_dict[query_id][doc_id] = score.item()
            exp_dict[query_id][doc_id] = "匹配得分："+str(score.item())+"\t"+response
            

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]
        end_time = time.time()
        print('total time:{} sec'.format(end_time - start_time))
        with open(args.save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))
        with open(args.exp_save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(exp_dict, ensure_ascii=False))
    print("test finish")

if __name__ == '__main__':
    test()