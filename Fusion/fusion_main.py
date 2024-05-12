import torch
import json
import os
from dataset import SimilarLawTestDataSet
from transformers import AutoModel, AutoTokenizer,AutoConfig, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
import argparse
from argparse import Namespace
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from peft import PeftModel
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description="Help info:")
parser.add_argument('--query_path', type=str,  default='../Lecard-main/query/test_summary_query.json', help='query_path')
parser.add_argument('--candidates_path', type=str,  default='../Lecard-main/candidates/test_summary_laws_law', help='candidates_path')
parser.add_argument('--base_model_path', type=str,  default='./prediction/', help='base_model_path')
parser.add_argument('--law_exp_path', type=str,  default='./Lecard_law_exps.json', help='law_exp_path')
parser.add_argument('--case_exp_path', type=str,  default='./Lecard_case_exps.json', help='case_exp_path')
parser.add_argument('--template_file', type=str,  default='././fusion_template_L.txt', help='case_exp_path')

parser.add_argument('--save_path', type=str,  default='./Lecard_fusion_bert_qwen_res.json', help='save path')
parser.add_argument('--exp_save_path', type=str,  default='./Lecard_fusion_bert_exps.json', help='save path')

parser.add_argument('--data_type', type=str, default="Lecard",choices= ['Lecard','elam'], help='dataset choice')
parser.add_argument('--gpus', type=str, default="0", help='dataset choice')

args = parser.parse_args()

device=torch.device('cuda:'+args.gpus) 

# elam test
if args.data_type == 'elam':
    args.candidates_path = '../elam_data/elam_candidates/test_summary_laws_law'
    args.query_path = '../elam_data/elam_test_summary_query.json'
    args.save_path = './elam_fusion_bert_qwen_res.json'
    args.exp_save_path = './elam_fusion_bert_exps.json'
    
    args.base_model_path = './prediction/'
    args.law_exp_path = './elam_law_exps.json'
    args.case_exp_path = './elam_case_exps.json'
    
    args.template_file='./fusion_template_E.txt'
    print('test elam')


with open(args.template_file, "r") as file:
    template = file.read()

base_path="Qwen/Qwen1.5-7B-Chat"
adapter_path = "../LLMs/Qwen1.5-7B-Chat/lora/train_2024-04-05-10-54-08"

def get_model():
    tokenizer = AutoTokenizer.from_pretrained(base_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=torch.bfloat16)
    
    peft_model = PeftModel.from_pretrained(model, adapter_path,torch_dtype=torch.bfloat16).eval()

    peft_model.to(device)
    
    return tokenizer, peft_model


tokenizer, model = get_model()

zero_id = tokenizer.encode("0", add_special_tokens=False)[0]
one_id = tokenizer.encode("1", add_special_tokens=False)[0]
two_id = tokenizer.encode("2", add_special_tokens=False)[0]
three_id = tokenizer.encode("3", add_special_tokens=False)[0]


dataset = SimilarLawTestDataSet(args.candidates_path, args.query_path, tokenizer,  args.data_type, args.base_model_path,args.law_exp_path,args.case_exp_path,template)
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
        exp_dict = {}
        
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
            
            yes_and_no_logits = logits[:, [zero_id, one_id,two_id,three_id]] 
            
            yes_and_no_logits = torch.softmax(yes_and_no_logits, dim=1)

            weights = [0, 1, 2, 3]  # weight
            score=sum(weight * value for weight, value in zip(weights, yes_and_no_logits[0]))
            
            if query_id not in ans_dict.keys():
                ans_dict[query_id] = {}
                exp_dict[query_id] ={}
            ans_dict[query_id][doc_id] = score.item()
            exp_dict[query_id][doc_id] = "匹配得分："+str(score.item())+"\t"+response
            

        for k in ans_dict.keys():
            ans_dict[k] = sorted(ans_dict[k].items(), key=lambda x: x[1], reverse=True)
        for k in ans_dict.keys():
            ans_dict[k] = [int(did) for did, _ in ans_dict[k]]

        with open(args.save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(ans_dict, ensure_ascii=False))
        with open(args.exp_save_path, mode='w', encoding='utf-8') as f:
            f.write(json.dumps(exp_dict, ensure_ascii=False))
    print("test finish")

if __name__ == '__main__':
    test()