import re
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader
from utils import *
import math

class SimilarLawTestDataSet(Dataset):
    def __init__(self, candidates_path, query_path,  tokenizer,data_type,base_model_path,law_exp_path,case_exp_path,template):
        super(SimilarLawTestDataSet, self).__init__()
        self.candidates_path = candidates_path
        self.query_path = query_path
        
        self.base_model_path=base_model_path
        self.law_exp_path=law_exp_path
        self.case_exp_path=case_exp_path
        self.template=template
        
        self.tokenizer = tokenizer
        self.data_type=data_type
        self.querys = read_query(self.query_path,self.data_type)  # query id : query
        self.test_data = self.read_test_data()
        self.data_pair_list = self.gen_data_pair()

    def read_test_data(self):
        if self.data_type=='elam':
            test_path='../elam_data/elam_test_top50.json'
        elif self.data_type=='Lecard':
            test_path='../Lecard-main/prediction/test_top100.json'
        else:
            print('data error')
            exit()
        with open(test_path, mode='r', encoding='utf-8')as f:
            js_dict = json.load(f)
            for k in js_dict.keys():
                js_dict[k] = [str(v) for v in js_dict[k]]
        return js_dict  # query id, can ids

    def gen_data_pair(self):
        data_pair_list = []
        for k in self.test_data.keys():
            query = self.querys[k]
            for v in self.test_data[k]:
                doc = get_doc(self.candidates_path, k, v,self.data_type)
                pos,law_exp,case_exp = get_exp(self.base_model_path,self.law_exp_path,self.case_exp_path,k, v,self.data_type)
                
                data_pair_list.append((k, v, query,doc, pos,law_exp,case_exp))
        return data_pair_list  # query, doc

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_id, d_id, q,d, pos,law_exp,case_exp = self.data_pair_list[item]

        
        if self.data_type=='Lecard':
            total_num=100
            instructions="给定一个查询案例和一个候选案例。\n\n\n你的任务是根据以下内容对查询案例和候选案例的相关性进行评级（0-3,0最低）。\n"
        elif self.data_type=='elam':
            total_num=50
            instructions="给定一个查询案例和一个候选案例。\n\n\n你的任务是根据以下内容对查询案例和候选案例的相关性进行评级（0-2,0最低）。\n"
        
        r_score=math.exp(-pos / total_num)
        r_score=str(r_score)

        x= self.template.replace('{{Query Case}}', q).replace('{{Candidate Case}}', d).replace('{{relevance score}}', r_score).replace('{{law-level exp}}', law_exp).replace('{{case-level exp}}', case_exp)
        
        messages = [
            {"role": "system", "content": instructions},
            {"role": "user", "content": x}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        
        return {'q_id':q_id,
                'd_id':d_id,
                'x':model_inputs}


def get_doc(candidates_path, text1_idx, text2_idx,data_type):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    doc = ''
    with open(file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        doc=js_dict['doc'] 

    return doc


def get_exp(base_model_path,law_exp_path,case_exp_path,k, v,data_type):
    with open(base_model_path, 'r') as f:
        base_dict = json.load(f)
    pos=base_dict[k].index(int(v))+1 
    with open(law_exp_path, 'r') as f:
        law_dict = json.load(f)
    law_exp=law_dict[k][v] #
    
    with open(case_exp_path, 'r') as f:
        case_dict = json.load(f)
    case_exp=case_dict[k][v] #
    
    return pos,law_exp,case_exp
    

def read_query(query_path,data_type):
    querys = {}
    with open(query_path, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data_type=='elam':
                ridx = data['ridx']
                query = "".join(data['q'])
            elif data_type=='Lecard':
                ridx = data['ridx']
                query = data['q']
            else:
                print("error")
                exit()

            querys[str(ridx)] = query
    return querys




