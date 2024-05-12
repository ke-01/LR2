import re
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader
from utils import *

class SimilarLawTestDataSet(Dataset):
    def __init__(self, candidates_path, query_path,  tokenizer,eva_type,law_exp_path,case_exp_path,template):
        super(SimilarLawTestDataSet, self).__init__()
        self.candidates_path = candidates_path
        self.query_path = query_path
        self.law_exp_path=law_exp_path
        self.case_exp_path=case_exp_path
        self.template=template
        self.eva_type=eva_type
        self.tokenizer = tokenizer
        self.querys = read_query(self.query_path,self.data_type)  # query id : query
        self.test_data = self.read_test_data()
        self.data_pair_list = self.gen_data_pair()

    def read_test_data(self):
        if self.data_type=='elam':
            test_path='../elam_data/elam_for_eva_exps_test.json'
        elif self.data_type=='Lecard':
            test_path='../Lecard-main/prediction/Lecard_for_eva_exps_test.json'
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
                
                pos,law_exp,case_exp = get_exp(self.law_exp_path,self.case_exp_path,k, v)
                
                data_pair_list.append((k, v, query,doc, pos,law_exp,case_exp))
        return data_pair_list  # query, doc

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_id, d_id, q,d, pos,law_exp,case_exp = self.data_pair_list[item]

        x= self.template.replace('{{Query Case}}', q).replace('{{Candidate Case}}', d).replace('{{law-level exp}}', law_exp).replace('{{case-level exp}}', case_exp)
        
        if self.eva_type=='flu':
            instruction="给定一个查询案例和候选案例的匹配解释。\n\n\n你的任务是对匹配解释的流畅性进行评级（0-5,0最低）。\n评估标准：解释应该容易阅读，没有复杂含糊不清的内容。\n评估步骤：\n1.仔细阅读匹配解释。\n2.评估匹配解释是否容易阅读，没有复杂含糊不清的内容。\n3.根据匹配解释的流畅性，给出一个0到5的流畅性评分。\n"
        elif self.eva_type=='rel':
            instruction="给定一个查询案例，一个候选案例以及它们的匹配解释。\n\n\n你的任务是对匹配解释的相关性进行评级（0-5,0最低）。\n评估标准：解释应该和查询和候选案例密切相关，没有其他无关的内容。\n评估步骤：\n1.仔细阅读查询案例，候选案例，匹配解释。\n2.对比查询案例和候选案例，评估匹配解释是否准确反映了查询案例和候选案例之间的关联，没有其他无关的内容。\n3.根据匹配解释的相关性，给出一个0到5的相关性评分。\n"
        elif self.eva_type=='logic':
            instruction="给定一个查询案例，一个候选案例以及它们的匹配解释。\n\n\n你的任务是对匹配解释的逻辑性进行评级（0-5,0最低）。\n评估标准：解释应当具有逻辑性，能够合理解释匹配的原因，没有任何逻辑跳跃，自相矛盾或不合理的内容。\n评估步骤：\n1.仔细阅读查询案例，候选案例，匹配解释。\n2.对比查询案例和候选案例，评估匹配解释是否逻辑上一致，检查是否有逻辑跳跃，自相矛盾或不合理的内容。\n3.根据匹配解释的逻辑一致性，给出一个0到5的逻辑性评分。\n"
        elif self.eva_type=='complete':            
            instruction="给定一个查询案例，一个候选案例以及它们的匹配解释。\n\n\n你的任务是对匹配解释的全面性进行评级（0-5,0最低）。\n评估标准：解释应该涵盖查询和候选案例匹配的关键要素，没有遗漏和匹配相关的重要信息。\n评估步骤：\n1.仔细阅读查询案例，候选案例，匹配解释。\n2.对比查询案例和候选案例，思考它们匹配的关键要素，评估匹配解释是否涵盖了查询和候选案例匹配的关键要素，没有遗漏和匹配相关的重要信息。\n3.根据匹配解释的全面性，给出一个0到5的全面性评分。\n"
        else:
            print('eva type error!')
            exit()
        text=instruction+x
        inputs = self.tokenizer(text, return_tensors='pt')
        
        
        return {'q_id':q_id,
                'd_id':d_id,
                'x':inputs}


def get_doc(candidates_path, text1_idx, text2_idx,data_type):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    doc = ''
    with open(file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        doc=js_dict['doc'] 

    return doc


def get_exp(law_exp_path,case_exp_path,k, v):
    with open(law_exp_path, 'r') as f:
        law_dict = json.load(f)
    law_exp=law_dict[k][v] #
    
    with open(case_exp_path, 'r') as f:
        case_dict = json.load(f)
    case_exp=case_dict[k][v] #
    pos=1
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





