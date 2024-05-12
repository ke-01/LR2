import re
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader
from utils import *

class SimilarLawTestDataSet(Dataset):
    def __init__(self, candidates_path, query_path,  tokenizer,data_type):
        super(SimilarLawTestDataSet, self).__init__()
        self.candidates_path = candidates_path
        self.query_path = query_path
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
                law_pres,law_rules,law_crimes = get_doc(self.candidates_path, k, v,self.data_type)
                data_pair_list.append((k, v, query, law_pres,law_rules,law_crimes))
        return data_pair_list  # query, doc

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_id, d_id, q, law_pres,law_rules,law_crimes = self.data_pair_list[item]

        pxx=[]
        rxx=[]
        for i in range(len(law_pres)): 
            px= "给定一个案例，一组谓词，推断出案例满足的谓词。\n------\n案例："+q+"\n谓词："+law_pres[i]+"\n###\n只输出满足的谓词，不要分析。案例满足谓词"
            # 跑
            rx="\n一阶逻辑公式的语法定义如下：\n1. expr1 和 expr2 的逻辑合取：expr1 * expr2。\n2. expr1 和 expr2 的逻辑析取：expr1 + expr2。\n3. expr1 的逻辑否定：~expr1。\n4. expr1 推出 expr2：expr1 = expr2。\n前提："+law_rules[i]+"结论："+law_crimes[i]+"\n基于案例满足的谓词与前提，判断是否可以推出结论，请回答是/否。回答："

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": px}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt")
            pxx.append(model_inputs)
            rxx.append(rx)


        return {'q_id':q_id,
                'd_id':d_id,
                'pxx':pxx,
                'rxx':rxx}


def get_doc(candidates_path, text1_idx, text2_idx,data_type):
    file_path = os.path.join(candidates_path, text1_idx, text2_idx+'.json')
    doc = ''
    with open(file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        law_pres=js_dict['predicate']
        law_rules=js_dict['rules']
        law_crimes=js_dict['crimes']

    return law_pres,law_rules,law_crimes


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




