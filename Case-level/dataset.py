import re
import torch
from torch.utils.data import Dataset
import json
import os
from torch.utils.data import DataLoader

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
            test_path='../dataset/elam_data/elam_for_eva_exps_test.json'
        elif self.data_type=='Lecard':
            test_path='/new_disk2/kepu_zhang/leagal_LLM/qwen/qwen/NS-LCR-main/Lecard-main/prediction/Lecard_for_eva_exps_test.json'
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
                data_pair_list.append((k, v, query, doc))
        return data_pair_list  # query, doc

    def __len__(self):
        return len(self.data_pair_list)

    def __getitem__(self, item):
        q_id, d_id, q, d = self.data_pair_list[item]

        x = "案例："+d+ "查询："+q+"案例是否与查询相关？请先回答是/否，并在之后附上原因。回答："
        
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
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
        if data_type=='elam':
            doc += "".join(js_dict['doc'])
        elif data_type=='Lecard':
            doc += js_dict['doc']

    return doc


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



