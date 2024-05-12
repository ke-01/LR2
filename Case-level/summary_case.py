from transformers import AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
import torch
import torch.nn.functional as F
from tqdm import tqdm
import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" # the device to load the model onto

mode_name_or_path = "Qwen/Qwen1.5-7B-Chat"

def get_model():
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16,  device_map="auto").eval()
  
    return tokenizer, model

def generate(model,tokenizer,prompt):
    with torch.no_grad():        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)  

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=256
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
    
        return response

tokenizer, model = get_model()

def read_query(outout_file,query_path,data_type):
    querys = {}  
    with open(query_path, mode='r', encoding='utf-8')as f:
        lines = f.readlines()
        for line in tqdm(lines):
            data = json.loads(line)
            if data_type=='elam':
                ridx = data['id']
                query=''.join(data['q'])
                query=query+"概括50字内："
            elif data_type=='Lecard':
                ridx = data['ridx']
                query=data['q']+"概括50字内："

            else:
                print("error")
                exit()
            query_g=generate(model,tokenizer,query) 
            print('query_g:{}'.format(query_g))
            querys[str(ridx)] = query_g
            
    with open(outout_file,'w',encoding='utf8') as fw:
        for key, value in querys.items():
            json_line = json.dumps({"ridx": key, "q": value},ensure_ascii=False) + '\n'
            fw.write(json_line)
            
    return querys

def create_folders(source_folder, dest_folder):
    for item in os.listdir(source_folder):
        source_item = os.path.join(source_folder, item)
        dest_item = os.path.join(dest_folder, item)
        if os.path.isdir(source_item):
            os.makedirs(dest_item)
            create_folders(source_item, dest_item)
            
def process_json_file(json_file_path,data_type,ttp):
    with open(json_file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)

        casess={}
        doc=''
        if data_type=='elam':
            doc += "".join(js_dict['doc'])
        elif data_type=='Lecard':
            doc += js_dict['ajjbqk']
        
        doc=doc+"概括50字内："
        case_g=generate(model,tokenizer,doc) 
        casess['doc']=case_g
        casess['laws']=js_dict['laws']
        
    if ttp=='train':
        outout_file=json_file_path.replace('train','train_laws')
    elif ttp=='eval':
        outout_file=json_file_path.replace('eval','eval_laws')
    
    with open(outout_file,'w',encoding='utf8') as fw:
        json.dump(casess,fw, ensure_ascii=False)
        
def traverse_folder(folder_path,data_type,ttp):
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                process_json_file(json_file_path,data_type,ttp)


logic_path='../Law-level/law_article_1_451.json'
with open(logic_path, 'r', encoding='utf-8') as f:
    logic_file = json.load(f)

def process_json_file(json_file_path,data_type):
    with open(json_file_path, mode='r', encoding='utf-8')as f:
        js_dict = json.load(f)
        casess={}
        casess['doc']=js_dict['doc']
        casess['laws']=js_dict['laws']
        
        law_idx = []
        for i in casess['laws']:
            p = re.compile('第(?:十|百|零|一|二|三|四|五|六|七|八|九){1,10}条(?:之(?:一|二|三|四|五|六|七|八|九))?(?:第(?:十|百|零|一|二|三|四|五|六|七|八|九)款)?')
            m = re.findall(pattern=p, string=i)
            law_idx.append(m[0])
        
        remove_pattern = re.compile(r'(?<=条)(第[^条]+)款')
        law_idx = [remove_pattern.sub('', item) for item in law_idx]
        law_idx = list(set(law_idx))
        
        law_pres=[]
        law_rules=[]
        law_crimes=[]
        for law_name in law_idx:
            law_pre=""
            law_rule=""
            if law_name not in logic_file:
                continue
            else:
                law_t=logic_file[law_name]
            
            predicates = {key: value for key, value in law_t['Predicate'].items() if key.startswith('P')}
            for item in predicates.items():
                law_pre += f"{item[0]}: {item[1]}。"

            rules = [re.sub(r'=.*', '=Y', rule) for sublist in law_t['Rules'] for rule in sublist]
            law_rule='。'.join(rules)
            
            if isinstance(law_t['Criminal'], str):
                law_crime=law_t['Criminal']
            else:
                law_crime=','.join(law_t['Criminal'])
            law_crime+="。"
            law_rule=law_rule+"。Y:"+str(law_crime)
            
            law_pres.append(law_pre)
            law_rules.append(law_rule)
            law_crimes.append(law_crime)
        casess['predicate']=law_pres
        casess['rules']=law_rules
        casess['crimes']=law_crimes

    outout_file=json_file_path.replace('train_part_laws','train_part_laws_law')
    
    
    with open(outout_file,'w',encoding='utf8') as fw:
        json.dump(casess,fw, ensure_ascii=False)

        
def traverse_folder(folder_path,data_type):
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.endswith('.json'):
                json_file_path = os.path.join(root, file)
                process_json_file(json_file_path,data_type)
                