'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

import time
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.parameter_extraction import extract_parameters
from src.bedrock import get_bearer_token

get_bearer_token()

# Extract QnAs with matching modules 
query_json_path = "data/filteredQna_param.json"
with open( query_json_path, "r", encoding='utf-8' ) as file:
    test_query_data = json.load(file)

sor_v2_intent_path = "data/sor-v2-intents-mapping.json"
with open(sor_v2_intent_path, "r", encoding='utf-8') as file:
    sor_v2_intent = json.load(file)
module_keys = list(sor_v2_intent.keys())

# Inference versions
dev_prompt_version = "V3"
model_id = "nova_pro"
output_path = f"results/filteredQna_param_{model_id}_prompt_{dev_prompt_version}.json"

res_list = []
latency_list = []

for test_query_dict in test_query_data:
    usr_query = test_query_dict['question']
    
    # Replace with estimated Id
    module_id = test_query_dict['id']

    start_time = time.time()
    # load mapping/function param
    sor_dict = sor_v2_intent[module_id]
    if "systemPromptTemplate" in sor_dict:
        inst_prompt = sor_dict["systemPromptTemplate"]
    else:
        inst_prompt = ""
        
    mapping = sor_dict['mapping']
    func_param_name_desc = {}
    func_param_name_desc["name"] = sor_dict['name']
    # func_param_name_desc["description"] = sor_dict['description']
    func_param_name_desc["parameters"] = sor_dict['parameters']
    
    response = extract_parameters(user_question=usr_query, 
                                  mapping_info=mapping, 
                                  function_desc=func_param_name_desc, 
                                  function_inst_prompt=inst_prompt,
                                  model_id=model_id, 
                                  dev_prompt_version=dev_prompt_version)
    latency = time.time() - start_time
    # time.sleep(1)
    print("-"*50)
    print(usr_query)
    print(response) 

    # For eval
    out_dict = test_query_dict
    out_dict['extracted_param'] = response
    out_dict['latency'] = latency
    res_list.append(out_dict)
    latency_list.append(latency)

with open(output_path, "w", encoding='utf-8') as file:
    json.dump(res_list, file, indent=4)

print("-"*50)
print("Latency")
print("-"*50)
print( f"Avg: {np.average(latency_list)}")
print( f"P10: {np.percentile(latency_list, 10)}")
print( f"P90: {np.percentile(latency_list, 90)}")
print( f"P99: {np.percentile(latency_list, 99)}")
print( f"Max: {np.max(latency_list)}")

print(len(latency_list))