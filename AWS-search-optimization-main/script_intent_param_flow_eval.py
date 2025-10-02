'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

import requests
import time
import json
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.parameter_extraction import extract_parameters
from src.intent_classification import make_query_json
from src.bedrock import get_bearer_token, get_titan_response, get_cohere_response

get_bearer_token()

# Parameter extraction 
# Extract QnAs with matching modules 
query_json_path = "data/sample-benefits-questions.txt"
test_query_data = []
with open(query_json_path, "r", encoding="utf-8") as file:
    for line in file:
        query_dict = {'question': line}
        test_query_data.append(query_dict)

# # Extract QnAs with matching modules 
# query_json_path = "data/filteredQna_param.json"
# with open( query_json_path, "r" ) as file:
#     test_query_data = json.load(file)        

# SOR for parameter extraction
sor_v2_intent_path = "data/sor-v2-intents-mapping.json"
with open(sor_v2_intent_path, "r", encoding="utf-8") as file:
    sor_v2_intent = json.load(file)
module_keys = list(sor_v2_intent.keys())

# Inference versions
dev_prompt_version = "V3"
model_id = "nova_pro"

# Intent classification
OPENSEARCH_DOMAIN_FQDN = 'https://vpc-int-use1-opensearch-ml-c32nkeiaudrckcr2ep7jghefje.us-east-1.es.amazonaws.com'
HEADERS = {
    'Content-Type': 'application/json',
    'Accept-Encoding': 'gzip',
}
TOP_K=1

# Titan V1 index
# embed_model_id = "amazon.titan-embed-text-v1-pgo"
# INDEX_NAME = 'search-data-titan-embed-2'

# Titan V2 
embed_model_id = "amazon.titan-embed-text-v2:0-pgo"
# Titan V2 index - 1024
INDEX_NAME = 'search-data-titan-embed-v2_1024'
# Titan V2 index - 512
# INDEX_NAME = 'search-data-titan-embed-v2_512'

# Cohere English Index
# embed_model_id = "cohere.embed-english-v3-pgo"
# INDEX_NAME = 'search-data-cohere'

output_path = f"results/complete_flow_sample-benefits-questions_{embed_model_id}_{model_id}_prompt_{dev_prompt_version}.json"

res_list = []
embed_latency_list = []
opensearch_latency_list = []
intent_cls_latency_list = []
param_ext_latency_list = []
overall_latency_list = []

for test_query_dict in test_query_data:
    usr_query = test_query_dict['question']

    # Query embedding
    start_embed = time.time()
    embed_dict = get_titan_response(usr_query, embed_model_id)
    embed_vec = embed_dict["body"]["embedding"]
    embed_latency = time.time() - start_embed
    embed_latency_list.append(embed_latency) 

    # Search
    start_search = time.time()
    query_json = make_query_json(usr_query, 
                                 embed_vec=embed_vec, 
                                 top_k=TOP_K, 
                                 search_type='hybrid', 
                                 boost_value=0.0001)
    try:
        result = requests.post(f"{OPENSEARCH_DOMAIN_FQDN}/{INDEX_NAME}/_search", 
                                headers=HEADERS, 
                                json=query_json,
                                timeout=(1000, 1000))
        result.raise_for_status()
    except Exception as error:
        print(error)
        continue
    
    result_dict = json.loads(result.text)
    module_id = result_dict['hits']['hits'][0]["_source"]["id"]
    search_latency = time.time() - start_search
    opensearch_latency_list.append(search_latency)
    intent_cls_latency_list.append(search_latency+embed_latency)

    # load mapping/function param
    start_param_ext = time.time()
    if module_id not in module_keys:
        out_dict = test_query_dict
        out_dict['extracted_param'] = "[]"
        out_dict['est_module_id'] = module_id
        out_dict['embed_latency'] = embed_latency
        out_dict['opensearch_latency']= search_latency
        out_dict['intent_cls_latency']= search_latency+embed_latency
        out_dict['param_ext_latency']= 0
        out_dict['overall_latency'] = search_latency+embed_latency
        res_list.append(out_dict)
        continue
        
    sor_dict = sor_v2_intent[module_id]
    if "systemPromptTemplate" in sor_dict:
        inst_prompt = sor_dict["systemPromptTemplate"]
    else:
        inst_prompt = ""
        
    mapping = sor_dict['mapping']
    func_param_name_desc = {}
    func_param_name_desc["name"] = sor_dict['name']
    func_param_name_desc["parameters"] = sor_dict['parameters']
    
    response = extract_parameters(user_question=usr_query, 
                                  mapping_info=mapping, 
                                  function_desc=func_param_name_desc, 
                                  function_inst_prompt=inst_prompt,
                                  model_id=model_id, 
                                  dev_prompt_version=dev_prompt_version)
    param_ext_latency = time.time() - start_param_ext
    param_ext_latency_list.append(param_ext_latency)
    overall_latency_list.append(param_ext_latency+search_latency+embed_latency)
    
    print("-"*50)
    print(usr_query)
    print(response) 

    # For eval
    out_dict = test_query_dict
    out_dict['extracted_param'] = response
    out_dict['est_module_id'] = module_id
    out_dict['embed_latency'] = embed_latency
    out_dict['opensearch_latency']= search_latency
    out_dict['intent_cls_latency']= search_latency+embed_latency
    out_dict['param_ext_latency']= param_ext_latency
    out_dict['overall_latency'] = param_ext_latency+search_latency+embed_latency
    res_list.append(out_dict)
    
with open(output_path, "w", encoding='utf-8') as file:
    json.dump(res_list, file, indent=4)

print("-"*50)
print("Latency")
print("-"*50)
print( f"Avg: {np.average(overall_latency_list)}")
print( f"P10: {np.percentile(overall_latency_list, 10)}")
print( f"P90: {np.percentile(overall_latency_list, 90)}")
print( f"P99: {np.percentile(overall_latency_list, 99)}")
print( f"Max: {np.max(overall_latency_list)}")