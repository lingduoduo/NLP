'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

import json
from src.parameter_extraction import extract_parameters

# Extract QnAs with matching modules 
query_json_path = "data/filteredQna.json"
with open( query_json_path, "r", encoding="utf-8" ) as file:
    test_query_data = json.load(file)

sor_v2_intent_path = "data/sor-v2-intents.json"
with open(sor_v2_intent_path, "r", encoding="utf-8") as file:
    sor_v2_intent = json.load(file)

module_keys = list(sor_v2_intent.keys())

print(module_keys) 

variant_matching_list = []
test_target_id_set = set()
for test_query_dict in test_query_data:
    target_id = test_query_dict['id']
    # print( target_id )
    if target_id in module_keys:
        variant_matching_list.append( test_query_dict )
        test_target_id_set.add(target_id)

print( len(variant_matching_list))

print(test_target_id_set)

out_json_path = "data/filteredQna_param.json"

with open( out_json_path, "w",encoding='utf-8') as file:
    json.dump( variant_matching_list, file, indent=4)