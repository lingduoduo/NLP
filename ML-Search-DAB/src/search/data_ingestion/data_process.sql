-- Databricks notebook source

CREATE OR REPLACE FUNCTION ${ml_catalog}.${ml_search_db}.literal_eval_search(x STRING) 
RETURNS ARRAY<STRING>
LANGUAGE PYTHON
AS $$
import ast
import json
import uuid

def eval_search(details_search_results):
    res = []
    fields = ['_id', 'resPos', 'traceId']
    try: 
        data = ast.literal_eval(details_search_results)
        query_id = str(uuid.uuid4())
        for item in reversed(data):
            d = {}
            for k in item:
                if k in fields:
                    d[k] = item[k]
                if k == 'scoreDetails':
                    d['finalScore'] = item[k].get('finalScore', 0.0)
            d['queryId'] = query_id
            res.append(json.dumps(d))
    except Exception as e:
        print(f"Error processing data: {e}")
        return ["Error"]
    return res

return eval_search(x)
$$;


CREATE OR REPLACE FUNCTION ${ml_catalog}.${ml_search_db}.literal_eval_people(x STRING) 
RETURNS ARRAY<STRING>
LANGUAGE PYTHON
AS $$
import ast
import json
import uuid

def eval_people(details_search_results):
    res = []
    fields = ['_id', 'legalName', 'displayName', 'position', 'location', 'eID', 'resPos', 'scoreDetails', 'traceId']
    try: 
        data = ast.literal_eval(details_search_results)
        query_id = str(uuid.uuid4())
        for item in reversed(data):
            d = {}
            for k in item:
                if k in fields:
                    d[k] = item[k]
                if k == 'scoreDetails':
                    d['finalScore'] = item[k].get('finalScore', 0.0)
            d['queryId'] = query_id
            res.append(json.dumps(d))
    except Exception as e:
        print(f"Error processing data: {e}")
        return ["Error"]
    return res

return eval_people(x)
$$;


CREATE OR REPLACE FUNCTION ${ml_catalog}.${ml_search_db}.literal_eval_action(x STRING) 
RETURNS ARRAY<STRING>
LANGUAGE PYTHON
AS $$
import ast
import json
import uuid

def eval_action(details_search_results):
    res = []
    fields = ['_id',  'subtitle', 'caption', 'description', 'resPos', 'traceId']
    try: 
        data = ast.literal_eval(details_search_results)
        query_id = str(uuid.uuid4())
        for item in reversed(data):
            d = {}
            for k in item:
                if k in fields:
                    d[k] = item[k]  
                if k == 'scoreDetails':
                    d['solrScore'] = item[k].get('solrScore', 0.0) 
                    d['finalScore'] = item[k].get('finalScore', 0.0)
                    d['scoreDistributionCaption'] = item[k].get('scoreDistribution', {}).get('caption', 0.0)
                    d['scoreDistributionDescription'] = item[k].get('scoreDistribution', {}).get('description', 0.0)
                    d['scoreDistributionKeywords'] = item[k].get('scoreDistribution', {}).get('keyword', 0.0)
            d['queryId'] = query_id
            res.append(json.dumps(d))
    except Exception as e:
        print(f"Error processing data: {e}")
        return ["Error"]
    return res

return eval_action(x)
$$;
