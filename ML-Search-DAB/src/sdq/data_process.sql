-- Databricks notebook source


%sql
CREATE OR REPLACE FUNCTION ${ml_catalog}.${ml_sdq_db}.literal_eval_sdq(x STRING) 
RETURNS ARRAY<STRING>
LANGUAGE PYTHON
AS $$
import ast
import json
import base64
import pandas as pd

def eval_sdq(encoded_list):
    res = []
    try: 
        items = ast.literal_eval(base64.b64decode(encoded_list).decode('utf-8'))
        df = pd.DataFrame(data=items['dataframe_split']["data"], 
                        columns=items['dataframe_split']["columns"])
        d = df.to_dict(orient="records")
        for record in d:
            res.append(json.dumps(record))
    except Exception as e:
        print(f"Error processing data: {e}")
        return ["Error"]
    return res
        
return eval_sdq(x)
$$;