'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

import pandas as pd
import time
from typing import Dict, List, Tuple
from src.bedrock import *
from src.rerank import rerank_results
from tqdm.auto import tqdm


def make_query_json(
    query_str: str,
    embed_vec: List[float],
    top_k: int,
    search_type: str = 'hybrid',
    boost_value: float = 0.0001
) -> Dict:
    """
    Creates query JSON based on search type and boost value for hybrid search.

    Args:
        query_str (str): The search query string.
        embed_vec (List[float]): Vector embedding of the query.
        top_k (int): Number of top results to return.
        search_type (str, optional): Type of search to perform ('keyword', 'semantic', or 'hybrid'). 
                                   Defaults to 'hybrid'.
        boost_value (float, optional): Boost value for hybrid search to balance between vector 
                                     and keyword search. Defaults to 0.0001.

    Returns:
        Dict: JSON query object with the following structure:
            - For hybrid search: Contains both vector and keyword search parameters
            - For semantic search: Contains only vector search parameters
            - For keyword search: Contains only keyword search parameters

    """
    
    query_configs = {
        'keyword': {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [{
                        "multi_match": {
                            "query": query_str,
                            "fields": ["description_t", "description_txt_edgeNgram"],
                            "analyzer": "english",
                            "boost": 1
                        }
                    }]
                }
            }
        },
        'semantic': {
            "size": top_k,
            "query": {
                "knn": {
                    "vec_embedding": {
                        "vector": embed_vec,
                        "k": top_k
                    }
                }
            }
        },
        'hybrid': {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query_str,
                                "fields": ["description_t", "description_txt_edgeNgram"],
                                "analyzer": "english",
                                "boost": boost_value
                            }
                        },
                        {
                            "knn": {
                                "vec_embedding": {
                                    "vector": embed_vec,
                                    "k": top_k,
                                    "boost": 1
                                }
                            }
                        }
                    ]
                }
            }
        }
    }
    
    return query_configs[search_type]


def process_search_type(
    query: str,
    target_id: str,
    embed_vec: List[float],
    search_type: str,
    boost: float = None,
    opensearch_domain: str = None,
    index_name: str = None,
    headers: Dict = None,
    top_k: int = None,
    rerank_model: str = None,
    rerank_prompt: str = None,
) -> Tuple[Dict, Dict]:
    """
    Helper function to process a single search type and its reranked version.
 
    Args:
        query (str): Search query string to be processed.
        target_id (str): Ground truth id.
        embed_vec (List[float]): Vector embedding of the query.
        search_type (str): Type of search to perform ('keyword', 'semantic', or 'hybrid').
        boost (float, optional): Boost value for hybrid search. Defaults to None.
        opensearch_domain (str, optional): OpenSearch domain URL. Defaults to None.
        index_name (str, optional): Name of the index to search. Defaults to None.
        headers (Dict, optional): Request headers for API calls. Defaults to None.
        top_k (int, optional): Number of top results to return. Defaults to None.
        rerank_model (str, optional): Model name for reranking results. Defaults to None.
        rerank_prompt (str, optional): Prompt template for reranking. Defaults to None.
       
    Returns:
        Tuple of (original_result, reranked_result) dictionaries
    """
   
    start_time = time.time()
    query_json = make_query_json(query, embed_vec, top_k, search_type, boost)
    try:
        response = requests.post(
            f"{opensearch_domain}/{index_name}/_search",
            headers=headers,
            json=query_json,
            timeout=(1000, 1000)
        )
        response.raise_for_status()
    except Exception as error:
        print(error)
        return 
       
    latency = time.time() - start_time
   
    result_dict = response.json()
    result_list = result_dict['hits']['hits']
   
    # Process original results
    orig_hit_id = result_list[0]["_source"]["id"] if result_list else "NaN"
   
    original_result = {
        'query': query,
        'target_id': target_id,
        'predicted_id': orig_hit_id,
        'latency': latency,
        'accuracy': 1 if target_id == orig_hit_id else 0
    }
   
    # Process reranked results
    if len(result_list) > 1:
        rerank_start_time = time.time()
        best_result = rerank_results(query, result_list, rerank_model, rerank_prompt)
        rerank_latency = time.time() - rerank_start_time
       
        reranked_hit_id = best_result.get("_source", {}).get("id", "NaN") if best_result else "NaN"
    else:
        rerank_latency = 0
        reranked_hit_id = orig_hit_id
 
    target_in_results = any(item["_source"]["id"] == target_id for item in result_list)
    target_result = next((item for item in result_list if item["_source"]["id"] == target_id), None)
 
    reranked_result = {
        'query': query,
        'target_id': target_id,
        'predicted_id': reranked_hit_id,
        'latency': latency + rerank_latency,
        'accuracy': 1 if target_id == reranked_hit_id else 0,
        'target_in_results': target_in_results,
       
        # Details about the predicted result
        'predicted_description_s': best_result.get("_source", {}).get("description_s", "") if best_result else "",
    	'predicted_description_txt_edgeNgram': best_result.get("_source", {}).get("description_txt_edgeNgram", "") if best_result else "",
    	'predicted_metadata_s': best_result.get("_source", {}).get("metadata_s", "") if best_result else "",
       
        # Details about the target (if it's in the results)
        'target_description_s': target_result["_source"].get("description_s", "") if target_result else "",
        'target_description_txt_edgeNgram': target_result["_source"].get("description_txt_edgeNgram", "") if target_result else "",
        'target_metadata_s': target_result["_source"].get("metadata_s", "") if target_result else "",
       
        # Position of target in results (if found)
        'target_position': next((i for i, item in enumerate(result_list) if item["_source"]["id"] == target_id), -1)
    }
 
    return original_result, reranked_result


def evaluate_queries(
    query_data: List[Dict],
    opensearch_domain: str,
    index_name: str,
    headers: Dict,
    top_k: int = 5,
    embed_model: str = "amazon.titan-embed-text-v1-pgo",
    rerank_model: str = "sonnet35_v2",
    rerank_prompt: str = None,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluates queries with accuracy for original and reranked results.

    Args:
        query_data (List[Dict]): List of dictionaries containing query information.
        opensearch_domain (str): OpenSearch domain URL.
        index_name (str): Name of the index to search.
        headers (Dict): Request headers for API calls.
        top_k (int, optional): Number of top results to return. Defaults to 5.
        embed_model (str, optional): Model name for embedding generation. 
                                   Defaults to "amazon.titan-embed-text-v1-pgo".
        rerank_model (str, optional): Model name for reranking. 
                                    Defaults to "sonnet35_v2".
        rerank_prompt (str, optional): Prompt template for reranking. Defaults to None.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary containing evaluation results with:
            - Keys: Different search types
            - Values: Pandas DataFrames with evaluation results and metrics
    """

    # base_search_types = ['keyword', 'semantic']
    base_search_types = []
    # boost_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]
    boost_values = [1e-4]
                    
    search_types = []
    for stype in base_search_types:
        search_types.append(stype)
        search_types.append(f'{stype}_reranked')
    for boost in boost_values:
        search_types.append(f'hybrid_boost_{boost}')
        search_types.append(f'hybrid_boost_{boost}_reranked')
    
    results = {stype: [] for stype in search_types}
    
    for i, query_dict in tqdm(enumerate(query_data), total=len(query_data)):
        input_query = query_dict['question']
        target_id = query_dict['id']

        embed_start_time = time.time()
        response_dict = get_titan_response(input_query, model_id=embed_model)
        embed_vec = response_dict["body"]["embedding"]
        embed_time = time.time() - embed_start_time
        
        # Process base search types
        for base_type in base_search_types:
            orig_result, rerank_result = process_search_type(
                query=input_query,
                target_id=target_id,
                embed_vec=embed_vec,
                search_type=base_type,
                opensearch_domain=opensearch_domain,
                index_name=index_name,
                headers=headers,
                top_k=top_k,
                rerank_model=rerank_model,
                rerank_prompt=rerank_prompt,
            )

            orig_result["latency"] += embed_time
            rerank_result["latency"] += embed_time
            
            results[base_type].append(orig_result)
            results[f'{base_type}_reranked'].append(rerank_result)
        
        # Process hybrid searches
        for boost in boost_values:
            search_type = f'hybrid_boost_{boost}'
            orig_result, rerank_result = process_search_type(
                query=input_query,
                target_id=target_id,
                embed_vec=embed_vec,
                search_type='hybrid',
                boost=boost,
                opensearch_domain=opensearch_domain,
                index_name=index_name,
                headers=headers,
                top_k=top_k,
                rerank_model=rerank_model,
                rerank_prompt=rerank_prompt,
            )

            orig_result["latency"] += embed_time
            rerank_result["latency"] += embed_time
            
            results[search_type].append(orig_result)
            results[f'{search_type}_reranked'].append(rerank_result)

        # update bearer token
        if (i+1) % 500 == 0:
            get_bearer_token()

        # # Due to rate limit on Nova models
        # if "nova" in rerank_model and (i+1) % 20 == 0:
        #     time.sleep(40)

    return {stype: pd.DataFrame(data) for stype, data in results.items()}


def get_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates summary metrics from results DataFrame
    """
    metrics = {
        'Accuracy': df['accuracy'].mean(),
        'Avg Latency': df['latency'].mean(),
        'P10 Latency': df['latency'].quantile(0.1),
        'P90 Latency': df['latency'].quantile(0.9),
        'P99 Latency': df['latency'].quantile(0.99),
        'Max Latency': df['latency'].max()
    }
    return pd.DataFrame([metrics]).T