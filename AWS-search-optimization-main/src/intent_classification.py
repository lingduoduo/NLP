'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

import requests
import time
import pandas as pd

from src.bedrock import get_titan_response, get_claude_response
from src.rerank import rerank_results
from typing import Dict, List

MAX_ERRORS = 10
OPENSEARCH_DOMAIN_FQDN = 'https://vpc-int-use1-opensearch-ml-c32nkeiaudrckcr2ep7jghefje.us-east-1.es.amazonaws.com'
HEADERS = {
    'Content-Type': 'application/json',
    'Accept-Encoding': 'gzip',
}

def format_search_results(search_results: List[Dict]) -> str:
    """
    Format search results into a clean, readable string
    """
    formatted_results = []
    for i, hit in enumerate(search_results):
        source = hit["_source"]
        result = f"[{i}]\n{source['description_s']}\n{source['metadata_s']}\n"
        formatted_results.append(result)
    
    return "\n".join(formatted_results)

def make_query_json(query_str: str, 
                    embed_vec: List[float], 
                    top_k: int, 
                    search_type: str = 'hybrid', 
                    boost_value: float = 0.0001) -> Dict:
    """
    Creates query JSON based on search type and boost value for hybrid search
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
                                "fields": ["description_t", "descriptiobun_txt_edgeNgram"],
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

def evaluate_queries(query_data: List[Dict], 
                     opensearch_domain: str,
                     index_name: str,
                     headers: Dict,
                     top_k: int = 5,
                     model_id = "amazon.titan-embed-text-v2:0-pgo"
                    ) -> Dict[str, pd.DataFrame]:
    """
    Evaluates queries for keyword, semantic, and hybrid searches with different boost values
    """
    # Define search types and boost values
    base_search_types = ['keyword', 'semantic']
    # boost_values = [1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10]  # from 0.0001 to 10
    boost_values = [1e-4, 1e-3]
    
    search_types = []
    for stype in base_search_types:
        search_types.append(stype)
        search_types.append(f'{stype}_reranked')
    for boost in boost_values:
        search_types.append(f'hybrid_boost_{boost}')
        search_types.append(f'hybrid_boost_{boost}_reranked')
    
    results = {stype: [] for stype in search_types}
    
    for query_dict in query_data:
        input_query = query_dict['question']
        target_id = query_dict['id']
        
        # Get embedding once for all search types
        embed_start_time = time.time()
        response_dict = get_titan_response(input_query, model_id=model_id)
        embed_vec = response_dict["body"]["embedding"]
        embed_time = time.time() - embed_start_time
        
        # Run base search types (keyword and semantic)
        for base_type in base_search_types:
            start_time = time.time()
            query_json = make_query_json(input_query, embed_vec, top_k, base_type)
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
                continue

            latency = time.time() - start_time + embed_time
            
            result_dict = response.json()
            result_list = result_dict['hits']['hits']
            
            # Process original results
            orig_result_list = result_list + [{"_source":{"id": "NaN"}}]*(top_k-len(result_list))
            orig_hit_id_list = [orig_result_list[i]["_source"]["id"] for i in range(top_k)]
            
            results[base_type].append({
                'query': input_query,
                'target_id': target_id,
                'hit_ids': orig_hit_id_list,
                'latency': latency,
                'recall@1': 1 if target_id in orig_hit_id_list[:1] else 0,
                'recall@3': 1 if target_id in orig_hit_id_list[:3] else 0,
                'recall@5': 1 if target_id in orig_hit_id_list[:5] else 0
            })
            
            # Process reranked results
            if len(result_list) > 1:
                rerank_start_time = time.time()
                reranked_list = rerank_results(input_query, result_list)
                rerank_latency = time.time() - rerank_start_time
                
                reranked_list = reranked_list + [{"_source":{"id": "NaN"}}]*(top_k-len(reranked_list))
                reranked_hit_id_list = [reranked_list[i]["_source"]["id"] for i in range(top_k)]
            else:
                rerank_latency = 0
                reranked_hit_id_list = orig_hit_id_list
            
            results[f'{base_type}_reranked'].append({
                'query': input_query,
                'target_id': target_id,
                'hit_ids': reranked_hit_id_list,
                'latency': latency + rerank_latency,  # Include both search and rerank time
                'recall@1': 1 if target_id in reranked_hit_id_list[:1] else 0,
                'recall@3': 1 if target_id in reranked_hit_id_list[:3] else 0,
                'recall@5': 1 if target_id in reranked_hit_id_list[:5] else 0
            })
        
        # Run hybrid searches with different boost values
        for boost in boost_values:
            start_time = time.time()
            
            query_json = make_query_json(input_query, embed_vec, top_k, 'hybrid', boost)
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
                continue

            latency = time.time() - start_time
            
            result_dict = response.json()
            result_list = result_dict['hits']['hits']
            
            # Process original hybrid results
            search_type = f'hybrid_boost_{boost}'
            orig_result_list = result_list + [{"_source":{"id": "NaN"}}]*(top_k-len(result_list))
            orig_hit_id_list = [orig_result_list[i]["_source"]["id"] for i in range(top_k)]
            
            results[search_type].append({
                'query': input_query,
                'target_id': target_id,
                'hit_ids': orig_hit_id_list,
                'latency': latency,
                'recall@1': 1 if target_id in orig_hit_id_list[:1] else 0,
                'recall@3': 1 if target_id in orig_hit_id_list[:3] else 0,
                'recall@5': 1 if target_id in orig_hit_id_list[:5] else 0
            })
            
            # Process reranked hybrid results
            if len(result_list) > 1:
                rerank_start_time = time.time()
                reranked_list = rerank_results(input_query, result_list)
                rerank_latency = time.time() - rerank_start_time
                
                reranked_list = reranked_list + [{"_source":{"id": "NaN"}}]*(top_k-len(reranked_list))
                reranked_hit_id_list = [reranked_list[i]["_source"]["id"] for i in range(top_k)]
            else:
                rerank_latency = 0
                reranked_hit_id_list = orig_hit_id_list
            
            results[f'{search_type}_reranked'].append({
                'query': input_query,
                'target_id': target_id,
                'hit_ids': reranked_hit_id_list,
                'latency': latency + rerank_latency,
                'recall@1': 1 if target_id in reranked_hit_id_list[:1] else 0,
                'recall@3': 1 if target_id in reranked_hit_id_list[:3] else 0,
                'recall@5': 1 if target_id in reranked_hit_id_list[:5] else 0
            })
    
    return {stype: pd.DataFrame(data) for stype, data in results.items()}

def get_summary_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates summary metrics from results DataFrame
    """
    metrics = {
        'Recall@1': df['recall@1'].mean(),
        'Recall@3': df['recall@3'].mean(),
        'Recall@5': df['recall@5'].mean(),
        'Avg Latency': df['latency'].mean(),
        'P10 Latency': df['latency'].quantile(0.1),
        'P90 Latency': df['latency'].quantile(0.9),
        'P99 Latency': df['latency'].quantile(0.99),
        'Max Latency': df['latency'].max()
    }
    return pd.DataFrame([metrics]).T