'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

from typing import Dict, List
from .bedrock import *

def format_search_results(search_results: List[Dict]) -> str:
    """
    Format search results into a clean, readable string for LLM reranking.

    Args:
        search_results (List[Dict]): List of search result dictionaries, each containing
            '_source' with description and metadata fields

    Returns:
        str: Formatted string with numbered results, where each result's fields are 
            separated by newlines
    """

    formatted_results = []
    for i, hit in enumerate(search_results):
        source = hit["_source"]
         
        result = f"[{i}]\n{source['description_s']}\n{source['description_txt_edgeNgram']}\n{source['metadata_s']}\n"
        formatted_results.append(result)
    
    return "\n".join(formatted_results)


def rerank_results(query: str, search_results: List[Dict], model_name: str, rerank_prompt: str) -> Dict:
    """
    Reranks search results using LLM.

    Args:
        query (str): The user's search query
        search_results (List[Dict]): List of search results to be reranked
        model_name (str): Name of the LLM model to use
        rerank_prompt (str): Prompt template for reranking

    Returns:
        Dict: Most relevant search result, or an empty dict if no relevant results found
    """
    if not search_results:
        return {}
 
    formatted_results = format_search_results(search_results)
    prompt = rerank_prompt.format(
        query=query,
        formatted_results=formatted_results,
        max_idx=len(search_results)-1
    )
   
    system_msg = "You are a search result ranking assistant. Only respond with the index of the best search result, or -1 if the query is out-of-scope or no results are relevant."
   
    try:
        if model_name == "sonnet35_v2":
            response = get_claude_response(system_msg, prompt)
        else:
            response, _ = get_nova_response(system_msg, prompt, model_name=model_name)
       
        idx = int(response.strip().split("\n")[0])
        return {} if idx == -1 else search_results[idx]
   
    except Exception as e:
        print(f"Reranking failed: {e}")
        return search_results[0]
