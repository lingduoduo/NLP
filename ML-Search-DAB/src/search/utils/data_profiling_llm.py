import json
import subprocess
import requests
from pathlib import Path
import httpx
from openai import AzureOpenAI

from typing import List, Optional
import numpy as np
import umap


path = Path.cwd().resolve()
cert_path = path / "utils"/ "ADP_Internal_Root_CA_GN2.pem"

def get_bearer_token(client_secret):
    client_id = "49162025-d5ad-486f-8d6a-f85c6ac1c739"
    scope = "0686f7e6-df73-4037-af26-c03f6fc75de4/.default"

    command = ["curl",
                "https://login.microsoftonline.com/4c2c8480-d3f0-485b-b750-807ff693802f/oauth2/v2.0/token",
                "-H", "Content-Type: application/x-www-form-urlencoded",
                "-H",
                "Cookie: fpc=Atdv_fH_1-BOhRtb1PImOIcWRzo2AQAAANvSQd8OAAAAyk5G3wEAAADy0kHfDgAAAA; stsservicecookie=estsfd; x-ms-gateway-slice=estsfd",
                "-d", "grant_type=client_credentials",
                "-d", f"client_id={client_id}",
                "-d", f"client_secret={client_secret}",
                "-d", f"scope={scope}"]

    result = subprocess.run(command, capture_output=True, text=True)
    token_dict = json.loads(result.stdout)
    token = token_dict['access_token']
    if not token:
        print(f"Failed to execute token request")
    return token

def invoke_titan_model(client_secret, query, model="amazon.titan-embed-text-v1-pgo"):
    url = f"https://aigateway-amrs-nonprod.oneadp.com/v0/r1/model/{model}/invoke"
    bearer_token = get_bearer_token(client_secret)
    headers = {
        "Authorization": f"Bearer {bearer_token}",
        "Content-Type": "application/json",
        "Acccept": "application/json"
    }
    request_body = {
        "inputText": query
    }
    request_json = json.dumps(request_body)
    try:
        response = requests.post(
            url=url,
            headers=headers,
            data=request_json,
            timeout=(1000,1000),
            verify=cert_path
        )
    except Exception as error:
        print(error)
        return None
    return response.json()

def get_openai_embedding(client_secret: str, text: str, model="text-embedding-3-large_1-pgo-amrs") -> List[float]:
    bearer_token = get_bearer_token(client_secret)
    text = text.replace("\n", " ")
    httpx_client = httpx.Client(verify=False)
    client = AzureOpenAI(
        api_version="2024-06-01",
        azure_endpoint="https://aigateway-amrs-nonprod.oneadp.com/v0/r0",
        azure_ad_token=f"{bearer_token}",
        http_client=httpx_client,
        timeout=30,
    )
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

def global_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: Optional[int] = None,
        metric: str = "cosine",
) -> np.ndarray:
    """
    Perform global dimensionality reduction on the input embeddings using UMAP.
    Args:
        embeddings: The embedding vectors to reduce.
        dim: The target number of dimensions.
        n_neighbors: Number of neighbors for UMAP; defaults to sqrt(number of embeddings) if None.
        metric: Distance metric to use for UMAP; default is cosine similarity.
    Returns: 
        A numpy array of embeddings reduced to the specified dimension.
    """
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
    
def local_cluster_embeddings(
        embeddings: np.ndarray,
        dim: int,
        n_neighbors: int = 10,
        metric: str = "cosine",
) -> np.ndarray:
    """
    Perform local dimensionality reduction on embeddings using UMAP, typically after global clustering.

    Args:
        embeddings: The embedding vectors to reduce.
        dim: The target number of dimensions.
        n_neighbors: Number of neighbors for UMAP.
        metric: Distance metric to use for UMAP; default is cosine similarity.
    Reurn: 
        A numpy array of embeddings reduced to the specified dimension.
    """
    return umap.UMAP(n_neighbors=n_neighbors, n_components=dim, metric=metric).fit_transform(embeddings)
