'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

from typing import List
import json
import subprocess
import shlex


BEARER_TOKEN_STR = ""

SONNET37_ID = "anthropic.claude-3-7-sonnet-20250219-v1:0-pgo"
SONNET35_ID = "anthropic.claude-3-5-sonnet-20241022-v2:0-pgo"
NOVA_PRO_ID = "amazon.nova-pro-v1:0-pgo"
NOVA_LITE_ID = "amazon.nova-lite-v1:0-pgo"

MODEL_IDS = {
    "sonnet35_v2": SONNET35_ID,
    "sonnet37": SONNET37_ID,
    "nova_pro": NOVA_PRO_ID,
    "nova_lite": NOVA_LITE_ID,
}

def get_bearer_token():
    """
    Retrieves a bearer token from Microsoft Azure AD using client credentials flow.
    
    This function sends a curl command to obtain an access token, then stores it
    in a global variable BEARER_TOKEN_STR.
    
    The function uses predefined client credentials and scope to authenticate 
    and authorize the request.
    
    Returns:
        None
    
    Side effects:
        - Sets the global variable BEARER_TOKEN_STR with the obtained access token.
        - Prints a confirmation message to the console.
    
    Raises:
        subprocess.CalledProcessError: If the curl command fails.
        json.JSONDecodeError: If the response cannot be parsed as JSON.
    """

    CLIENT_ID = "[YOUR ADP AI GATEWAY CLIENT ID]"
    CLIENT_SECRET = "[YOUR ADP AI GATEWAY CLIENT SECRET]"
    SCOPE = "[YOUR ADP AI GATEWAY SCOPE]"

    sanitized_CLIENT_ID = shlex.escape(CLIENT_ID)
    sanitized_CLIENT_SECRET = shlex.escape(CLIENT_SECRET)
    sanitized_SCOPE = shlex.escape(SCOPE)


    command = ["curl",
               "https://login.microsoftonline.com/4c2c8480-d3f0-485b-b750-807ff693802f/oauth2/v2.0/token",
               "-H", "Content-Type: application/x-www-form-urlencoded",
               "-H", "Cookie: fpc=Atdv_fH_1-BOhRtb1PImOIcWRzo2AQAAANvSQd8OAAAAyk5G3wEAAADy0kHfDgAAAA; stsservicecookie=estsfd; x-ms-gateway-slice=estsfd",
               "-d", "grant_type=client_credentials",
               "-d", f"client_id={sanitized_CLIENT_ID}",
               "-d", f"client_secret={sanitized_CLIENT_SECRET}",
               "-d", f"scope={sanitized_SCOPE}" ]

    
    result = subprocess.run(command, capture_output=True, text=True)
    
    os_token_dict = json.loads(result.stdout)

    global BEARER_TOKEN_STR    
    BEARER_TOKEN_STR = os_token_dict['access_token']
    
    print( "Bearer token set as BEARER_TOKEN_STR global variable" )
    
def invoke_model(requested_model, json_payload, ssl_verify=False):
    """
    Invoke a specified AWS Bedrock model through the AI gateway.  Use python requests to send
    a POST with authentication included in the header and some friendly defaults for timeouts.
    Trust the default certificate store unless directed otherwise.
    """
    url = f"https://aigateway-amrs-nonprod.oneadp.com/v0/r1/model/{requested_model}/invoke"

    try:
        response_check = requests.get(url,timeout=(1000,1000))
        response_check.raise_for_status() 
    except Exception as e:
        print(f"Request failed:{e}")
        return None

    headers = {
        "Authorization": f"Bearer {BEARER_TOKEN_STR}",
        "Content-Type": "application/json",
        "Acccept": "application/json",
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            data=json_payload,
            timeout=(1000, 1000),
            verify=ssl_verify,
        )
        response.raise_for_status()        
    except Exception as error:
        print(error)
        return None

    return response.json()

def get_claude_response(
    system_msg,
    human_msg,
    assistant_msg="",
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0-pgo",
    temp=0.001,
    max_tokens=4096,
    stop_sequences=["Human:"]
):
    """
    Send a request to Claude 3.5 Sonnet and return its response and usage data.

    Creates a properly formatted request body for Claude, invokes the model,
    and extracts the text response along with token usage information.

    Args:
        system_msg: The system message that provides instructions to Claude
        human_msg: The user's message to Claude
        temp: Temperature parameter controlling randomness (0-1, default: 0)
        max_tokens: Maximum number of tokens in the response (default: 8192)

    Returns:
        Tuple containing:
            - The text response from Claude
            - Dictionary with token usage information
    """

    if assistant_msg == "":
        request_body = {
            "system": system_msg,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temp,
            "stop_sequences": stop_sequences,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": human_msg
                        }
                    ]
                }
            ]
        }
    else:
        request_body = {
            "system": system_msg,
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temp,
            "stop_sequences": stop_sequences,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": human_msg
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": assistant_msg
                        }
                    ]
                }
            ]
        }

    request_json = json.dumps(request_body)
    try:
        response = invoke_model(model_id, request_json)
        response_text = response.get("body")["content"][0]["text"]
    except Exception as e:
        print(f"Exception: {e}")
        return "Bedrock Exception"
        
    return response_text

def get_titan_response(query, model_id="amazon.titan-embed-text-v1-pgo", dimensions=None):
    """
    Send a request to the Titan embedding model.

    Args:
        query (str): User query
        model_id (str): Titan Embedding model ID
        dimensions (int): Embedding vector dimension for Titan V2. 512 or 1024.
    """
    request_body = {
        "inputText": query
    }

    if dimensions is not None:
        request_body["dimensions"] = dimensions
        
    request_json = json.dumps(request_body)
    
    response = invoke_model(model_id, request_json)
    return response


def get_cohere_response(query, input_type, model_id="cohere.embed-english-v3-pgo"):
    '''
    Send a request to the Cohere embedding model. 
    
    Args:
        query (str): User query
        input_type (str): Input type. 'search_query' for query, 'search_document' for document/data indexing
        model_id (str): Cohere model ID.
    '''
    request_body = {
        "texts": [query],
        "input_type": input_type,
    }   

    request_json = json.dumps(request_body)
    
    response = invoke_model(model_id, request_json)
    return response
    

def get_nova_response(
    system_msg: str,
    human_msg: str,
    assistant_msg: str="",
    model_name: str = "nova_pro",
    temp: float = 0,
    max_tokens: int = 5000,
    stop_sequences: List[str]=["Human:"]
) -> str:
    """
    Send a request to Amazon's Nova model and return its response.

    Creates a properly formatted request body for Nova, invokes the model,
    and extracts the text response.

    Args:
        system_msg: The system message that provides instructions to Nova
        human_msg: The user's message to Nova
        model_name: The key in MODEL_IDS dictionary for the specific Nova model (default: "nova_pro")
        temp: Temperature parameter controlling randomness (0-1, default: 0)
        max_tokens: Maximum number of tokens in the response (default: 5000)

    Returns:
        The text response from Nova
    """

    inf_params = {"maxTokens": max_tokens, "temperature": temp, "stopSequences": stop_sequences}
    request_body = {
        "system": [{"text": system_msg}],
        "schemaVersion": "messages-v1",
        "inferenceConfig": inf_params,
        "messages": [
            {"role": "user", "content": [{"text": human_msg}]},
            {"role": "assistant", "content":[{"text":assistant_msg}]}
        ],
    }

    request_json = json.dumps(request_body)
    response = invoke_model(MODEL_IDS[model_name], request_json)
    # print(response)

    response_text = response.get("body")["output"]["message"]["content"][0]["text"]
    usage_info = response.get("body")["usage"]
    return response_text, usage_info


# Cost per token for different models
model_cost_dict = {
    "sonnet35_v2": {"input": 0.000003, "output": 0.000015},
    "sonnet37": {"input": 0.000003, "output": 0.000015},
    "nova_pro": {"input": 8e-7, "output": 0.0000032},
    "nova_lite": {"input": 6e-8, "output": 2.4e-7},
}


def calculate_model_cost(input_tokens, output_tokens, model_name):
    """
    Calculate the cost for a model query based on token usage

    Args:
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        model_name: The model identifier (sonnet35_v2, sonnet37, nova_pro, nova_lite)

    Returns:
        Dictionary with cost information
    """
    # Verify the model is supported
    if model_name not in model_cost_dict:
        raise ValueError(f"Unknown model: {model_name}")

    # Get cost rates for the model
    costs = model_cost_dict[model_name]

    # Calculate costs
    input_cost = costs["input"] * input_tokens
    output_cost = costs["output"] * output_tokens
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": total_cost,
    }
    