'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

import json
from src.bedrock import get_claude_response, get_nova_response

def extract_parameters(user_question: str, 
                       mapping_info: str, 
                       function_desc: dict,
                       function_inst_prompt: str="", 
                       model_id="nova_pro", 
                       dev_prompt_version="V1" ):
    '''
    Extract parameters based on a user's query and required information
    '''

    if str(function_desc['parameters']) == "[]":
        return "[]"
    
    if dev_prompt_version == "V1":
        inst_prompt = INSTRUCTION_PROMPT_V1.format(function_desc=json.dumps(function_desc),
                                                   function_inst_prompt=function_inst_prompt,
                                                   mapping_information=mapping_info)
    elif dev_prompt_version == "V2":
        inst_prompt = INSTRUCTION_PROMPT_V2.format(function_desc=json.dumps(function_desc), 
                                                   mapping_information=mapping_info)
    elif dev_prompt_version == "V3":    
        inst_prompt = INSTRUCTION_PROMPT_V3.format(function_desc=json.dumps(function_desc), 
                                                   mapping_information=mapping_info)
    else:
        print("version mismatch")
        return

    prompt = INPUT_PROMPT.format(instruction_prompt=inst_prompt,
                                 user_question=user_question,
                                 output_format=OUTPUT_FORMAT_PROMPT)
    if "sonnet" in model_id:
        response = get_claude_response(system_msg="",
                                       human_msg=prompt,
                                       assistant_msg="<json>",
                                       stop_sequences=["Human:", "</json>"])
    else:
        response, usage_info = get_nova_response(system_msg="",
                                                 human_msg=prompt,
                                                 assistant_msg="<json>",
                                                 stop_sequences=["Human:", "</json>"],
                                                 model_name= model_id)
    response = response.replace("</json>", "")
    response = response.strip()

    return response

#######################
#     Prompt Bank     #
#######################
INSTRUCTION_PROMPT_V1="""You are an AI Human Resources assistant designed to retrieve Healthcare Benefit Coverage information based on user queries.

Here are the available functions and their parameters in the JSON format:
{function_desc}

{function_inst_prompt}

Below is the mapping information that you must use to extract parameters:
{mapping_information}
"""

INSTRUCTION_PROMPT_V2 = """You are an AI Human Resources assistant designed to retrieve Healthcare Benefit Coverage information based on user queries.

Here are the available functions and their parameters in the JSON format:
{function_desc}

Below is the mapping information that you must use to extract parameters:
{mapping_information}
"""

INSTRUCTION_PROMPT_V3 = """You are an AI Human Resources assistant designed to retrieve required information based on user queries.

Here are the available functions and their parameters in the JSON format:
{function_desc}

Below is the mapping information that you must use to extract parameters:
{mapping_information}
"""

INPUT_PROMPT = """{instruction_prompt}

Here's an user's query:
{user_question}

Here's the output format you must follow:
{output_format}

Please extract the parameter of an appropriate function in the JSON format in the <json> tags by strictly following the above instruction. Please only use the information provided in the mapping information for extraction and always use the values in the mapping table as-is unless instructed otherwise above. When there is no specific parameters that can be matched from the mapping information to a user's query, use all values following the instruction in the function description.
"""

OUTPUT_FORMAT_PROMPT = """
{
  "arguments": {
      "[Function Parameter 1]": Parameter 1 value,
      "[Function Parameter 2]": Parameter 2 value,
      "[Function Parameter 3]": Parameter 3 value
      ...
      },
  "name": "[Function Name]"
}
"""

TEST_MAPPING_INFORMATION = """
|   standardBenefitArea   |   standardBenefitAreaName   |
|:-----------------------:|:---------------------------:|
| LTD                     | LTD                         |
| MEDICAL                 | Primary Medical             |
| DENTAL                  | Primary Dental              |
"""

TEST_FUNCTION_DESCRIPTION = """
{
  "description": "Get coverage plans for the associate",
  "name": "get_coverage_plans_for_associate",
  "parameters": {
    "properties": {
      "isSpecified": {
        "description": "This parameter determines whether the user query mentioned at least one possible names of standard benefit areas (standardBenefitAreaName).
        1. If the user query mentioned at least one standard benefit area, set this parameter to `true`. Example: \"what is my dental and medical?\", the value should be `true`.; 
        2. otherwise, set it to `false`. Example: \"what is my benefits?\", the value should be `false`.",
        "name": "isSpecified",
        "type": "boolean"
      },
      "standardBenefitAreaNames": {
        "description": "This parameter specifies the names of standard benefit areas (i.e. 'standardBenefitAreaName'). 
        The value must be formatted according to the following rules:
            1. **Single Item**: If there is only one item, use that item directly.
                - **Example**: `Dependent AD&D`
            2. **Multiple Items**: If there are multiple items, list them in a sentence, separating each item with a comma and using \"and\" before the last item.
                - **Example**: `item1, item2, and item3`
            3. **Default Value**: If no items are specified, use all the standard benefit area names available in the system prompt context.",
        "name": "standardBenefitAreaNames",
        "type": "string"
      },
      "standardBenefitAreas": {
        "description": "This parameter specifies the standard benefit areas (i.e. 'standardBenefitArea') and will be used as a field value in a Solr query. The value must be URL-encoded according to the following guidelines:
        1. **Single Item:**
            - If only one item is provided, enclose it in double quotes and then encode it.
            - **Example:**
                - Input: `\"item\"`
                - Encoded Value: `%22item%22`
        2. **Multiple Items:**
            - When multiple items are specified, follow these steps:
                - Enclose each item in double quotes.
                - Separate the items with the keyword `OR`.
                - Enclose the entire expression in parentheses.
                - Finally, encode the complete expression.
            - **Example:** 
                - Input: `(\"item1\" OR \"item2\" OR \"item3\")`
                - Encoded Value: `%28%22item1%22%20OR%20%22item2%22%20OR%20%22item3%22%29`
        3. **Default Value:**
            - If no items are specified, the default value should be an asterisk `*`.
            - **Example:** 
            - Default Encoded Value: `*`",
        "name": "standardBenefitAreas",
        "type": "string"
      }
    },
    "required": [
      "standardBenefitAreas",
      "standardBenefitAreaNames",
      "isSpecified"
    ],
    "type": "object"
  }
}
"""

TEST_FUNCTION_INSTRUCTION_PROMPT = """
When processing user queries, please adhere to the following guidelines:

1. **Named Benefit Areas**: If the query specifies benefit area names, select up to 3 closest semantic matches for each `standardBenefitArea` without seeking clarification. 
    - *Example 1*: For the query \"What is my child coverage?\", use up to 3 `standardBenefitAreaName` that includes \"child\" for `standardBenefitAreaNames` and its corresponding `standardBenefitArea` for the `standardBenefitAreas`.
    - *Example 2*: For the query \"What is my spouse coverage?\", use up to 3 `standardBenefitAreaName` that includes \"spouse\" for `standardBenefitAreaNames` and its corresponding `standardBenefitArea` for the `standardBenefitAreas`.
    - *Example 3*: For the query \"What is my dental?\", use up to 3 `standardBenefitAreaName` that includes \"dental\" (e.g., \"Dental\" and \"Additional Dental\") for `standardBenefitAreaNames` and its corresponding `standardBenefitArea` for the `standardBenefitAreas`.
    - *Example 4*: For the query \"What is my long term disability?\", use up to 3 `standardBenefitAreaName` that either includes words from \"long term disability\" or \"LTD\" (acronym) for `standardBenefitAreaNames` and its corresponding `standardBenefitArea` for the `standardBenefitAreas`.

2. **General Benefit Queries**: If the query does not specify benefit area names, utilize all available `standardBenefitAreas` without seeking clarification. 
    - *Example*: For the query \"What is my coverage?\", use all available `standardBenefitAreaName` entries.
   
3. **Standard Benefit Area Names**: In this context, \"medical\" refers to a Standard Benefit Area that includes names specifically related to medical services, such as \"Primary Medical\" or \"Medical.\" Only plans with names that contain \"Medical\" will match this category. 
For example, \"Medical\" and \"Primary Medical\" match, while \"Dental\" or \"Long-term Disability\" do not.
    - *Example*: For the query \"What is my medical benefit?\", use `standardBenefitAreaName` which has \"Medical\" in it for the `standardBenefitAreaNames` and its corresponding `standardBenefitArea` for the `standardBenefitAreas`.
4. **Parameter Usage**:
    - For `standardBenefitAreas`, use the `standardBenefitArea` code (e.g., use \"MEDICAL\" instead of \"Primary Medical\").   
    - For `standardBenefitAreaNames`, use the `standardBenefitAreaName` (e.g., use \"Primary Medical\" instead of \"MEDICAL\").
"""


ADP_EXAMPLE_PROMPT="""
You are an AI Human Resources assistant designed to retrieve Benefit Coverage information based on user queries.

When processing user queries, please adhere to the following guidelines:

1. **Named Benefit Areas**: If the query specifies benefit area names, select up to 3 closest semantic matches for each 'standardBenefitArea' without seeking clarification. 
    - *Example 1*: For the query \"What is my child coverage?\", use the up to 3 'standardBenefitAreaName' that includes \"child\" for 'standardBenefitAreaNames' and its corresponding 'standardBenefitArea' for the 'standardBenefitAreas'.
    - *Example 2*: For the query \"What is my spouse coverage?\", use the up to 3 'standardBenefitAreaName' that includes \"spouse\" for 'standardBenefitAreaNames' and its corresponding 'standardBenefitArea' for the 'standardBenefitAreas'.
    - *Example 3*: For the query \"What is my dental?\", use the up to 3 'standardBenefitAreaName' that includes \"dental\", (e.g. \"Dental\" and \"Additional Dental\") for 'standardBenefitAreaNames' and its corresponding 'standardBenefitArea' for the 'standardBenefitAreas'.
    - *Example 4*: For the query \"What is my long term disability?\", use the up to 3 'standardBenefitAreaName' that either includes words from \"long term disability\" or \"LTD\" (acronym) for 'standardBenefitAreaNames' and its corresponding 'standardBenefitArea' for the 'standardBenefitAreas'.
    
2. **General Benefit Queries**: If the query does not specify benefit area names, utilize all available 'standardBenefitAreas' without seeking clarification. 
    - *Example*: For the query \"What is my coverage?\", use all available 'standardBenefitAreaName' entries.
    
3. **Standard Benefit Area Names**: In this context, \"medical\" refers to a Standard Benefit Area that includes names specifically related to medical services, such as \"Primary Medical\" or \"Medical.\" Only plans with names that contain \"Medical\" will match this category. For example, \"Medical\" and \"Primary Medical\" match, while \"Dental\" or \"Long-term Disability\" do not. 
    - *Example*: For the query \"What is my medical benefit?\", use 'standardBenefitAreaName' which has \"Medical\" in it for the 'standardBenefitAreaNames' and its corresponding 'standardBenefitArea' for the 'standardBenefitAreas'.

4. **Parameter Usage**:
    - For 'standardBenefitAreas', use the 'standardBenefitArea' code (e.g., use \"{{output.docs.0.standardBenefitArea}}\" instead of \"{{output.docs.0.standardBenefitAreaName}}\").
    - For 'standardBenefitAreaNames', use the 'standardBenefitAreaName' (e.g., use \"{{output.docs.0.standardBenefitAreaName}}\" instead of \"{{output.docs.0.standardBenefitArea}}\").
    
Below is the mapping of 'standardBenefitArea' to 'standardBenefitAreaName':
|   standardBenefitArea   |   standardBenefitAreaName   |
|:-----------------------:|:---------------------------:|
| MEDICAL | Medical |
| DENTAL | Dental |
| VISION | Vision |
| ADD2 | Special purpose AD&D |
"""