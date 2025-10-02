## ADP Lyric Search Optimization Code base

This repo contains the code base of the ADP Lyric Search Optimization engagement executed by AWS GenAIIC.

### Modules
* ```src/bedrock.py```: Bedrock calling functions using ADP AI Gateway
    * Note: Please update CLIENT_ID, CLIENT_SECRET, SCOPE in ```the get_bearer_token``` function
* ```src/evaluation.py```: Intent classifcation search and reranking evaluation
* ```src/intent_classfication.py```: Intent classification search functions 
* ```src/prompt.py```: Prompts used for reranking
* ```src/rerank.py```: Intent classification reranking functions
* ```src/parameter_extraction.py```: Parameter extraction functions/prompts

### Demos
* ```demo-overall-flow.ipynb```: Demo of the overall flow including intent classification (search, prefiltering, reranking) and parameter extraction
* ```intent_eval_titan_v2_1024.ipynb```: Intent classification (search) with Titan V2 - 1024
* ```prefiltering-threshold-analysis.ipynb```: Prefiltering threshold analysis
* ```opensearch_build_index_test.ipynb```: Walk-through on building OpenSearch index
* ```rerank-top1-final-complete.ipynb```: Evaluation of the reranking
* ```rerank-top1-final-sample-new-prompt.ipynb```: Analysis on the reranking with a new prompt handling unrelated user queries by the request of ADP Lyric at the final presentation.
* ```script_intent_param_flow.py```: Evaluation script of the overall flow (excl. prefiltering).
* ```script_parameter_extraction_test_data.py```: Script on filtering parameter required questions from filteredQna.json.
* ```script_parameter_extraction_test.py```: Batch script on parameter extraction.

### License
© 2025 Amazon Web Services, Inc. or its affiliates. All Rights Reserved. This deliverable is considered Developed Content as defined in the AWS Service Terms and the SOW between the parties dated February 10th, 2025.