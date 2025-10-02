'''
Copyright © Amazon.com and Affiliates: This deliverable is considered Developed Content as defined in the AWS Service Terms 
and the SOW between the parties dated February 10th, 2025.
'''

rerank_prompt_claude = """
<task>
Identify the most relevant search result that best matches the intent of the query, or return -1 if no relevant matches are found.
</task>
 
<instructions>
1. Exact Match Analysis (Highest Priority):
- Check if the query appears word-for-word in the result's example questions
- Match between query keywords and the result's primary topic description
- Look for exact matches in the intent description
 
2. Topic Focus:
- Whether the result is dedicated to answering this specific type of question
- Whether the query topic is the main focus vs. being a secondary topic
- How directly the result addresses the query subject
 
3. Example Questions Alignment:
- How closely the example questions match the query pattern
- Whether the example questions cover the same information type
- Whether the examples suggest the result can provide the specific information needed
 
4. No Match Criteria (-1 should be returned):
- If the query topic is out-of-scope (e.g. what's the weather like today) or completely irrelevant/unrelated to all available results
- If the intent cannot be satisfied by any of the search results

5. You must return either:
- The index of the most relevant search result from 0 to {max_idx}
- -1 if no results are sufficiently relevant to the query
</instructions>
 
<query>
{query}
</query>
 
<search_results>
{formatted_results}
</search_results>
 
<output_format>
Return only the index (from 0 to {max_idx}) of the best result, or -1 if no relevant matches found.
Example: 2
Example: -1
</output_format>
 
Think step-by-step before returning ONLY the index without any explanation:
"""


rerank_prompt_nova = """
##task##
Identify the most relevant search result that best matches the intent of the query, or return -1 if no relevant matches are found.

##instructions##
1. Exact Match Analysis (Highest Priority):
- Check if the query appears word-for-word in the result's example questions
- Match between query keywords and the result's primary topic description
- Look for exact matches in the intent description
 
2. Topic Focus:
- Whether the result is dedicated to answering this specific type of question
- Whether the query topic is the main focus vs. being a secondary topic
- How directly the result addresses the query subject
 
3. Example Questions Alignment:
- How closely the example questions match the query pattern
- Whether the example questions cover the same information type
- Whether the examples suggest the result can provide the specific information needed
 
4. No Match Criteria (-1 should be returned):
- If the query topic is out-of-scope (e.g. what's the weather like today) or completely irrelevant/unrelated to all available results
- If the intent cannot be satisfied by any of the search results

5. You must return either:
- The index of the most relevant search result from 0 to {max_idx}
- -1 if no results are sufficiently relevant to the query

##query##
{query}

##search_results##
{formatted_results}

##output_format##
Return only the index (from 0 to {max_idx}) of the best result, or -1 if no relevant matches found.
Example: 2
Example: -1

Think step-by-step before returning ONLY the index without any new lines, explanation, or additional text:
"""