# Databricks notebook source
# MAGIC %md 
# MAGIC ### A cluster has been created for this demo
# MAGIC To run this demo, just select the cluster `dbdemos-llm-rag-chatbot-ling_huang` from the dropdown menu ([open cluster configuration](https://dbc-57a6b8b4-32b6.cloud.databricks.com/#setting/clusters/1229-161536-ch1qd3gj/configuration)). <br />
# MAGIC *Note: If the cluster was deleted after 30 days, you can re-create it with `dbdemos.create_cluster('llm-rag-chatbot')` or re-install the demo: `dbdemos.install('llm-rag-chatbot')`*

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC # 1/ Data preparation for LLM Chatbot RAG
# MAGIC
# MAGIC ## Building and indexing our knowledge base into Databricks Vector Search
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-flow-1.png?raw=true" style="float: right; width: 800px; margin-left: 10px">
# MAGIC
# MAGIC In this notebook, we'll ingest our documentation pages and index them with a Vector Search index to help our chatbot provide better answers.
# MAGIC
# MAGIC Preparing high quality data is key for your chatbot performance. We recommend taking time to implement these next steps with your own dataset.
# MAGIC
# MAGIC Thankfully, Lakehouse AI provides state of the art solutions to accelerate your AI and LLM projects, and also simplifies data ingestion and preparation at scale.
# MAGIC
# MAGIC For this example, we will use Databricks documentation from [docs.databricks.com](docs.databricks.com):
# MAGIC - Download the web pages
# MAGIC - Split the pages in small chunks of text
# MAGIC - Compute the embeddings using a Databricks Foundation model as part of our Delta Table
# MAGIC - Create a Vector Search index based on our Delta Table  
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=local&notebook=%2F01-quickstart%2F01-Data-Preparation-and-Index&demo_name=llm-rag-chatbot&event=VIEW&path=%2F_dbdemos%2Fdata-science%2Fllm-rag-chatbot%2F01-quickstart%2F01-Data-Preparation-and-Index&version=1">

# COMMAND ----------

# DBTITLE 1,Install required external libraries 
# MAGIC %pip install mlflow==2.9.0 lxml==4.9.3 transformers==4.30.2 langchain==0.0.344 databricks-vectorsearch==0.22
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Init our resources and catalog
# MAGIC %run ../_resources/00-init $reset_all_data=false

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Extracting Databricks documentation sitemap and pages
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-1.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC First, let's create our raw dataset as a Delta Lake table.
# MAGIC
# MAGIC For this demo, we will directly download a few documentation pages from `docs.databricks.com` and save the HTML content.
# MAGIC
# MAGIC Here are the main steps:
# MAGIC
# MAGIC - Run a quick script to extract the page URLs from the `sitemap.xml` file
# MAGIC - Download the web pages
# MAGIC - Use BeautifulSoup to extract the ArticleBody
# MAGIC - Save the HTML results in a Delta Lake table

# COMMAND ----------

if not spark.catalog.tableExists("raw_documentation") or spark.table("raw_documentation").isEmpty():
    # Download Databricks documentation to a DataFrame (see _resources/00-init for more details)
    doc_articles = download_databricks_documentation_articles()
    #Save them as a raw_documentation table
    doc_articles.write.mode('overwrite').saveAsTable("raw_documentation")

display(spark.table("raw_documentation").limit(2))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Splitting documentation pages into small chunks
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-2.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC LLM models typically have a maximum input context length, and you won't be able to compute embeddings for very long texts.
# MAGIC In addition, the longer your context length is, the longer it will take for the model to provide a response.
# MAGIC
# MAGIC Document preparation is key for your model to perform well, and multiple strategies exist depending on your dataset:
# MAGIC
# MAGIC - Split document into small chunks (paragraph, h2...)
# MAGIC - Truncate documents to a fixed length
# MAGIC - The chunk size depends on your content and how you'll be using it to craft your prompt. Adding multiple small doc chunks in your prompt might give different results than sending only a big one
# MAGIC - Split into big chunks and ask a model to summarize each chunk as a one-off job, for faster live inference
# MAGIC - Create multiple agents to evaluate each bigger document in parallel, and ask a final agent to craft your answer...
# MAGIC

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ### Splitting our big documentation pages in smaller chunks (h2 sections)
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/chunk-window-size.png?raw=true" style="float: right" width="700px">
# MAGIC <br/>
# MAGIC In this demo, we have big documentation articles, which are too long for the prompt to our model. 
# MAGIC
# MAGIC We won't be able to use multiple documents as RAG context as they would exceed our max input size. Some recent studies also suggest that bigger window size isn't always better, as the LLMs seem to focus on the beginning and end of your prompt.
# MAGIC
# MAGIC In our case, we'll split these articles between HTML `h2` tags, remove HTML and ensure that each chunk is less than 500 tokens using LangChain. 
# MAGIC
# MAGIC #### LLM Window size and Tokenizer
# MAGIC
# MAGIC The same sentence might return different tokens for different models. LLMs are shipped with a `Tokenizer` that you can use to count tokens for a given sentence (usually more than the number of words) (see [Hugging Face documentation](https://huggingface.co/docs/transformers/main/tokenizer_summary) or [OpenAI](https://github.com/openai/tiktoken))
# MAGIC
# MAGIC Make sure the tokenizer you'll be using here matches your model. We'll be using the `transformers` library to count llama2 tokens with its tokenizer. This will also keep our document token size below our embedding max size (1024).
# MAGIC
# MAGIC <br/>
# MAGIC <br style="clear: both">
# MAGIC <div style="background-color: #def2ff; padding: 15px;  border-radius: 30px; ">
# MAGIC   <strong>Information</strong><br/>
# MAGIC   Remember that the following steps are specific to your dataset. This is a critical part to building a successful RAG assistant.
# MAGIC   <br/> Always take time to manually review the chunks created and ensure that they make sense and contain relevant information.
# MAGIC </div>

# COMMAND ----------

# DBTITLE 1,Splitting our html pages in smaller chunks
from langchain.text_splitter import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

max_chunk_size = 500

tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(tokenizer, chunk_size=max_chunk_size, chunk_overlap=50)
html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[("h2", "header2")])

# Split on H2, but merge small h2 chunks together to avoid too small. 
def split_html_on_h2(html, min_chunk_size = 20, max_chunk_size=500):
  if not html:
      return []
  h2_chunks = html_splitter.split_text(html)
  chunks = []
  previous_chunk = ""
  # Merge chunks together to add text before h2 and avoid too small docs.
  for c in h2_chunks:
    # Concat the h2 (note: we could remove the previous chunk to avoid duplicate h2)
    content = c.metadata.get('header2', "") + "\n" + c.page_content
    if len(tokenizer.encode(previous_chunk + content)) <= max_chunk_size/2:
        previous_chunk += content + "\n"
    else:
        chunks.extend(text_splitter.split_text(previous_chunk.strip()))
        previous_chunk = content + "\n"
  if previous_chunk:
      chunks.extend(text_splitter.split_text(previous_chunk.strip()))
  # Discard too small chunks
  return [c for c in chunks if len(tokenizer.encode(c)) > min_chunk_size]

# Let's create a user-defined function (UDF) to chunk all our documents with spark
@pandas_udf("array<string>")
def parse_and_split(docs: pd.Series) -> pd.Series:
    return docs.apply(split_html_on_h2)
  
# Let's try our chunking function
html = spark.table("raw_documentation").limit(1).collect()[0]['text']
split_html_on_h2(html)

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## What's required for our Vector Search Index
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-vector-search-type.png?raw=true" style="float: right" width="800px">
# MAGIC
# MAGIC Databricks provides multiple types of vector search indexes:
# MAGIC
# MAGIC - **Managed embeddings**: you provide a text column and endpoint name and Databricks synchronizes the index with your Delta table 
# MAGIC - **Self Managed embeddings**: you compute the embeddings and save them as a field of your Delta Table, Databricks will then synchronize the index
# MAGIC - **Direct index**: when you want to use and update the index without having a Delta Table
# MAGIC
# MAGIC In this demo, we will show you how to setup a **Self-managed Embeddings** index. 
# MAGIC
# MAGIC To do so, we will have to first compute the embeddings of our chunks and save them as a Delta Lake table field as `array&ltfloat&gt`

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Introducing Databricks BGE Embeddings Foundation Model endpoints
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-4.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Foundation Models are provided by Databricks, and can be used out-of-the-box.
# MAGIC
# MAGIC Databricks supports several endpoint types to compute embeddings or evaluate a model:
# MAGIC - A **foundation model endpoint**, provided by Databricks (ex: llama2-70B, MPT...)
# MAGIC - An **external endpoint**, acting as a gateway to an external model (ex: Azure OpenAI)
# MAGIC - A **custom**, fined-tuned model hosted on Databricks model service
# MAGIC
# MAGIC Open the [Model Serving Endpoint page](/ml/endpoints) to explore and try the foundation models.
# MAGIC
# MAGIC For this demo, we will use the foundation model `BGE` (embeddings) and `llama2-70B` (chat). <br/><br/>
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/databricks-foundation-models.png?raw=true" width="600px" >

# COMMAND ----------

# DBTITLE 1,Using Databricks Foundation model BGE as an embedding endpoint
import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": ["What is Apache Spark?"]})
embeddings = [e['embedding'] for e in response.data]
print(embeddings)

# COMMAND ----------

# DBTITLE 1,Create the final databricks_documentation table containing chunks
# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS databricks_documentation (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   url STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Computing the chunk embeddings and saving them to our Delta Table
# MAGIC
# MAGIC The last step is to compute an embedding for all our documentation chunks. Let's create a udf to compute the embeddings using the foundation model endpoint.
# MAGIC
# MAGIC *Note that this part would typically be setup as a production-grade job, running as soon as a new documentation page is updated. <br/> This could be setup as a Delta Live Table pipeline to incrementally consume updates.*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    def get_embeddings(batch):
        #Note: this will gracefully fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

(spark.table("raw_documentation")
      .filter('text is not null')
      .withColumn('content', F.explode(parse_and_split('text')))
      .withColumn('embedding', get_embedding('content'))
      .drop("text")
      .write.mode('overwrite').saveAsTable("databricks_documentation"))

display(spark.table("databricks_documentation"))

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ### Our dataset is now ready! Let's create our Self-Managed Vector Search Index.
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/llm-rag-self-managed-prep-3.png?raw=true" style="float: right; width: 600px; margin-left: 10px">
# MAGIC
# MAGIC Our dataset is now ready. We chunked the documentation page into small sections, computed the embeddings and saved it as a Delta Lake table.
# MAGIC
# MAGIC Next, we'll configure Databricks Vector Search to ingest data from this table.
# MAGIC
# MAGIC Vector search index uses a Vector search endpoint to serve the embeddings (you can think about it as your Vector Search API endpoint). <br/>
# MAGIC Multiple Indexes can use the same endpoint. Let's start by creating one.

# COMMAND ----------

# DBTITLE 1,Creating the Vector Search endpoint
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()

if VECTOR_SEARCH_ENDPOINT_NAME not in [e['name'] for e in vsc.list_endpoints().get('endpoints', [])]:
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC You can view your endpoint on the [Vector Search Endpoints UI](#/setting/clusters/vector-search). Click on the endpoint name to see all indexes that are served by the endpoint.

# COMMAND ----------

# DBTITLE 1,Create the Self-managed vector search using our endpoint
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.catalog as c

#The table we'd like to index
catalog = "ling_test_demo"
db = "default"

source_table_fullname = f"{catalog}.{db}.databricks_documentation"
vs_index_fullname = f"{catalog}.{db}.databricks_documentation_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="id",
    embedding_dimension=1024, #Match your model embedding size (bge)
    embedding_vector_column="embedding"
  )
else:
  #Trigger a sync to update our vs content with the new data saved in the table
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

#Let's wait for the index to be ready and all our embeddings to be created and indexed
wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
print(f"index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Searching for similar content
# MAGIC
# MAGIC That's all we have to do. Databricks will automatically capture and synchronize new entries in your Delta Live Table.
# MAGIC
# MAGIC Note that depending on your dataset size and model size, index creation can take a few seconds to start and index your embeddings.
# MAGIC
# MAGIC Let's give it a try and search for similar content.
# MAGIC
# MAGIC *Note: `similarity_search` also support a filters parameter. This is useful to add a security layer to your RAG system: you can filter out some sensitive content based on who is doing the call (for example filter on a specific department based on the user preference).*

# COMMAND ----------

import mlflow.deployments
deploy_client = mlflow.deployments.get_deploy_client("databricks")

question = "How can I track billing usage on my workspaces?"
response = deploy_client.predict(endpoint="databricks-bge-large-en", inputs={"input": [question]})
embeddings = [e['embedding'] for e in response.data]

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_vector=embeddings[0],
  columns=["url", "content"],
  num_results=1)
docs = results.get('result', {}).get('data_array', [])
docs

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Next step: Deploy our chatbot model with RAG
# MAGIC
# MAGIC We've seen how Databricks Lakehouse AI makes it easy to ingest and prepare your documents, and deploy a Vector Search index on top of it with just a few lines of code and configuration.
# MAGIC
# MAGIC This simplifies and accelerates your data projects so that you can focus on the next step: creating your real-time chatbot endpoint with well-crafted prompt augmentation.
# MAGIC
# MAGIC Open the [02-Deploy-RAG-Chatbot-Model]($./02-Deploy-RAG-Chatbot-Model) notebook to create and deploy a chatbot endpoint.
