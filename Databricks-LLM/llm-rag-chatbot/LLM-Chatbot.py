# Databricks notebook source
# MAGIC %pip install dbdemos

# COMMAND ----------

import dbdemos
dbdemos.install('llm-rag-chatbot')

# COMMAND ----------

# MAGIC %pip install "git+https://github.com/mlflow/mlflow.git@gateway-migration" langchain==0.0.344 databricks-vectorsearch==0.22 databricks-sdk==0.12.0 mlflow[databricks]
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ./_resources/00-init $reset_all_data=false

# COMMAND ----------


