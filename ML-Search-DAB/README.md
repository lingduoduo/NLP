# ML Search DAB

This project implements an intelligent data processing pipeline for Lifion Search, utilizing Databricks for ETL operations and search intent detection.

The Lifion Search ETL pipeline is designed to ingest, process, and analyze data for search functionality. It incorporates advanced features such as intent detection using prefix gram models and natural language processing techniques. The pipeline is built to handle large-scale data processing tasks efficiently, leveraging Databricks' distributed computing capabilities.

Key features of this project include:
- Automated data ingestion and processing
- Advanced intent detection using multiple NLP models
- Scalable data exploration for people, action and click data
- Integration with Databricks for efficient ETL operations
- Configurable deployment for different environments (development and production)

## Prerequisites

- Brew package manager (for macOS users)
- Databricks CLI
- A Databricks workspace

## Getting started

1. Install the Databricks CLI 
   For macOS users:
   ```
   brew tap databricks/tap
   brew install databricks
   databricks -v
   ```
   For other operating systems, please refer to the Databricks CLI documentation.

2. Configure Workspace Authentication
   Set up authentication to your Databricks workspace:
    ```
    $ databricks configure
    ```

3. To deploy a development copy of this project, type:
    ```
    $ databricks bundle deploy --target dev
    ```
    (Note that "dev" is the default target, so the `--target` parameter
    is optional here.)
   Alternatively, use One Data DAB deployments via Jenkins in DIT/FIT environments. 

   
4. Similarly, to deploy a production copy, type:
   ```
   $ databricks bundle deploy --target prod
   ```
   Alternatively, use One Data DAB deployments via Jenkins in PROD environments. 

### Project Structure
```
.
├── databricks.yml                  # Configuration file for Databricks asset bundles, defines jobs, clusters, and resources
├── Jenkinsfile                     # Jenkins pipeline script for automating CI/CD workflows and deployments
├── project.json                    # Contains project metadata, dependencies, and configuration settings
├── README.md                       # Main project documentation and usage instructions
├── resources
│   └── data_pipleline_jobs.yml     # YAML definitions for Databricks job scheduling and orchestration
└── src
   ├── sdq
   │   ├── data
   │   │   └── sdq_test_data_2025-02-14.jsonl   # Example JSONL file for SDQ module validation
   │   ├── data_ingestion.ipynb                 # Jupyter notebook for ingesting SDQ data
   │   └── data_process.sql                     # SQL script for transforming and cleaning SDQ data
   └── search
      ├── data
      │   ├── action_data.txt                   # Reference file listing possible user actions
      │   ├── common_action_typos.txt           # List of common typos in action data for normalization
      │   ├── people_data.txt                   # Reference file with sample people/entity data
      │   └── position_data.txt                 # Reference file with position/job titles for search
      ├── data_exploration
      │   ├── action.ipynb                      # Notebook for exploring and analyzing action data
      │   ├── click.ipynb                       # Notebook for clickstream data exploration
      │   └── people.ipynb                      # Notebook for exploring people/entity data
      ├── data_ingestion
      │   ├── config.yaml                       # Configuration file for data ingestion parameters
      │   ├── data_preparation.ipynb            # Notebook for preparing and cleaning ingested data
      │   ├── data_process.sql                  # SQL script for transforming ingested data
      │   └── weekly_ingestion.ipynb            # Notebook for automating weekly data ingestion tasks
      ├── data_profiling
      │   ├── action_query.ipynb                # Notebook for profiling and querying action data
      │   └── people_query.ipynb                # Notebook for profiling and querying people data
      ├── intent_detector_model
      │   ├── __init__.py                       # Package initializer for intent detector module
      │   ├── en_core_web_sm-3.7.1.tar.gz       # English spaCy model package for NLP tasks
      │   ├── intentDetector.py                 # Main Python module for intent detection logic
      │   └── xx_ent_wiki_sm-3.7.0.tar.gz       # Multilingual spaCy model package for entity recognition
      ├── modeling_tasks
      │   ├── action_link_correlation.ipynb     # Notebook analyzing correlations between actions and links
      │   ├── cold_start.ipynb                  # Notebook addressing cold start problems in modeling
      │   └── intent_classifier.ipynb           # Notebook for training and evaluating intent classification models
      └── utils
         ├── __init__.py                        # Package initializer for utility 
         init
         │   └── data_profiling.cpython-311.pyc # Compiled bytecode for data profiling utilities
         ├── ADP_Internal_Root_CA_GN2.pem       # Internal root CA certificate for secure connections
         ├── data_exploration.py                # Python script for data exploration utilities
         ├── data_profiling_llm.py              # Utility functions for profiling data using LLMs
         └── data_profiling_nlp.py              # Utility functions for NLP-based data profiling
```

### Environment Configurations

- Development (dev):
  - Workspace: https://adpdc-share1-dev.cloud.databricks.com
  - Secret Scope: dit-nas-lifion_ml-sdq
  - Catalog: onedata_us_east_1_shared_dit
  - Database: nas_raw_lyric_search_dit

Note that the data also populated to https://adpdc-onedata-share1-dev-us-east.cloud.databricks.com.

- Production (prod):
  - Workspace: https://adpdc-share1-prod.cloud.databricks.com
  - Secret Scope: prod-nas-lifion_ml-sdq
  - Catalog: onedata_us_east_1_shared_prod
  - Database: nas_raw_lyric_search_prod

Similarly, the data also populated to https://adpdc-onedata-share1-prod-us-east.cloud.databricks.com.

### Running the Pipeline

To run the data pipeline:

1. Ensure you're connected to the appropriate Databricks workspace.
2. Navigate to the Databricks Jobs UI.
3. Locate the related data job.
4. Click "Run Now" to execute the pipeline manually, or review the schedule settings for automated runs.

The Lifion Search ETL pipeline processes data through several stages, ensuring efficient and accurate intent detection for search queries.

### Pipeline Stages

1. **Data Ingestion:** External data sources are imported into the Databricks environment.
2. **Data Processing:** Raw data is transformed using SQL UDFs defined in `data_process.sql`.
3. **Data Preparation:** The processed data is cleaned and structured for downstream analysis and model training.
4. **Data Exploration:** Exploratory analysis is performed on people, action, and click datasets to gain insights and validate data quality.
5. **Data Profiling:** This module creates profiling reports for action and people queries, including summary statistics, data distributions, and anomaly detection to ensure data integrity and readiness for modeling.
6. **Modeling Tasks:** Machine learning models are developed and evaluated using the prepared and profiled data, supporting intent detection, cold start recommendation and other advanced analytics.


## Documentation References

For Databricks asset bundles documentation and CI/CD configuration: 

https://docs.databricks.com/dev-tools/bundles/index.html
