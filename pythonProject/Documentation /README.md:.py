# Databricks notebook source
# MAGIC %md
# MAGIC This is the first document anyone accessing your repository will see. It should provide a high-level overview of your project, including its purpose, goals, structure, and instructions for getting started.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Project Overview
# MAGIC
# MAGIC #### Background
# MAGIC Workplace violence presents a significant challenge for operational staff, particularly within Transport for London (TFL). Compliance Policing Operational Security (CPOS) recognizes the importance of understanding the frequency and nature of these incidents. However, the existence of multiple reporting tools introduces complexities, leading to potential duplication of incident records and inflated reporting figures.
# MAGIC
# MAGIC #### Aim
# MAGIC This project leverages Natural Language Processing (NLP) techniques to systematically identify semantic duplicates across reporting tools used for documenting workplace violence incidents. The primary objective is to provide an accurate account of workplace violence reports while reducing the burden on the Workplace Violence Aggression (WVA) team.
# MAGIC
# MAGIC #### Key Requirements
# MAGIC 1. **Semantic Similarity Identification**: Develop an NLP model capable of discerning semantic similarities across incident reports to capture duplicates with varied phrasing.
# MAGIC    
# MAGIC 2. **Contextual Understanding**: Ensure the model comprehends the context and essence of each incident report, going beyond surface-level analysis.
# MAGIC    
# MAGIC 3. **Effective Duplicate Identification**: Accurately identify and flag semantically similar incidents to streamline reporting processes.
# MAGIC
# MAGIC ### Methodology
# MAGIC
# MAGIC This project utilizes Natural Language Processing techniques, particularly focusing on semantic processing, to understand human languages effectively. Python, with its versatility and rich libraries, serves as the implementation environment. The Sentence Transformer technique, inspired by BERT, is employed to transform text into fixed-dimensional vectors, facilitating semantic analysis. 
# MAGIC
# MAGIC To quantify similarity between text passages, cosine similarity is utilized, complemented by contextual features such as time, date, and location. This holistic approach provides a comprehensive view of incidents, enhancing duplicate detection accuracy.
# MAGIC
# MAGIC ### Results
# MAGIC
# MAGIC The model successfully identified duplicate incidents across reporting tools, demonstrating its ability to capture semantic similarities despite variations in sentence length or structure. 
# MAGIC
# MAGIC - **Surface Staff Reports**: Upon analyzing 18,638 reports, the model identified 206 duplicate pairs, as validated by James White and Christopher Pearson from the WVA team. This indicates the model's effectiveness in capturing incidents across different reporting tools.
# MAGIC
# MAGIC - **London Underground Reports**: In the context of London Underground workplace violence reports, the model retrieved 46 duplicate pairs, with an impressive 80% being true positive pairs. 
# MAGIC
# MAGIC ### Next Steps
# MAGIC
# MAGIC #### Automated Workflow Enhancement Proposal
# MAGIC
# MAGIC **Objective:** Streamline and automate the duplicate detection process, reducing human intervention and improving efficiency.
# MAGIC
# MAGIC 1. **Data Retrieval**
# MAGIC    - Direct connectivity to source databases for real-time data retrieval.
# MAGIC    - Leveraging Databricks connectors or APIs for seamless ingestion.
# MAGIC
# MAGIC 2. **Notebook Automation** - Completed using Databricks In-Built Function 
# MAGIC    - Implementing automated execution schedules using Databricks Jobs or Apache Airflow.
# MAGIC    - Ensuring data freshness by fetching the latest data upon execution.
# MAGIC
# MAGIC 3. **Email Automation** - Completed Using A Logic App 
# MAGIC    - Completed email automation using a logic app for seamless delivery of output files.
# MAGIC    - Triggering emails with output files directly to the WVA team.
# MAGIC
# MAGIC 4. **Scheduled Execution**
# MAGIC    - Setting up recurring schedules for notebook execution.
# MAGIC    - Utilizing Databricks built-in functions for scheduled runs.
# MAGIC
# MAGIC 5. **Documentation and Training**
# MAGIC    - Providing comprehensive documentation on workflow steps.
# MAGIC    - Conducting training sessions on Git version control practices.
# MAGIC
# MAGIC By automating and enhancing the workflow, the proposal aims to improve overall efficiency, reduce dependencies, and facilitate a seamless duplicate detection system for the WVA team.

# COMMAND ----------


