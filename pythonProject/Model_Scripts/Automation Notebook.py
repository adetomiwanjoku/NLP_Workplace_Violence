# Databricks notebook source
import pandas as pd
import requests
import json
import base64

# COMMAND ----------

# Your Logic App HTTP trigger URL
url = "https://prod-08.northeurope.logic.azure.com:443/workflows/0613a91a9f66435a9c0d3f96ba989a61/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=6slbcxFusde4k0aa9J4ccZBb-bA6PzzF3zXrtHCKFyw"


# COMMAND ----------

# Read your CSV file and encode it in base64
with open("/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/trial.csv", "rb") as file:
    csv_content_base64 = base64.b64encode(file.read()).decode("utf-8")


# COMMAND ----------


# Prepare the JSON payload
data = {
    "receiver": ["AdetomiwaNjoku@tfl.gov.uk", "AdetomiwaNjoku@tfl.gov.uk"],
    "subject": "Your Email Subject",
    "message": "Your email body message",
    }



# COMMAND ----------

# Send the POST request
response = requests.post(url, json=data)

# Print the status code of the response
print(response.status_code)

# COMMAND ----------

import requests
import json
import base64

# Your Logic App HTTP trigger URL
url = 'https://prod-08.northeurope.logic.azure.com:443/workflows/0613a91a9f66435a9c0d3f96ba989a61/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=6slbcxFusde4k0aa9J4ccZBb-bA6PzzF3zXrtHCKFyw'

# Read your CSV file and encode it in base64
with open("/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/trial.csv", 'rb') as file:
    csv_content_base64 = base64.b64encode(file.read()).decode('utf-8')

# Prepare the JSON payload
data = {
    "receiver": ["adetomiwanjoku@tfl.gov.uk", "adetomiwanjoku@tfl.gov.uk"],
    "subject": "Your Email Subject",
    "message": "Your email body message",
    "csvFileContent": csv_content_base64
}

# Send the POST request
response = requests.post(url, json=data)

# Check response
if response.status_code == 200:
    print("Data sent successfully")
else:
    print("Failed to send data")
