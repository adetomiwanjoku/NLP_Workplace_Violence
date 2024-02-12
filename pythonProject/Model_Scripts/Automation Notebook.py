# Databricks notebook source
import pandas as pd
import requests
import json
import base64

# COMMAND ----------


# Your Logic App HTTP trigger URL
url = "https://prod-21.northeurope.logic.azure.com:443/workflows/735927b7438049a4adeb6dbf350e7baa/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=57blbS72UgObJ2IK-pkaB22RgJMnxzujLoGWUFoAdMQ"

with open("trial.csv", "rb") as file:
    csv_content_base64 = base64.b64encode(file.read()).decode("utf-8")

# Prepare the JSON payload without attachment
data = {
    "Message": "Good morning this is the duplicate list the model has come up with.",
    "Subject": "Duplicate Detection Model Results",
    "To": 'adetomiwanjoku@tfl.gov.uk',
    "Attachment": csv_content_base64}

# Send the POST request with the 'json' parameter
response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

print(response.status_code)


# COMMAND ----------


