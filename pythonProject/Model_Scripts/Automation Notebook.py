# Databricks notebook source
import pandas as pd
import requests
import json
import base64

# COMMAND ----------

import requests

# Your Logic App HTTP trigger URL
url = "https://prod-21.northeurope.logic.azure.com:443/workflows/735927b7438049a4adeb6dbf350e7baa/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=57blbS72UgObJ2IK-pkaB22RgJMnxzujLoGWUFoAdMQ"

# Prepare the JSON payload without attachment
data = {
    "To": ["adetomiwanjoku@tfl.gov.uk", "adetomiwanjoku@tfl.gov.uk"],
    "Subject": "Your Email Subject",
    "Message": "Your email body message",
}

# Send the POST request
response = requests.post(url, data=data)

print(response.status_code)

