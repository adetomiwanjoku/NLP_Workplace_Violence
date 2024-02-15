# Databricks notebook source
import pandas as pd
import requests
import json
import base64

# COMMAND ----------

import base64
import json
import requests

# Specify the file name
csv_file_name = "trial.csv"

# Your Logic App HTTP trigger URL
url = "https://prod-21.northeurope.logic.azure.com:443/workflows/735927b7438049a4adeb6dbf350e7baa/triggers/manual/paths/invoke?api-version=2016-10-01&sp=%2Ftriggers%2Fmanual%2Frun&sv=1.0&sig=57blbS72UgObJ2IK-pkaB22RgJMnxzujLoGWUFoAdMQ"

try:
    # Read the content of the CSV file in binary mode
    with open(csv_file_name, "rb") as file:
        # Decode the CSV content to a string
        csv_content_string = file.read().decode("utf-8")

    # Prepare the JSON payload with decoded CSV content as a string
    data = {
        "Message": "Good morning, this is the duplicate list the model has come up with.",
        "Subject": "Duplicate Detection Model Results",
        "To": 'adetomiwanjoku@tfl.gov.uk',
        "Attachment": csv_content_string
    }

    # Send the POST request with the 'json' parameter
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    # Print the response status code
    print(response.status_code)

except Exception as e:
    print(f"An error occurred: {str(e)}")




# COMMAND ----------

headers={'Content-Type': 'application/json'}
