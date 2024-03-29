# Databricks notebook source
# MAGIC %md
# MAGIC # Install Packages

# COMMAND ----------

# MAGIC %pip install sentence-transformers
# MAGIC %pip install pandas
# MAGIC %pip install nltk 
# MAGIC %pip install gensim
# MAGIC %pip install semantic-text-similarity
# MAGIC %pip install torch
# MAGIC %pip install numpy 
# MAGIC %pip install matplotlib 

# COMMAND ----------

# MAGIC %md
# MAGIC # Import functions 

# COMMAND ----------

import pandas as pd
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 
import torch
import os 
import numpy
from datetime import timedelta
import numpy as np 
import sys 
import base64
import json
import requests


# COMMAND ----------

path_helper = os.path.join('..','py') # import the hand written function that lives in repo under py folder 
#path = '/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/nlp.py'
sys.path.append(path_helper)
from nlp import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in data

# COMMAND ----------

df1 = pd.read_csv('/dbfs/FileStore/270324___WVA_incidents_LU.csv') # read in the WVA data 

# COMMAND ----------

df1.shape

# COMMAND ----------

df1

# COMMAND ----------

# Rename columns 
df1 = df1.rename(columns={'Data Ref Num': 'Reference_Number', 'Incident Updated Date' : 'Incident_Date'})


# COMMAND ----------

df1 = df1.dropna(how='all') # remove any rows with na's in all the columns 

# COMMAND ----------

df1['Time'] = pd.to_datetime(df1['Time']).dt.strftime('%H:%M') # standardise the time column to hours and minutes 


# COMMAND ----------

print("Incident ID:", df1['Date']) # 14 weeks worth of data 

# COMMAND ----------

# MAGIC %md
# MAGIC # Define the model  

# COMMAND ----------

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # load in the pretrained NLP model 

# COMMAND ----------

# MAGIC %md
# MAGIC # Data pre-processing 

# COMMAND ----------

# Load NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english')) # determine the stop words to be removed 
custom_stop_words = {'male', 'female'} # add these two words as well 

# Combine standard English stopwords and custom stopwords
stop_words.update(custom_stop_words)

# COMMAND ----------


# Clean and preprocess sentences using the hand written function that live in py folder
sentences_from_file1 = [clean_text(sentence, stop_words) for sentence in df1['DESCRIPTION'].astype(str).tolist()]


# COMMAND ----------

# MAGIC %md
# MAGIC # Embeds the tokens 

# COMMAND ----------

# Encode sentences to obtain embeddings which is a numerical represntation of text
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)


# COMMAND ----------

# MAGIC %md
# MAGIC # Define cosine similarity threshold 

# COMMAND ----------

# Reset the index of df1
df1 = df1.reset_index(drop=True)

# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1)

# Set similarity threshold
threshold = 0.60

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Logic to Identify Semantic Duplicates 

# COMMAND ----------

# Define a time window duration as 20 minutes
time_window_duration = timedelta(minutes=20)

# Find indices of similar reports above the threshold with the same location, same incident date, and different reference number
similar_reports_indices = [
    (i, j)
    for i in range(len(sentences_from_file1))
    for j in range(i + 1, len(sentences_from_file1))
    if (
        (similarity_matrix[i, j] > threshold) and  # Check if the similarity score is above the threshold
        ((df1['LOCATION'][i] == df1['LOCATION'][j])) and  # Check if the locations are the same
        (df1['Date'][i] == df1['Date'][j]) and  # Check if the incident dates are the same
        (abs(pd.to_datetime(str(df1['Date'][i]) + ' ' + str(df1['Time'][i])) -
             pd.to_datetime(str(df1['Date'][j]) + ' ' + str(df1['Time'][j]))) <= time_window_duration))  # Check if the time difference is less than or equal to 20 minutes 
    ]




# COMMAND ----------

df1

# COMMAND ----------

# MAGIC %md
# MAGIC # Creation of the dataframe 

# COMMAND ----------

# Create a DataFrame for similar reports with Bus_Route column
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': i,  # Index of the incident in the first file
        'File2_Index': j,  # Index of the duplicate incident in the second file
        'Incident': sentences_from_file1[i],  # Text of the incident in the first file
        'Incident_Duplicate': sentences_from_file1[j],  # Text of the duplicate incident in the second file
        'Similarity_Score': similarity_matrix[i, j],  # Similarity score between the two incidents
        'Location': df1['LOCATION'][i],  # Location of the incident in the first file
        'Date': df1['Date'][i],  # Date of the incident in the first file
        'Incident_Time': df1['Time'][i],  # Time of the incident in the first file
        'Incident_Time_Duplicate': df1['Time'][j],  # Time of the duplicate incident in the second file
        'Report_Type': 'EIRF' if df1['EIRF'][i] == 'X' else 'WAASB',  # Report type of the incident in the first file
        'Report_Type_Duplicate': 'EIRF' if df1['EIRF'][j] == 'X' else 'WAASB',  # Report type of the duplicate incident in the second file
        'Reference_Number': df1['Reference_Number'][i],  # Reference number associated with the incident in the first file
        'Reference_Number_Duplicate': df1['Reference_Number'][j]  # Reference number associated with the duplicate incident in the second file
    }
    for i, j in similar_reports_indices  # Loop through the indices of similar reports
])


# COMMAND ----------

# MAGIC %md
# MAGIC # Determine the columns of the output file 

# COMMAND ----------

similar_reports_df['Similarity_Score_Percent'] = (similar_reports_df['Similarity_Score'] * 100).round() # convert the cosine similarity into percentage 


# COMMAND ----------

# Reorder columns with the new index as the first column
similar_reports_df = similar_reports_df[['Reference_Number', 'Reference_Number_Duplicate', 'Report_Type', 'Report_Type_Duplicate','Incident_Time','Incident_Time_Duplicate', 'Incident', 'Incident_Duplicate','Location', 'Similarity_Score_Percent']]

# COMMAND ----------

similar_reports_df['Duplicate'] = '' #create an empty column called 'Duplicate'

# COMMAND ----------


# remove any entries with 'Incident' column which has just nan value
similar_reports_df= similar_reports_df[similar_reports_df['Incident'] != 'nan']


# COMMAND ----------

display(similar_reports_df)

# COMMAND ----------

similar_reports_df.to_csv('similar_reports_df', index=False)


# COMMAND ----------


# Specify the file name
csv_file_name = "similar_reports_df"

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
        "To": 'adam.mullineux@tfl.gov.uk',
        "Attachment": csv_content_string
    }

    # Send the POST request with the 'json' parameter
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    # Print the response status code
    print(response.status_code)

except Exception as e:
    print(f"An error occurred: {str(e)}")

