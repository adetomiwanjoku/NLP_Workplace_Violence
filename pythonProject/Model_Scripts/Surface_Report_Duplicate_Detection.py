# Databricks notebook source
# MAGIC %md
# MAGIC # Install Packages

# COMMAND ----------

# MAGIC %pip install sentence-transformers
# MAGIC %pip install pandas
# MAGIC %pip install nltk 
# MAGIC %pip install gensim
# MAGIC %pip install semantic-text-similarity
# MAGIC %pip install numpy 
# MAGIC %pip install matplotlib 
# MAGIC %pip install keplergl==0.2.2 
# MAGIC %pip install powerbiclient

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Packages 

# COMMAND ----------

import sys
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
from powerbiclient import Report

# COMMAND ----------

path_helper = os.path.join('..','py') # import the hand written function that lives in repo under py folder 
#path = '/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/nlp.py'
sys.path.append(path_helper)
from nlp import *

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in the data

# COMMAND ----------

df1 = pd.read_csv('/dbfs/FileStore/April_2023_Jan_2024_Surface_Data.csv') # read in the WVA file 

# COMMAND ----------

display(df1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Data Renaming 

# COMMAND ----------

df1.shape

# COMMAND ----------

df1 = df1.rename(columns={'Location / Road Name': 'Location'})

# COMMAND ----------

# Assuming your DataFrame is similar_reports_df
df1 = df1.rename(columns={'Bus Route ': 'Bus_Route','Incident Date': 'Incident_Date'})

# COMMAND ----------

df1['Location'] = df1['Location'].str.title() # Useful as this station names are written each word is capatalised

# COMMAND ----------

# MAGIC %md
# MAGIC # Load model and tokenizer

# COMMAND ----------

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2') # load in the pre-trained nlp model 

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Preprocessing 

# COMMAND ----------

stop_words = set(stopwords.words('english')) # define the stop words to be removed 
custom_stop_words = {'male', 'female'} # add these two words as well 

# Combine standard English stopwords and custom stopwords
stop_words.update(custom_stop_words)

# COMMAND ----------

# Clean and preprocess sentences check the handwritten function in the repo in the py folder 
sentences_from_file1 = [clean_text(sentence, stop_words=stop_words) for sentence in df1['Description'].astype(str).tolist()]

# COMMAND ----------

# MAGIC %md
# MAGIC # Embed the words 

# COMMAND ----------

# Encode sentences to obtain embeddings which give words numerical representations 
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define cosine similarity 

# COMMAND ----------


# Reset the index of df1
df1 = df1.reset_index(drop=True)

similarity_matrix = cosine_similarity(embeddings1) # mathematical technique which calculates the distances of the embeddings 

# Set similarity threshold
threshold = 0.55

# COMMAND ----------

# MAGIC %md
# MAGIC # Model Logic to Identify Semantic Duplicates 

# COMMAND ----------


# Define a time window duration of 90 minutes
time_window_duration = timedelta(minutes=90)

# Initialize a list to store indices of similar reports
similar_reports_indices = [
    # Nested loop to compare each pair of sentences
    (i, j)
    for i in range(len(sentences_from_file1))
    for j in range(i + 1, len(sentences_from_file1))
    if (
        # Check if the similarity between sentences is above a certain threshold
        (similarity_matrix[i, j] > threshold) and
        # Check if the reports are related by either Bus Route or Location
        ((df1['Bus Route'][i] == df1['Bus Route'][j]) or (df1['Location'][i] == df1['Location'][j])) and
        # Check if the reports have the same Incident Date
        (df1['Incident_Date'][i] == df1['Incident_Date'][j]) and 
        # Check if the reports have different URN (Unique Reference Numbers)
        (df1['URN'][i] != df1['URN'][j]) and
        # Check if the reports are of opposite directions (DIR and IRIS)
        ((df1['DIR'][i] == 'DIR' and df1['IRIS'][j] == 'IRIS') or
         (df1['IRIS'][i] == 'IRIS' and df1['DIR'][j] == 'DIR')) and
        # Check if the time difference between reports is within the defined time window
        (abs(pd.to_datetime(str(df1['Incident_Date'][i]) + ' ' + str(df1['Time'][i])) -
             pd.to_datetime(str(df1['Incident_Date'][j]) + ' ' + str(df1['Time'][j]))) <= time_window_duration)
    )
]



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
        'Location': df1['Location'][i],  # Location of the incident in the first file
        'Bus_Route': df1['Bus Route'][i],  # Bus route associated with the incident in the first file
        'Date': df1['Incident_Date'][i],  # Date of the incident in the first file
        'Incident_Time': df1['Time'][i],  # Time of the incident in the first file
        'Incident_Time_Duplicate': df1['Time'][j],  # Time of the duplicate incident in the second file
        'Report_Type': 'DIR' if df1['DIR'][i] == 'DIR' else 'IRIS' if df1['IRIS'][i] == 'IRIS' else None,  # Report type of the incident in the first file
        'Report_Type_Duplicate': 'DIR' if df1['DIR'][j] == 'DIR' else 'IRIS' if df1['IRIS'][j] == 'IRIS' else None,  # Report type of the duplicate incident in the second file
        'URN': df1['URN'][i],  # URN number associated with the incident in the first file
        'URN_Duplicate' : df1['URN'][j]  # URN number associated with the duplicate incident in the second file
    }
    for i, j in similar_reports_indices  # Loop through the indices of similar reports
])


# COMMAND ----------

display(similar_reports_df)

# COMMAND ----------

similar_reports_df['Similarity_Score_Percent'] = (similar_reports_df['Similarity_Score'] * 100).round() #make the similarity score a percentage

# COMMAND ----------

# Add an empty column called 'is_duplicate'
similar_reports_df['Is_Duplicate'] = ''

# COMMAND ----------

# MAGIC %md
# MAGIC # Define the columns of the Dataframe 

# COMMAND ----------

# Reorder columns with the new index as the first column
similar_reports_df = similar_reports_df[[ 'URN', 'URN_Duplicate', 'Report_Type', 'Report_Type_Duplicate','Incident_Time','Incident_Time_Duplicate' ,'Incident', 'Incident_Duplicate','Similarity_Score_Percent', 'Location', 'Bus_Route', 'Is_Duplicate']]

# COMMAND ----------

similar_reports_df

# COMMAND ----------

# Save the DataFrame as a CSV file
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
        "To": 'adetomiwanjoku@tfl.gov.uk',
        "Attachment": csv_content_string
    }

    # Send the POST request with the 'json' parameter
    response = requests.post(url, data=json.dumps(data), headers={'Content-Type': 'application/json'})

    # Print the response status code
    print(response.status_code)

except Exception as e:
    print(f"An error occurred: {str(e)}")
