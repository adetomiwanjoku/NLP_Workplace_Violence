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

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in data

# COMMAND ----------

df1 = pd.read_csv('/dbfs/FileStore/London_Underground_Workplace_Violence_Incidents.csv')

# COMMAND ----------

df1.shape

# COMMAND ----------

df1

# COMMAND ----------

# Assuming your DataFrame is similar_reports_df
df1 = df1.rename(columns={'Data Ref Num': 'Reference_Number', 'Incident Updated Date' : 'Incident_Date'})


# COMMAND ----------

df1 = df1.dropna(how='all')

# COMMAND ----------

df1['Time'] = pd.to_datetime(df1['Time']).dt.strftime('%H:%M')


# COMMAND ----------

print("Incident ID:", df1['Date']) # 14 weeks worth of data 

# COMMAND ----------

# MAGIC %md
# MAGIC # Define the model  

# COMMAND ----------

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# COMMAND ----------

# MAGIC %md
# MAGIC # Data pre-processing 

# COMMAND ----------

# Load NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
custom_stop_words = {'male', 'female'}

# Combine standard English stopwords and custom stopwords
stop_words.update(custom_stop_words)

# COMMAND ----------

def clean_text(text):
    # Handle NaN values by replacing them with an empty string
    text = str(text) if not pd.isnull(text) else ''
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords and custom words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a cleaned text
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

# COMMAND ----------


# Clean and preprocess sentences
sentences_from_file1 = [clean_text(sentence) for sentence in df1['DESCRIPTION'].astype(str).tolist()]


# COMMAND ----------

# MAGIC %md
# MAGIC # Embeds the tokens 

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)


# COMMAND ----------

embeddings1 

# COMMAND ----------

# Define cosine similarity threshold 

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

time_window_duration = timedelta(minutes=20)
# Find indices of similar reports above the threshold with the same location, same incident date and different reference number
similar_reports_indices = [
    (i, j)
    for i in range(len(sentences_from_file1))
    for j in range(i + 1, len(sentences_from_file1))
    if (
        (similarity_matrix[i, j] > threshold) and
        ((df1['LOCATION'][i] == df1['LOCATION'][j])) and
        (df1['Date'][i] == df1['Date'][j]) and 
            (abs(pd.to_datetime(str(df1['Date'][i]) + ' ' + str(df1['Time'][i])) -
                 pd.to_datetime(str(df1['Date'][j]) + ' ' + str(df1['Time'][j]))) <= time_window_duration)
        )
    ]



# COMMAND ----------

# MAGIC %md
# MAGIC # Creation of the dataframe 

# COMMAND ----------


# Create a DataFrame for similar reports with Bus_Route column
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': i,
        'File2_Index': j,
        'Incident': sentences_from_file1[i],
        'Incident_Duplicate': sentences_from_file1[j],
        'Similarity_Score': similarity_matrix[i, j],
        'Location': df1['LOCATION'][i],
        'Date': df1['Incident_Date'][i],
        'Incident_Time': df1['Time'][i],
        'Incident_Time_Duplicate': df1['Time'][j],
        'Report_Type': 'EIRF' if df1['EIRF'][i] == 'X' else 'WAASB',
        'Report_Type_Duplicate': 'EIRF' if df1['EIRF'][j] == 'X' else 'WAASB',
        'Reference_Number': df1['Reference_Number'][i],  # Add URN number
        'Reference_Number_Duplicate': df1['Reference_Number'][j]
    }
    for i, j in similar_reports_indices
])





# COMMAND ----------

# MAGIC %md
# MAGIC # Determine the columns of the output file 

# COMMAND ----------

similar_reports_df['Similarity_Score_Percent'] = (similar_reports_df['Similarity_Score'] * 100).round()


# COMMAND ----------

# Reorder columns with the new index as the first column
similar_reports_df = similar_reports_df[['Reference_Number', 'Reference_Number_Duplicate', 'Report_Type', 'Report_Type_Duplicate','Incident_Time','Incident_Time_Duplicate', 'Incident', 'Incident_Duplicate','Location', 'Similarity_Score_Percent']]

# COMMAND ----------

similar_reports_df['Duplicate'] = ''

# COMMAND ----------

display(similar_reports_df)
