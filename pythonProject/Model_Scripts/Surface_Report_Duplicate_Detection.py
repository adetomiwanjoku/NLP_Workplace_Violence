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

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Packages 

# COMMAND ----------

dbutils.library.restartPython()

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
# MAGIC # Read in the data

# COMMAND ----------

df1 =  pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/April_2023_Jan_2024_Surface_Data.csv', low_memory=False)

# COMMAND ----------

df1 = df1.rename(columns={'Location / Road Name': 'Location'})

# COMMAND ----------


# Calculate the percentage of non-null values in the column in one step
percentage_non_null = (df1['Location'].count() / len(df1['Location'])) * 100

print(f"The percentage of non-null values in the column is: {percentage_non_null:.2f}%")

# COMMAND ----------


# Calculate the percentage of non-null values in the column in one step
percentage_non_null = (df1['Bus Route'].count() / len(df1['Bus Route'])) * 100

print(f"The percentage of non-null values in the column is: {percentage_non_null:.2f}%")

# COMMAND ----------

# Assuming your DataFrame is similar_reports_df
df1 = df1.rename(columns={'Bus Route ': 'Bus_Route','Incident Date': 'Incident_Date'})

# COMMAND ----------

df1['Location'] = df1['Location'].str.title() # Useful as this station names are written each word is capatalised

# COMMAND ----------

df1.shape

# COMMAND ----------

df1.head()

# COMMAND ----------

# MAGIC %md
# MAGIC # Load model and tokenizer

# COMMAND ----------

#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# COMMAND ----------

# MAGIC %md
# MAGIC # Text Preprocessing 

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
sentences_from_file1 = [clean_text(sentence) for sentence in df1['Description'].astype(str).tolist()]

# COMMAND ----------

# MAGIC %md
# MAGIC # Embed the words 

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define cosine similarity 

# COMMAND ----------


# Assuming you have the necessary imports and data loaded before this code snippet
import datetime

# Reset the index of df1
df1 = df1.reset_index(drop=True)

similarity_matrix = cosine_similarity(embeddings1)

# Set similarity threshold
threshold = 0.55

time_window_duration = timedelta(minutes=90)


similar_reports_indices = [
        (i, j)
        for i in range(len(sentences_from_file1))
        for j in range(i + 1, len(sentences_from_file1))
        if (
            (similarity_matrix[i, j] > threshold) and
            ((df1['Bus Route'][i] == df1['Bus Route'][j]) or (df1['Location'][i] == df1['Location'][j])) and
            (df1['Incident_Date'][i] == df1['Incident_Date'][j]) and 
            (df1['URN'][i] != df1['URN'][j]) and
            ((df1['DIR'][i] == 'DIR' and df1['IRIS'][j] == 'IRIS') or
             (df1['IRIS'][i] == 'IRIS' and df1['DIR'][j] == 'DIR')) and
            (abs(pd.to_datetime(str(df1['Incident_Date'][i]) + ' ' + str(df1['Time'][i])) -
                 pd.to_datetime(str(df1['Incident_Date'][j]) + ' ' + str(df1['Time'][j]))) <= time_window_duration)
        )
    ]


# Create a DataFrame for similar reports with Bus_Route column
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': i,
        'File2_Index': j,
        'Incident': sentences_from_file1[i],
        'Incident_Duplicate': sentences_from_file1[j],
        'Similarity_Score': similarity_matrix[i, j],
        'Location': df1['Location'][i],
        'Bus_Route': df1['Bus Route'][i],
        'Date': df1['Incident_Date'][i],
        'Incident_Time': df1['Time'][i],  # Include the Time column in the output DataFrame
        'Incident_Time_Duplicate' : df1['Time'][j]
    }
    for i, j in similar_reports_indices
])

# Print the resulting DataFrame
print(similar_reports_df)


# Add new index columns for each file index that adds two to the existing index
similar_reports_df['Row_Num'] = similar_reports_df['File1_Index'] + 2
similar_reports_df['Row_Num_Duplicate'] = similar_reports_df['File2_Index'] + 2





# COMMAND ----------

display(similar_reports_df)

# COMMAND ----------

similar_reports_df['Similarity_Score_Percent'] = (similar_reports_df['Similarity_Score'] * 100).round()

# COMMAND ----------

# Add an empty column called 'is_duplicate'
similar_reports_df['Is_Duplicate'] = ''

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a data frame output 

# COMMAND ----------

# Reorder columns with the new index as the first column
similar_reports_df = similar_reports_df[['Incident_Time','Incident_Time_Duplicate' ,'Row_Num', 'Row_Num_Duplicate','Incident', 'Incident_Duplicate','Similarity_Score_Percent', 'Location', 'Bus_Route', 'Is_Duplicate']]

# COMMAND ----------



# Assuming you have a dataframe named 'df' and a column named 'description'
# Replace 'your_description' with the actual description you want to remove
description_to_remove = '[ blank ]'

# Filter rows based on the condition
similar_reports_df = similar_reports_df[similar_reports_df['Description'] != description_to_remove]

similar_reports_df.drop(similar_reports_df[similar_reports_df['Description'] == description_to_remove].index, inplace=True)


# COMMAND ----------

display(similar_reports_df)

