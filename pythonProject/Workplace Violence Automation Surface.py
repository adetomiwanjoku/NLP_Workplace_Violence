# Databricks notebook source
# MAGIC %pip install sentence-transformers
# MAGIC %pip install pandas
# MAGIC %pip install nltk 
# MAGIC %pip install gensim
# MAGIC %pip install semantic-text-similarity
# MAGIC %pip install torch

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

# COMMAND ----------


nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add('driver')  # Add 'driver' to stop words

# COMMAND ----------

DIR_df = pd.read_csv('20231010_DIRsampleData.csv', usecols=['DETAILS', 'BOROUGH'], encoding='latin1')

# COMMAND ----------

IRIS_df = pd.read_csv('20231010_IRISsampleData.csv', encoding='latin1')

# COMMAND ----------

DIR_df = DIR_df.rename(columns={'BOROUGH': 'Borough', 'DETAILS': 'Description'})

# COMMAND ----------

IRIS_df = IRIS_df.rename(columns={'ï»¿Boroughs': 'Borough'})

# COMMAND ----------

DIR_df

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# COMMAND ----------

nltk.download('punkt')

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
  

  
# Clean and preprocess sentences
sentences_from_file1 = [clean_text(sentence) for sentence in DIR_df['Description'].astype(str).tolist()]
sentences_from_file2 = [clean_text(sentence) for sentence in IRIS_df['Description'].astype(str).tolist()]

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)
embeddings2 = model.encode(sentences_from_file2, convert_to_tensor=True)

# COMMAND ----------

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have 'Location' column in both dataframes
locations_from_file1 = DIR_df['Borough'].tolist()
locations_from_file2 = IRIS_df['Borough'].tolist()

# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# Set similarity threshold
threshold = 0.8

# Find indices of similar reports above the threshold with the same location
similar_reports_indices = [(i, j) for i in range(len(sentences_from_file1)) for j in range(len(sentences_from_file2))
                            if similarity_matrix[i, j] > threshold and locations_from_file1[i] == locations_from_file2[j]]

# Create DataFrame to store similar reports
similar_reports_list = []
for i, j in similar_reports_indices:
    similar_reports_list.append({
        'File1_Index': i,
        'File2_Index': j,
        'File1_Description': sentences_from_file1[i],
        'File2_Description': sentences_from_file2[j],
        'Similarity_Score': similarity_matrix[i, j],
        'Location': locations_from_file1[i]  # Add location information to the result
    })

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame(similar_reports_list)



# COMMAND ----------

display(similar_reports_df)
