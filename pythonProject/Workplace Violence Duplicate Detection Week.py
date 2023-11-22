# Databricks notebook source
# MAGIC %pip install sentence-transformers
# MAGIC %pip install pandas
# MAGIC %pip install nltk 
# MAGIC %pip install gensim
# MAGIC %pip install semantic-text-similarity
# MAGIC %pip install torch
# MAGIC %pip install numpy 
# MAGIC %pip install matplotlib 

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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# COMMAND ----------

df1 = pd.read_csv('output_file_bart.csv', usecols=['Summary_BART', 'LOCATION'], encoding='latin1')
df2 = pd.read_csv('20231010_WAASBsampleData.csv', usecols=['DESCRIPTION', 'LOCATION'], encoding='latin1')

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

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
sentences_from_file1 = [clean_text(sentence) for sentence in df1['Summary_BART'].astype(str).tolist()]
sentences_from_file2 = [clean_text(sentence) for sentence in df2['DESCRIPTION'].astype(str).tolist()]

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)
embeddings2 = model.encode(sentences_from_file2, convert_to_tensor=True)

# COMMAND ----------


# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# Set similarity threshold
threshold = 0.8
# Find indices of similar reports above the threshold
similar_reports_indices = [(i, j) for i in range(len(sentences_from_file1)) for j in range(len(sentences_from_file2)) if similarity_matrix[i, j] > threshold]

# Create DataFrame to store similar reports
similar_reports_list = []
for i, j in similar_reports_indices:
    similar_reports_list.append({
        'File1_Index': i,
        'File2_Index': j,
        'File1_Description': sentences_from_file1[i],
        'File2_Description': sentences_from_file2[j],
        'Similarity_Score': similarity_matrix[i, j]
    })

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame(similar_reports_list)

# COMMAND ----------

display(similar_reports_df)


# COMMAND ----------


# Calculate the percentage of non-null values in the column in one step
percentage_non_null = (df1['LOCATION'].count() / len(df1['LOCATION'])) * 100

print(f"The percentage of non-null values in the column is: {percentage_non_null:.2f}%")

# COMMAND ----------


# Calculate the percentage of non-null values in the column in one step
percentage_non_null = (df2['LOCATION'].count() / len(df2['LOCATION'])) * 100

print(f"The percentage of non-null values in the column is: {percentage_non_null:.2f}%")


