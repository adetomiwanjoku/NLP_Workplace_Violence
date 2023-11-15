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

IRIS_df

# COMMAND ----------

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

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


# Tokenize input sentences with truncation and padding using multilingual tokenizer
encoded_inputs_file1 = tokenizer(sentences_from_file1, padding=True, truncation=True, return_tensors='pt', max_length=512, add_special_tokens=True)
encoded_inputs_file2 = tokenizer(sentences_from_file2, padding=True, truncation=True, return_tensors='pt', max_length=512, add_special_tokens=True)

# Compute BERT embeddings for input sentences using multilingual model
with torch.no_grad():
    outputs_file1 = model(**encoded_inputs_file1)
    outputs_file2 = model(**encoded_inputs_file2)

# COMMAND ----------

embeddings_file1 = outputs_file1.last_hidden_state.mean(dim=1)  # Using mean pooling to get sentence embeddings
embeddings_file2 = outputs_file2.last_hidden_state.mean(dim=1)

print(embeddings_file1.shape)
print(embeddings_file2.shape)

# COMMAND ----------

# Extract the embeddings from the model outputs
embeddings_file1 = outputs_file1.last_hidden_state.mean(dim=1)  # Using mean pooling to get sentence embeddings
embeddings_file2 = outputs_file2.last_hidden_state.mean(dim=1)  # Using mean pooling to get sentence embeddings

# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings_file1, embeddings_file2)

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


