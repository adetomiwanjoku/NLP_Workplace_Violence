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

df1 = pd.read_csv('20231010_EIRFsampleData.csv', usecols=['DESCRIPTION'], encoding='latin1')
df2 = pd.read_csv('20231010_WAASBsampleData.csv', usecols=['DESCRIPTION'], encoding='latin1')

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained("StructBERTRoBERTa ensemble")

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
  

  
# Clean and preprocess sentences
sentences_from_file1 = [clean_text(sentence) for sentence in df1['DESCRIPTION'].astype(str).tolist()]
sentences_from_file2 = [clean_text(sentence) for sentence in df2['DESCRIPTION'].astype(str).tolist()]

# COMMAND ----------


# Tokenize input sentences with truncation and padding using multilingual tokenizer
encoded_inputs_file1 = tokenizer(sentences_from_file1, padding=True, truncation=True, return_tensors='pt', max_length=512, add_special_tokens=True)
encoded_inputs_file2 = tokenizer(sentences_from_file2, padding=True, truncation=True, return_tensors='pt', max_length=512, add_special_tokens=True)

# Compute BERT embeddings for input sentences using multilingual model
with torch.no_grad():
    outputs_file1 = model(**encoded_inputs_file1)
    outputs_file2 = model(**encoded_inputs_file2)

# COMMAND ----------

type(encoded_inputs_file1)

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

# Save similar reports to a new CSV file
similar_reports_df.to_csv('similar_reports.csv', index=False)

# COMMAND ----------

display(similar_reports_df)

