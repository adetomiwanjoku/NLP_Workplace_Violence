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

df1 = pd.read_csv('Week_Data.csv', usecols = ['DESCRIPTION', 'LOCATION'], encoding = 'latin1')

# COMMAND ----------

df1.head()

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
sentences_from_file1 = [clean_text(sentence) for sentence in df1['DESCRIPTION'].astype(str).tolist()]


# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)


# COMMAND ----------

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have embeddings, sentences_from_file, and your DataFrame df defined
# df should have a 'Location' column

# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1)

# Set similarity threshold
threshold = 0.85

# Find indices of similar reports above the threshold with the same location
similar_reports_indices = [
    (i, j) 
    for i in range(len(sentences_from_file1)) 
    for j in range(i + 1, len(sentences_from_file1)) 
    if (similarity_matrix[i, j] > threshold) and (df1['LOCATION'].iloc[i] == df1['LOCATION'].iloc[j])
]

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': df1.index[i],
        'File2_Index': df1.index[j],
        'File1_Description': sentences_from_file1[i],
        'File2_Description': sentences_from_file1[j],
        'Similarity_Score': similarity_matrix[i, j],
        'Location': df1['LOCATION'].iloc[i]
    }
    for i, j in similar_reports_indices
])

# Display the DataFrame
print(similar_reports_df)



# COMMAND ----------

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have embeddings, sentences_from_file, and your DataFrame df defined
# df should have a 'Location' column

# Reset the index of df1
df1 = df1.reset_index(drop=True)

# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1)

# Set similarity threshold
threshold = 0.85

# Find indices of similar reports above the threshold with the same location
similar_reports_indices = [
    (i, j)
    for i in range(len(sentences_from_file1))
    for j in range(i + 1, len(sentences_from_file1))
    if (similarity_matrix[i, j] > threshold) and (df1['LOCATION'].iloc[i] == df1['LOCATION'].iloc[j])
]

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': i,
        'File2_Index': j,
        'File1_Description': sentences_from_file1[i],
        'File2_Description': sentences_from_file1[j],
        'Similarity_Score': similarity_matrix[i, j],
        'Location': df1['LOCATION'].iloc[i]
    }
    for i, j in similar_reports_indices
])

# Add new index columns for each file index that adds two to the existing index
similar_reports_df['New_File1_Index'] = similar_reports_df['File1_Index'] + 2
similar_reports_df['New_File2_Index'] = similar_reports_df['File2_Index'] + 2

# Display the updated DataFrame
print(similar_reports_df)






# COMMAND ----------

# Drop the old file index columns
similar_reports_df = similar_reports_df.drop(['File1_Index', 'File2_Index'], axis=1)

# Reorder columns with the new index as the first column
similar_reports_df = similar_reports_df[['New_File1_Index', 'New_File2_Index', 'File1_Description', 'File2_Description', 'Similarity_Score', 'Location']]

# Display the updated DataFrame

# COMMAND ----------

display(similar_reports_df)
