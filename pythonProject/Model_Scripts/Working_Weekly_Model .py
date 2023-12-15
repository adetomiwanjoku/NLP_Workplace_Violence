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
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in week's worth of workplace violence reports 

# COMMAND ----------

#df1 = pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/Week_Data.csv', encoding = 'latin1')
df1 = pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/Copy_WVA_Incidents.csv', encoding = 'latin1')
#df1 = pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/WVA_Incidents.csv', encoding= 'latin')

# COMMAND ----------

display(df1)

# COMMAND ----------

# Assuming your DataFrame is similar_reports_df
df1 = df1.rename(columns={'ï»¿Data Ref Num': 'Reference_Number', 'Incident Updated Date' : 'Incident_Date'})


# COMMAND ----------

# MAGIC %md
# MAGIC # Define the model and the tokenizer 

# COMMAND ----------

#tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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

sentences_from_file1 # look into this 

# COMMAND ----------

# MAGIC %md
# MAGIC # Embeds the tokens 

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)


# COMMAND ----------

embeddings1 # looj 

# COMMAND ----------

# MAGIC %md
# MAGIC # Identify duplicates using threshold with cosine similarity 

# COMMAND ----------



# Reset the index of df1
df1 = df1.reset_index(drop=True)

# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1)

# Set similarity threshold
threshold = 0.70

# Find indices of similar reports above the threshold with the same location
similar_reports_indices = [
    (i, j)
    for i in range(len(sentences_from_file1))
    for j in range(i + 1, len(sentences_from_file1))
    if (similarity_matrix[i, j] >= threshold) and (df1['LOCATION'].iloc[i] == df1['LOCATION'].iloc[j])
]

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': i,
        'File2_Index': j,
        'Description': sentences_from_file1[i],
        'Duplicate': sentences_from_file1[j],
        'Similarity_Score': similarity_matrix[i, j],
        'Location': df1['LOCATION'].iloc[i],
    }
    for i, j in similar_reports_indices
])

# Add new index columns for each file index that adds two to the existing index
similar_reports_df['Row_Num'] = similar_reports_df['File1_Index'] + 2
similar_reports_df['Row_Num_Duplicate'] = similar_reports_df['File2_Index'] + 2






# COMMAND ----------

# MAGIC %md
# MAGIC # Create Output File 

# COMMAND ----------

# Drop the old file index columns
similar_reports_df = similar_reports_df.drop(['File1_Index', 'File2_Index'], axis=1)


# COMMAND ----------


# Add a column called 'Duplicate_Reference_Number'
similar_reports_df['Duplicate_Reference_Number'] = similar_reports_df['Row_Num_Duplicate'].apply(
    lambda file2_index: df1.iloc[file2_index - 2]['Reference_Number']
)
# Add an empty column called 'is_duplicate'
similar_reports_df['Is_Duplicate'] = ''


# COMMAND ----------

similar_reports_df.columns

# COMMAND ----------

similar_reports_df['Similarity_Score_Percent'] = (similar_reports_df['Similarity_Score'] * 100).round()


# COMMAND ----------

# Reorder columns with the new index as the first column
similar_reports_df = similar_reports_df[['Row_Num', 'Row_Num_Duplicate', 'Description', 'Duplicate', 'Duplicate_Reference_Number', 'Location', 'Similarity_Score_Percent', 'Is_Duplicate']]

# COMMAND ----------

description_to_remove = 'nan'

# Filter rows based on the condition
similar_reports_df = similar_reports_df[similar_reports_df['Description'] != description_to_remove]

similar_reports_df.drop(similar_reports_df[similar_reports_df['Description'] == description_to_remove].index, inplace=True)

# COMMAND ----------

display(similar_reports_df)
