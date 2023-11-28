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
# MAGIC %pip install keplergl==0.2.2 
# MAGIC %pip install h3
# MAGIC %pip install h3pandas
# MAGIC %pip install geopandas

# COMMAND ----------

# MAGIC %md
# MAGIC # Import Packages 

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
from keplergl import KeplerGl 
import H3 

# COMMAND ----------

# MAGIC %md
# MAGIC # Read in the data

# COMMAND ----------

df1 = pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/three_day_copy_surface_reports.csv', encoding = 'latin1')

# COMMAND ----------

display(df1)

# COMMAND ----------

df1.columns

# COMMAND ----------

df1 = df1.rename(columns={'Location / Road Name': 'Location'})

# COMMAND ----------

df1['Location'] = df1['Location'].str.title() # Useful as this station names are written each word is capatalised

# COMMAND ----------

# MAGIC %md
# MAGIC # Load model and tokenizer

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
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

# Embed the words 

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)

# COMMAND ----------

# MAGIC %md
# MAGIC # Define cosine similarity 

# COMMAND ----------


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
    if (similarity_matrix[i, j] > threshold)
]

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame([
    {
        'File1_Index': i,
        'File2_Index': j,
        'Description': sentences_from_file1[i],
        'Duplicate': sentences_from_file1[j],
        'Similarity_Score': similarity_matrix[i, j],
        
    }
    for i, j in similar_reports_indices
])

# Add new index columns for each file index that adds two to the existing index
similar_reports_df['Row_Num'] = similar_reports_df['File1_Index'] + 2
similar_reports_df['Row_Num_Duplicate'] = similar_reports_df['File2_Index'] + 2


# COMMAND ----------

df1

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
similar_reports_df = similar_reports_df[['Row_Num', 'Row_Num_Duplicate', 'Description', 'Duplicate','Similarity_Score_Percent', 'Is_Duplicate']]

# COMMAND ----------

similar_reports_df


# COMMAND ----------

import geopandas as gpd
latlong = 'epsg:4326'
ukgrid = 'epsg:27700'
map_df= gpd.GeoDataFrame(df1.drop(['Easting','Northing'], axis=1), 
geometry=gpd.points_from_xy(df1['Easting'],df1['Northing']), crs=ukgrid)
df1.to_crs(latlong, inplace=True)
#convert easting and northing to long & lat 

# COMMAND ----------

from h3 import h3
h3_level = 9
 
def lat_lng_to_h3(row):
    return h3.geo_to_h3(
      row.geometry.y, row.geometry.x, h3_level)
 
df1['h3'] = df1.apply(lat_lng_to_h3, axis=1)

# COMMAND ----------

df1['lon'] = df1['geometry'].x
df1['lat'] = df1['geometry'].y

# COMMAND ----------

df1 = df1.drop(['geometry'], axis=1)

# COMMAND ----------

def create_kepler_html(data, config, height):
  map_1 = KeplerGl(height=height, data=data, config=config)
  html = map_1._repr_html_().decode("utf-8")
  new_html = html + """<script>
      var targetHeight = "{height}px";
        var interval = window.setInterval(function() {{
        if (document.body && document.body.style && document.body.style.height !== targetHeight) {{
          document.body.style.height = targetHeight;
        }}
      }}, 250);</script>""".format(height=height)
  return new_html 

# COMMAND ----------


map_2 = KeplerGl(height=400, data={ "data_1" : df1}, map_config=map_config) # use the map config created with the two datasets 
display(map_2)
