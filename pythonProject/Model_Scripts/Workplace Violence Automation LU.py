# Databricks notebook source
# Install Packages 

# COMMAND ----------

# MAGIC %pip install sentence-transformers
# MAGIC %pip install pandas
# MAGIC %pip install nltk 
# MAGIC %pip install gensim
# MAGIC %pip install semantic-text-similarity
# MAGIC %pip install numpy 
# MAGIC %pip install matplotlib 
# MAGIC

# COMMAND ----------

# Import Functions 

# COMMAND ----------

import pandas as pd
from sentence_transformers import SentenceTransformer
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

# Read in files 

# COMMAND ----------

df1 = pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/EIRF_YEAR.csv', encoding = 'latin1')
df2 = pd.read_csv('/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Data/WAASB.csv',encoding = 'latin1')

# COMMAND ----------

# Assuming your DataFrame is similar_reports_df
df2 = df2.rename(columns={'LOCATION': 'Location'})

# COMMAND ----------

df2['Location'] = df2['Location'].str.title() # Useful as this station names are written each word is capatalised 

# COMMAND ----------


# Calculate the percentage of non-null values in the column in one step
percentage_non_null = (df1['Location'].count() / len(df1['Location'])) * 100

print(f"The percentage of non-null values in the column is: {percentage_non_null:.2f}%")

# COMMAND ----------

# Load in the model and tokenizer

# COMMAND ----------

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# COMMAND ----------

# Text Preprocessing 

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
    # Remove stopwords, including custom stopwords
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a cleaned text
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

# Example usage
sentences_from_file1 = [clean_text(sentence) for sentence in df1['Description'].astype(str).tolist()]
sentences_from_file2 = [clean_text(sentence) for sentence in df2['DESCRIPTION'].astype(str).tolist()]


# COMMAND ----------

# Embed the tokens 

# COMMAND ----------

# Encode sentences to obtain embeddings
embeddings1 = model.encode(sentences_from_file1, convert_to_tensor=True)
embeddings2 = model.encode(sentences_from_file2, convert_to_tensor=True)


# COMMAND ----------

# Determine the cosine similarity 

# COMMAND ----------



# Compute cosine similarity between sentence embeddings
similarity_matrix = cosine_similarity(embeddings1, embeddings2)

# Set similarity threshold
threshold = 0.75

# Create DataFrame to store similar reports
similar_reports_list = []

for i, row1 in df1.iterrows():
    for j, row2 in df2.iterrows():
        similarity_score = similarity_matrix[i, j]
        
        # Check if similarity score is above threshold and locations are the same
        if similarity_score > threshold and row1['Location'] == row2['Location']:
            similar_reports_list.append({
                'File1_Index': i,
                'File2_Index': j,
                'File1_Description': sentences_from_file1[i],
                'File2_Description': sentences_from_file2[j],
                'Similarity_Score': similarity_score,
                'EIRF_Location': row1['Location'],
                'EIRF_Date': row1['Date'],
                'WAASB_Location': row2['Location'],
                'WAASB_Date': row2['Date'],
            })

# Create a DataFrame for similar reports
similar_reports_df = pd.DataFrame(similar_reports_list)

# Save the DataFrame to a CSV file
similar_reports_df.to_csv('similar_reports_output.csv', index=False)




# COMMAND ----------

display(similar_reports_df)


# COMMAND ----------

# Save the output file

# COMMAND ----------

# Save the DataFrame with BART summaries to a new CSV file
output_file = 'similar_reports_df'  # Replace with your desired output file path
similar_reports_df.to_csv(output_file, index=False)

# COMMAND ----------



# Assuming embeddings_dataset1 and embeddings_dataset2 are your embeddings
#combined_embeddings = numpy.concatenate((embeddings1, embeddings2), axis=0)


# COMMAND ----------

# Using the previously defined scaler
#scaler = StandardScaler()
#scaler.fit(combined_embeddings)
#scaled_combined_data = scaler.transform(combined_embeddings)


# COMMAND ----------

#n_components = 2
#pca = PCA(n_components=n_components)
#principal_components_combined = pca.fit_transform(scaled_combined_data)


# COMMAND ----------

# Assuming dataset1_size and dataset2_size are the sizes of the original datasets
#df1_size = 256
#df2_size = 88

#plt.scatter(
    #principal_components_combined[:df1_size, 0], principal_components_combined[:df1_size, 1],
    #label='EIRF Workplace Violence Reports'
)
#plt.scatter(
    #principal_components_combined[df1_size:, 0], principal_components_combined[df1_size:, 1],
    #label='WAABS Workplace Violence Reports'
)

#plt.title('PCA Plot of NLP Embeddings for Two Datasets')
#plt.xlabel('Principal Component 1')
#plt.ylabel('Principal Component 2')
#plt.legend()
#plt.show()

