# Databricks notebook source
# MAGIC %pip install sumy 
# MAGIC %pip install pandas 
# MAGIC %pip install nltk 
# MAGIC %pip install transformers 
# MAGIC %pip install torch 

# COMMAND ----------

# Import the summarizer
from sumy.summarizers.lsa import LsaSummarizer
# Parsing the text string using PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
import pandas as pd
import nltk
import torch

# COMMAND ----------


from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig


# Read the CSV file into a DataFrame
df = pd.read_csv('EIRF_YEAR.csv', encoding = 'latin')

# Tokenizer and Model for BART
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')

# Function to summarize a given text using BART
def summarize_text_bart(text):
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Apply the BART summarization function to the 'Description' column
df['Summary_BART'] = df['Description'].apply(summarize_text_bart)

# Save the DataFrame with BART summaries to a new CSV file
output_file_path_bart = 'output_file_bart_year.csv'  # Replace with your desired output file path
df.to_csv(output_file_path_bart_year, index=False)

# Display the DataFrame with BART summaries
print(df)


# COMMAND ----------

display(df)
