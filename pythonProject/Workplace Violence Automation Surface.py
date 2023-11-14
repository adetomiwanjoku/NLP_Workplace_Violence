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


