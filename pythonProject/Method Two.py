import torch
from transformers import BertTokenizer, BertModel
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import re
from collections import Counter

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Read data from CSV files
file_path_1 = r'C:\Users\ChinweNjoku\PycharmProjects\pythonProject\20231010_EIRFsampleData.csv'

file_path_2 = r'C:\Users\ChinweNjoku\PycharmProjects\pythonProject\20231010_WAASBsampleData.csv'

df1 = pd.read_csv(file_path_1, usecols=['DESCRIPTION'], encoding='latin1')

df2 = pd.read_csv(file_path_2, usecols=['DESCRIPTION'], encoding='latin1')


import re
from nltk.tokenize import word_tokenize
from collections import Counter

def preprocess_sentence(sentence, min_word_frequency=2, max_sequence_length=40):
    # Check if the input is a float; if it is, convert it to a string
    if isinstance(sentence, float):
        sentence = str(sentence)

    # Step 1: Convert sentence to lowercase
    sentence = sentence.lower()

    # Step 2: Refit dashes for single words and space punctuation
    sentence = re.sub(r'\b-\b', ' ', sentence)
    sentence = re.sub(r'([.,!?()])', r' \1 ', sentence)

    # Step 3: Tokenize sentence using NLTK tokenizer
    tokens = word_tokenize(sentence)

    # Step 4: Remove non-alphanumeric symbols
    tokens = [word for word in tokens if word.isalnum()]

    # Step 5: Remove words appearing at most once
    word_counts = Counter(tokens)
    tokens = [word if word in word_counts and word_counts[word] >= min_word_frequency else 'UNK' for word in tokens]

    # Step 6: Pad sentences to max_sequence_length
    padded_tokens = tokens[:max_sequence_length] + ['PAD'] * (max_sequence_length - len(tokens))

    return padded_tokens


df1['Preprocessed_Description'] = df1['DESCRIPTION'].apply(preprocess_sentence)

df2['Preprocessed_Description'] = df2['DESCRIPTION'].apply(preprocess_sentence)


from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess and tokenize sentences
sentences_df1 = df1['Preprocessed_Description'].tolist()
sentences_df2 = df2['Preprocessed_Description'].tolist()

# Encode sentences and get BERT embeddings
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Preprocess and tokenize sentences
sentences_df1 = df1['Preprocessed_Description'].tolist()
sentences_df2 = df2['Preprocessed_Description'].tolist()

# Encode sentences and get BERT embeddings
def get_bert_embeddings(sentences):
    input_ids = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')['input_ids']
    with torch.no_grad():
        outputs = model(input_ids)
    # Use the [CLS] token embeddings as sentence embeddings
    cls_embeddings = outputs.last_hidden_state[:, 0, :]
    return cls_embeddings

# Get BERT embeddings for sentences in both DataFrames
embeddings_df1 = get_bert_embeddings(sentences_df1)
embeddings_df2 = get_bert_embeddings(sentences_df2)

# Calculate cosine similarity between the BERT embeddings
cosine_similarities = cosine_similarity(embeddings_df1, embeddings_df2)

# Set the similarity threshold
threshold = 0.8

# Filter pairs of sentences with similarity scores above the threshold
similar_pairs = []
for i in range(len(cosine_similarities)):
    for j in range(len(cosine_similarities[i])):
        if cosine_similarities[i][j] >= threshold:
            similar_pairs.append((i, j, cosine_similarities[i][j]))

# Print similar pairs above the threshold
print("Similar Sentence Pairs Above 80% Similarity:")
for pair in similar_pairs:
    row_index_df1 = pair[0]
    column_index_df2 = pair[1]
    similarity_score = pair[2]
    print(f"df1 sentence {row_index_df1} - df2 sentence {column_index_df2}: Similarity Score - {similarity_score:.2f}")
