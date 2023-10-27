from sentence_transformers import SentenceTransformer, util
import pandas as pd
import nltk
from gensim.parsing.preprocessing import remove_stopwords

# Load the SentenceTransformer model
model = SentenceTransformer('stsb-roberta-large')

# Load data from Excel sheets into pandas DataFrames, considering only the 'Description' column
df1 = pd.read_csv(r'C:\Users\ChinweNjoku\PycharmProjects\pythonProject\20231010_EIRFsampleData.csv', usecols=['DESCRIPTION'],
                    encoding='latin1')  # Specify the correct encoding (e.g., 'latin1')
df2 = pd.read_csv(r'C:\Users\ChinweNjoku\PycharmProjects\pythonProject\20231010_WAASBsampleData.csv', usecols=['DESCRIPTION'],
                    encoding='latin1')  # Specify the correct encoding (e.g., 'latin1')

# Download NLTK resources
nltk.download('punkt')

# Preprocess text data
def preprocess_text(text):
    # Handle NaN values by replacing them with an empty string
    text = str(text) if not pd.isnull(text) else ''
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word.isalnum() and word not in remove_stopwords(text)]
    # Join tokens back into a cleaned text
    cleaned_text = ' '.join(tokens)
    return cleaned_text

# Compute sentence embeddings for the first DataFrame after text preprocessing
embeddings1 = model.encode(df1['DESCRIPTION'].apply(preprocess_text).astype(str).tolist(), convert_to_tensor=True)

# Compute sentence embeddings for the second DataFrame after text preprocessing
embeddings2 = model.encode(df2['DESCRIPTION'].apply(preprocess_text).astype(str).tolist(), convert_to_tensor=True)

# List to store duplicates
duplicates = []

# Iterate through rows and identify duplicates based on cosine similarity
for index1, embedding1 in enumerate(embeddings1):
    for index2, embedding2 in enumerate(embeddings2):
        # Calculate cosine similarity between embeddings
        similarity_score = util.pytorch_cos_sim(embedding1, embedding2)[0][0].item()

        # If similarity score is above a threshold, consider them duplicates
        similarity_threshold = 0.8
        if similarity_score > similarity_threshold:
            duplicates.append((index1 + 1, index2 + 1, similarity_score))  # Store duplicates

# Create a DataFrame from the duplicates list
duplicates_df = pd.DataFrame(duplicates, columns=['Sheet 1 Row', 'Sheet 2 Row', 'Similarity Score'])

# Save the duplicates to a CSV file
duplicates_df.to_csv('duplicates.csv', index=False)

print("Duplicates saved to 'duplicates.csv'")

