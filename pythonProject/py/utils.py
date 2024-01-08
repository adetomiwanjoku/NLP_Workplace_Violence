%pip install pandas
%pip install nltk 
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK resources
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Assuming you have a pandas DataFrame with a 'text_column'
# Replace 'df' and 'text_column' with your actual DataFrame and column names

# Define the clean_text function
def clean_text(text):
    # Handle NaN values by replacing them with an empty string
    text = str(text) if not pd.isnull(text) else ''
    # Lowercase the text
    text = text.lower()
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a cleaned text
    cleaned_text = ' '.join(filtered_tokens)
    return cleaned_text

# Create a pipeline
'''def create_preprocessing_pipeline(df, text_column):
    # Apply the clean_text function
    df['cleaned_text'] = df[text_column].apply(clean_text)
    # Tokenize the cleaned text
    df['tokens'] = df['cleaned_text'].apply(word_tokenize)
    return df'''


