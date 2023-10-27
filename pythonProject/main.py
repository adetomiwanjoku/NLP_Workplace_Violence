import pandas as pd
import spacy

# Load spaCy NLP model
nlp = spacy.load('en_core_web_sm')

# Load data from Excel sheets into pandas DataFrames
df1 = pd.read_csv(r'C:\Users\ChinweNjoku\PycharmProjects\pythonProject\20231010_EIRFsampleData.csv', encoding = 'latin1')
df2 = pd.read_csv(r'C:\Users\ChinweNjoku\PycharmProjects\pythonProject\20231010_WAASBsampleData.csv', encoding = 'latin1')


# Define a function to calculate semantic similarity
def calculate_similarity(text1, text2):
    doc1 = nlp(text1.lower())
    doc2 = nlp(text2.lower())
    return doc1.similarity(doc2)


# Iterate through rows and identify duplicates
for index1, row1 in df1.iterrows():
    # Iterate through rows and identify duplicates based on the 'Description' column
    for index1, row1 in df1.iterrows():
        for index2, row2 in df2.iterrows():
            similarity_score = calculate_similarity(str(row1['DESCRIPTION']), str(row2['DESCRIPTION']))

            # If similarity score is above a threshold, consider them duplicates
            similarity_threshold = 0.9
            if similarity_score > similarity_threshold:
                print(
                    f'Duplicate Found: Row {index1 + 1} in Sheet 1 and Row {index2 + 1} in Sheet 2, Similarity Score: {similarity_score:.2f}')






