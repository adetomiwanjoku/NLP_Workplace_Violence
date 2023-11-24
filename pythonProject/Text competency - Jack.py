# Databricks notebook source
# MAGIC %pip install pandas
# MAGIC %pip install fuzzywuzzy
# MAGIC %pip install seaborn 
# MAGIC %pip install matplotlib 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Workplace violence is a serious threat to the well-being our operational coleagues. There are various report froms that the staff can record the incidents they are involved. The Electronic Incidence Report form is the main reporting tool used to report violence for London Underground staff. 

# COMMAND ----------

import pandas as pd
from fuzzywuzzy import fuzz
import seaborn 
import matplotlib.pyplot as plt
df = pd.read_csv('20231010_EIRFsampleData.csv', encoding = 'latin')


# Assume your CSV has a 'description' column
description_column = 'DESCRIPTION'

# COMMAND ----------

df

# COMMAND ----------

df[description_column].str.contains(r'Male', case=False).count() # every EIRF form has the word male appearing at least once 

# COMMAND ----------

import re

# COMMAND ----------

pattern = (r'\b(?:violence|incident|assault|harassment|threat|injury|attack|harrassing|aggresive|abusive|absue)\b')
# Apply the regex pattern to the 'report_text' column
df['violence_related'] = df['DESCRIPTION'].apply(lambda x: bool(re.search(pattern, x, flags=re.IGNORECASE)))


# COMMAND ----------

display(df)

# COMMAND ----------

df['violence_related'].mean() * 100

# COMMAND ----------

word_count = df[description_column].str.split().apply(len) # counting the number of words that is in each entry 

# COMMAND ----------

# Plotting a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(word_count, vert=False) # outliers showing that there are entries considerably higher than the median. Due to this I actually carried out text summarisation using BART a neural network 
plt.xlabel('Word Count')
plt.title('Box Plot of Word Counts in Descriptions')
plt.show()


# COMMAND ----------

df_stripped = df[description_column].str.strip() # removes whitespace

# COMMAND ----------

df_stripped

# COMMAND ----------

# Fuzzy Matching
reference_description = "Physical altercation in the workplace"
df['fuzzy_similarity_to_reference'] = df[description_column].apply(lambda x: fuzz.ratio(x, reference_description)) 

# COMMAND ----------

display(df)

# COMMAND ----------


# Term Frequency
terms_to_check = ['physical', 'injury', 'harassment', 'pushed', 'abuse']
for term in terms_to_check:
    df[f'term_frequency_{term}'] = df[description_column].apply(lambda x: x.lower().split().count(term.lower())) # Counts the number of times this words occur per entry 

# COMMAND ----------


# Joining Strings (Concatenate)
df['full_report'] = df[['Date', 'LOCATION', description_column]].apply(lambda x: ' '.join(map(str, x)), axis=1) # key info at a snap shot to understand 

# COMMAND ----------

# Converting Cases
df['lowercase_description'] = df[description_column].str.lower()
df['uppercase_description'] = df[description_column].str.upper()
df['titlecase_description'] = df[description_column].str.title() # Useful as this station names are written each word is capatalised 

# COMMAND ----------

display(df)

# COMMAND ----------

# This work gives a glimpse into the nature and severity of workplace violence and gave me a deeper understanding of the data.
