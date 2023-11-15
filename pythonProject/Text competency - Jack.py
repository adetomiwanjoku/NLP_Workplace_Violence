# Databricks notebook source
# MAGIC %pip install pandas
# MAGIC %pip install fuzzywuzzy
# MAGIC %pip install seaborn 
# MAGIC %pip install matplotlib 
# MAGIC

# COMMAND ----------

import pandas as pd
from fuzzywuzzy import fuzz
import seaborn 
import matplotlib.pyplot as plt
df = pd.read_csv('20231010_EIRFsampleData.csv', encoding = 'latin')


# Assume your CSV has a 'description' column
description_column = 'DESCRIPTION'

# COMMAND ----------

df[description_column].str.contains(r'weapon|violence', case=False)

# COMMAND ----------

df[description_column].str.replace(r'threat|violence', '***', case=False)

# COMMAND ----------

word_count = df[description_column].str.split().apply(len)

# COMMAND ----------

# Plotting a box plot
plt.figure(figsize=(8, 6))
plt.boxplot(word_count, vert=False)
plt.xlabel('Word Count')
plt.title('Box Plot of Word Counts in Descriptions')
plt.show()


# COMMAND ----------

df[description_column].str.strip()

# COMMAND ----------

# Fuzzy Matching
reference_description = "Physical altercation in the workplace"
df['fuzzy_similarity_to_reference'] = df[description_column].apply(lambda x: fuzz.ratio(x, reference_description))

# COMMAND ----------


# Term Frequency
terms_to_check = ['physical', 'altercation', 'harassment', 'weapon']
for term in terms_to_check:
    df[f'term_frequency_{term}'] = df[description_column].apply(lambda x: x.lower().split().count(term.lower()))

# COMMAND ----------


# Joining Strings (Concatenate)
df['full_report'] = df[['Date', 'LOCATION', description_column]].apply(lambda x: ' '.join(map(str, x)), axis=1)

# COMMAND ----------

# Converting Cases
df['lowercase_description'] = df[description_column].str.lower()
df['uppercase_description'] = df[description_column].str.upper()
df['titlecase_description'] = df[description_column].str.title()

# COMMAND ----------

df
