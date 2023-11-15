# Databricks notebook source
# MAGIC %pip install pandas
# MAGIC %pip install fuzzywuzzy
# MAGIC

# COMMAND ----------

import pandas as pd
from fuzzywuzzy import fuzz

df = pd.read_csv('20231010_EIRFsampleData.csv', encoding = 'latin')


# Assume your CSV has a 'description' column
description_column = 'DESCRIPTION'

# COMMAND ----------

df[description_column].str.contains(r'threat|violence', case=False)

# COMMAND ----------

df[description_column].str.replace(r'threat|violence', '***', case=False)

# COMMAND ----------

df[description_column].str.split().apply(len)

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
df['full_report'] = df[['date', 'location', description_column]].apply(lambda x: ' '.join(map(str, x)), axis=1)

# COMMAND ----------

# Converting Cases
df['lowercase_description'] = df[description_column].str.lower()
df['uppercase_description'] = df[description_column].str.upper()
df['titlecase_description'] = df[description_column].str.title()
