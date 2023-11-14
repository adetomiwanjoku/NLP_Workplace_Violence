# Databricks notebook source
dbutils.fs.mount(
  source = "wasbs://<container-name>@<storage-account-name>.blob.core.windows.net",
  mount_point = "/mnt/iotdata",
  extra_configs = {"fs.azure.account.key.<storage-account-name>.blob.core.windows.net":dbutils.secrets.get(scope = "<scope-name>", key = "<key-name>")})

# COMMAND ----------

dbutils.fs.mounts()

# COMMAND ----------

dbutils.fs.ls("/mnt")
