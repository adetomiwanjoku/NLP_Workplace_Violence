# Databricks notebook source
# MAGIC %pip install dbutils

# COMMAND ----------

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
import dbutils

# COMMAND ----------


# Email configuration
sender_email = "your_email@hotmail.com"  # Update with your Outlook email
receiver_email = "recipient_email@gmail.com"
subject = "Databricks Output File"
body = "This email contains a table with proposed semantic duplicates identified by the model."

# Databricks output file path
output_file_path = "/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/similar_reports_df"


# COMMAND ----------



# Retrieve email credentials from Secrets
smtp_username = dbutils.secrets.get("email-secrets", "outlook-username")
smtp_password = dbutils.secrets.get("email-secrets", "outlook-password")

# Create the MIME object
msg = MIMEMultipart()
msg['From'] = sender_email
msg['To'] = receiver_email
msg['Subject'] = subject
msg.attach(MIMEText(body, 'plain'))

# Attach the Databricks output file
with open(output_file_path, "rb") as attachment:
    part = MIMEApplication(attachment.read(), Name="output_file.txt")
    part['Content-Disposition'] = f'attachment; filename="{output_file_path}"'
    msg.attach(part)

# SMTP server configuration for Outlook
smtp_server = "smtp.office365.com"
smtp_port = 587




# COMMAND ----------

# Connect to the SMTP server and send the email
with smtplib.SMTP(smtp_server, smtp_port) as server:
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.sendmail(sender_email, receiver_email, msg.as_string())

print("Email sent successfully.")
