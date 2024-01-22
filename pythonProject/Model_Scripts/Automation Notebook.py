# Databricks notebook source
# MAGIC %pip install dbutils

# COMMAND ----------

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication


# COMMAND ----------

# Set up the email parameters
sender_email = "adetomiwanjoku@tfl.gov.uk"
receiver_email = "adetomiwanjoku@tfl.gov.uk"
subject = "Subject of the email"
body = "Body of the email"

# COMMAND ----------


# Create the MIMEText object
message = MIMEMultipart()
message['From'] = sender_email
message['To'] = receiver_email
message['Subject'] = subject

# COMMAND ----------


# Attach the body of the email
message.attach(MIMEText(body, 'plain'))

# COMMAND ----------



# Attach the CSV file
csv_file_path = "/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/similar_reports_df"  # Replace with the actual path to your CSV file
with open(csv_file_path, "rb") as file:
    csv_attachment = MIMEApplication(file.read(), Name= csv_file_path)
    csv_attachment['Content-Disposition'] = f'attachment; filename="{csv_attachment["Name"]}"'
    message.attach(csv_attachment)

# COMMAND ----------

# Set up the SMTP server
smtp_server = "smtp-mail.outlook.com"
smtp_port = 587
smtp_username = "adetomiwanjoku@tfl.gov.uk"
smtp_password = "SouthKorea25.."

# COMMAND ----------


# Create the SMTP session
with smtplib.SMTP(smtp_server, smtp_port) as server:
    # Start TLS for security
    server.starttls()

    # Login to the server
    server.login(smtp_username, smtp_password)

    # Send the email
    server.sendmail(sender_email, receiver_email, message.as_string())

print("Email with CSV attachment sent successfully!")

