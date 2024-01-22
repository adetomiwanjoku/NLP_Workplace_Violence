# Databricks notebook source
# MAGIC %pip install dbutils

# COMMAND ----------

# Email parameters
sender_email = "adetomiwanjoku@tfl.gov.uk" 
receiver_email = "adetomiwanjoku@tfl.gov.uk"
password = "SouthKorea25.."

# COMMAND ----------

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders

# COMMAND ----------

# Attachment parameters
attachment_path = "/Workspace/Repos/adetomiwanjoku@tfl.gov.uk/NLP_Workplace_Violence/pythonProject/Model_Scripts/similar_reports_df" 

# Create a multipart message and set headers
message = MIMEMultipart()
message["From"] = sender_email
message["To"] = receiver_email
message["Subject"] = "CSV Attachment Test"
# Customize the body message
body = 'Hello, please find the duplicate reports identified for workplace violence.'


# COMMAND ----------

# convert the body to a MIME compatible string
body = MIMEText(body) 
# attach it to your main message
message.attach(body) 

# Open the attachment file in bynary
with open(attachment_path, "rb") as attachment:
    # Add file as application/octet-stream
    part = MIMEBase("application", "octet-stream")
    part.set_payload((attachment).read())

# Encode file in ASCII characters to send by email    
encoders.encode_base64(part)

# Add header with pdf name
part.add_header(
    "Content-Disposition",
    f"attachment; filename={attachment_path}",
)

# Add attachment to message and convert message to string
message.attach(part)
text = message.as_string()

# COMMAND ----------

text 

# COMMAND ----------

# Connect to the Outlook SMTP server and send the email
with smtplib.SMTP("smtp-mail.outlook.com", 587) as server:
    server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email, text)
