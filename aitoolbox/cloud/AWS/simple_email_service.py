import os
import boto3
from botocore.exceptions import ClientError
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart


class SESSender:
    def __init__(self, sender_name, sender_email, recipient_email,
                 aws_region='eu-west-1'):
        """AWS Simple Email Service sender

        Used for sending email notifications about the progression of the training.

        Args:
            sender_name (str): Name of the email sender
            sender_email (str): Email of the email sender
            recipient_email (str): Email where the email will be sent
            aws_region (str): AWS SES region
        """
        self.sender_name = sender_name
        self.sender_email = sender_email
        self.sender = f'{sender_name} <{sender_email}>'

        self.recipient_email = recipient_email

        self.aws_region = aws_region
        self.CHARSET = "UTF-8"

        # Create a new SES resource and specify a region.
        self.client = boto3.client('ses', region_name=self.aws_region)

    def send_email(self, subject, body, attachment_file_paths=None):
        """Send email text with optional attachments

        Args:
            subject (str): email subject
            body (str): HTML body of the email
            attachment_file_paths (list or None): list of local paths pointing to the email attachment files

        Returns:
            None
        """
        # Create a multipart/mixed parent container.
        msg = MIMEMultipart('mixed')
        msg['Subject'] = subject
        msg['From'] = self.sender_email
        msg['To'] = self.recipient_email

        # Create a multipart/alternative child container.
        msg_body = MIMEMultipart('alternative')

        # The HTML body of the email.
        BODY_HTML = f"""<html>
                <head></head>
                <body>
                  {body}
                </body>
                </html>
                            """

        # Encode the text and HTML content and set the character encoding. This step is
        # necessary if you're sending a message with characters outside the ASCII range.
        # textpart = MIMEText(BODY_TEXT.encode(CHARSET), 'plain', CHARSET)
        htmlpart = MIMEText(BODY_HTML.encode(self.CHARSET), 'html', self.CHARSET)

        # Add the text and HTML parts to the child container.
        # msg_body.attach(textpart)
        msg_body.attach(htmlpart)

        # Define the attachment part and encode it using MIMEApplication.
        # Add a header to tell the email client to treat this part as an attachment,
        # and to give the attachment a name.
        # Add the attachment to the parent container.
        if attachment_file_paths is not None and len(attachment_file_paths) > 0:
            for attachment_path in attachment_file_paths:
                with open(attachment_path, 'rb') as f:
                    att = MIMEApplication(f.read())
                    att.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                    msg.attach(att)

        # Attach the multipart/alternative child container to the multipart/mixed
        # parent container.
        msg.attach(msg_body)

        try:
            # Provide the contents of the email.
            response = self.client.send_raw_email(
                Source=self.sender,
                Destinations=[self.recipient_email],
                RawMessage={
                    'Data': msg.as_string(),
                }
            )
        # Display an error if something goes wrong.
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            print("Email successfully sent!")
