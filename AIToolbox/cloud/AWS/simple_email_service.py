import boto3
from botocore.exceptions import ClientError


class SESSender:
    def __init__(self,sender_name, sender_email, recipient_email,
                 aws_region='eu-west-1'):
        """

        Args:
            sender_name:
            sender_email:
            recipient_email:
            aws_region:
        """
        self.sender_name = sender_name
        self.sender_email = sender_email
        self.sender = f'{sender_name} <{sender_email}>'

        self.recipient_email = recipient_email

        self.aws_region = aws_region
        self.CHARSET = "UTF-8"

        # Create a new SES resource and specify a region.
        self.client = boto3.client('ses', region_name=self.aws_region)

    def send_email(self, subject, body):
        """

        Args:
            subject:
            body:

        Returns:

        """
        # The HTML body of the email.
        BODY_HTML = f"""<html>
        <head></head>
        <body>
          {body}
        </body>
        </html>
                    """

        try:
            # Provide the contents of the email.
            response = self.client.send_email(
                Destination={
                    'ToAddresses': [self.recipient_email],
                },
                Message={
                    'Body': {
                        'Html': {
                            'Charset': self.CHARSET,
                            'Data': BODY_HTML,
                        },
                        # 'Text': {
                        #     'Charset': self.CHARSET,
                        #     'Data': BODY_TEXT,
                        # },
                    },
                    'Subject': {
                        'Charset': self.CHARSET,
                        'Data': subject,
                    },
                },
                Source=self.sender
            )
        # Display an error if something goes wrong.
        except ClientError as e:
            print(e.response['Error']['Message'])
        else:
            print("Email sent! Message ID:"),
            print(response['MessageId'])
