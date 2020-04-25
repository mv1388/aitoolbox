AWS Simple Email Service
========================

AWS has an email sending service (*Simple Email Service*) which can be used to programmatically send emails from your
python code. Under the hood of *torchtrain* this is used to send training progress notifications. However, the email
sending component can also be used independently.

Email sending component build on top of Simple Email Service is implemented in
:class:`aitoolbox.cloud.AWS.simple_email_service.SESSender`.

To send the email, the user has to provide source and target email address and then specify email's subject and body.
In the case of body content, to achieve a more advanced formatting, the body text should be provided as the HTML
document. Additionally, attachment files can be also sent in the email by proving the list of attachment file paths.
