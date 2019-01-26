import smtplib

smtp_server = "smtp.gmail.com"
port = 465  # For SSL
sender_email = "mostafa.3zazi3@gmail.com" 
receiver_email = "mostafa.3zazi@gmail.com"
message = """\
Subject: Hi ghost mecky

This message is sent from azazi."""

#gmail pass for sender email
password = ""   

try:
    server = smtplib.SMTP_SSL(smtp_server, port)
    server.ehlo()
    #server.starttls()
    server.login(sender_email, password)
    server.sendmail(sender_email, receiver_email , message)
    server.close()
    print ('successfully sent the mail')
except Exception as e:
    print ("failed to send mail")
