#!/usr/bin/env bash

# usage function
function usage()
{
   cat << HEREDOC

   Usage: ./send_log_email.sh -f <sender email> -t <recipient email> -s <email subject> -a <attachment path>

   arguments:
     -f,--sender-email STR        email address of the sender
     -t,--recipient-email STR     email address of the recipient
     -s,--subject-text STR        email subject text
     -a,--attachment-path STR     path to the log file which should be sent as an attachment

   optional arguments:
     -b,--body-text STR           email body text
     -n,--filter-tail-n INT       filtering for only last N rows of the logging file. Default is 100.
     -o,--only-filtered           send only the filtered (last N rows) file as an email attachment to save space
     -h, --help                   show this help message and exit

HEREDOC
}

sender_email=''
recipient_email=''
subject_text=''
body_text=''

attachment_path=''

filter_last_lines=100
send_only_tail_filtered_log=false


while [[ $# -gt 0 ]]; do
key="$1"

case $key in
    -f|--sender-email)
    sender_email="$2"
    shift 2 # past argument value
    ;;
    -t|--recipient-email)
    recipient_email="$2"
    shift 2 # past argument value
    ;;
    -s|--subject-text)
    subject_text="$2"
    shift 2 # past argument value
    ;;
    -b|--body-text)
    body_text="$2"
    shift 2 # past argument value
    ;;
    -a|--attachment-path)
    attachment_path="$2"
    shift 2 # past argument value
    ;;
    -n|--filter-tail-n)
    filter_last_lines="$2"
    shift 2 # past argument value
    ;;
    -o|--only-filtered)
    send_only_tail_filtered_log=true
    shift 1 # past argument value
    ;;
    -h|--help )
    usage;
    exit;
    ;;
    *)    # unknown option
    echo "Don't know the argument"
    usage;
    exit;
    ;;
esac
done

if [ "$sender_email" == "" ] || [ "$recipient_email" == "" ] || [ "$subject_text" == "" ] || [ "$attachment_path" == "" ]; then
    echo "Not provided required parameters"
    usage
    exit
fi


attachment_filename=$(basename "$attachment_path")

if [ "$send_only_tail_filtered_log" == true ]; then
    echo '{"Data": "From: '${sender_email}'\nTo: '${recipient_email}'\nSubject: '${subject_text}'\nMIME-Version: 1.0\nContent-type: Multipart/Mixed; boundary=\"NextPart\"\n\n--NextPart\nContent-Type: text/plain\n\n'${body_text}'\n\n--NextPart\nContent-Type: text/plain;\nContent-Transfer-Encoding: base64;\nContent-Disposition: attachment; filename=\"tail_'${attachment_filename}'\"\n\n'$(tail -n $filter_last_lines $attachment_path | base64)'\n\n--NextPart--"}' > ~/log_email_message.json
else
    echo '{"Data": "From: '${sender_email}'\nTo: '${recipient_email}'\nSubject: '${subject_text}'\nMIME-Version: 1.0\nContent-type: Multipart/Mixed; boundary=\"NextPart\"\n\n--NextPart\nContent-Type: text/plain\n\n'${body_text}'\n\n--NextPart\nContent-Type: text/plain;\nContent-Transfer-Encoding: base64;\nContent-Disposition: attachment; filename=\"'${attachment_filename}'\"\n\n'$(base64 $attachment_path)'\n\n--NextPart\nContent-Type: text/plain;\nContent-Transfer-Encoding: base64;\nContent-Disposition: attachment; filename=\"tail_'${attachment_filename}'\"\n\n'$(tail -n $filter_last_lines $attachment_path | base64)'\n\n--NextPart--"}' > ~/log_email_message.json

    attachment_filesize=$(ls -l ~/log_email_message.json | awk '{print  $5}')

    # If file size is above 10MB (10000000) limit from aws ses
    if [ "$attachment_filesize" -gt 10000000 ]; then
        echo "Full attachments too large. Switching to tail filtered attachment only"
        body_text="${body_text}\nFull attachments too large. Switched to tail filtered attachment only!"
        echo '{"Data": "From: '${sender_email}'\nTo: '${recipient_email}'\nSubject: '${subject_text}'\nMIME-Version: 1.0\nContent-type: Multipart/Mixed; boundary=\"NextPart\"\n\n--NextPart\nContent-Type: text/plain\n\n'${body_text}'\n\n--NextPart\nContent-Type: text/plain;\nContent-Transfer-Encoding: base64;\nContent-Disposition: attachment; filename=\"tail_'${attachment_filename}'\"\n\n'$(tail -n $filter_last_lines $attachment_path | base64)'\n\n--NextPart--"}' > ~/log_email_message.json
    fi
fi

aws ses send-raw-email --region eu-west-1 --raw-message file://~/log_email_message.json

rm ~/log_email_message.json
