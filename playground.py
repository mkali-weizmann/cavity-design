import win32com.client
import datetime
import pytz  # install with `pip install pytz` if you don't already have it

def check_outlook_for_subject(subject_keyword="אישור עסקה"):
    # Connect to Outlook
    outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")

    # Get Inbox (6 = inbox folder)
    inbox = outlook.GetDefaultFolder(6)
    messages = inbox.Items
    messages.Sort("[ReceivedTime]", True)

    # Create a timezone-aware datetime for comparison
    local_tz = datetime.datetime.now().astimezone().tzinfo
    now = datetime.datetime.now(local_tz)
    time_limit = now - datetime.timedelta(hours=24)

    for message in messages:
        try:
            if message.ReceivedTime < time_limit:
                break
            if subject_keyword in message.Subject:
                print("✔ Found:", message.Subject, "|", message.ReceivedTime)
                return True
        except AttributeError:
            continue

    print("✘ No matching email found in the last 24 hours.")
    return False

# Run the check
check_outlook_for_subject()
