from discord_webhook import DiscordWebhook
from dotenv import load_dotenv
load_dotenv("./.env")
import os

webhook_url = os.getenv("DISCORD_WEBHOOK") # Replace with your own webhook URL


def send_msg(content):
    """
       Send a message to a Discord webhook.

       Args:
           content: The content of the message.
       """
    webhook = DiscordWebhook(url=webhook_url, content=content)
    webhook.execute()