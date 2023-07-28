from dotenv import load_dotenv
import os
from icloud_connection import get_and_connect_api_to_icloud, get_all_undownload_photos

if __name__ == "__main__":
    # Initalise enviroment
    load_dotenv()
    apple_id = os.getenv("APPLE_ID")
    apple_password = os.getenv("APPLE_PASSWORD")

    # Syncing - probably don't want this to run every time
    api = get_and_connect_api_to_icloud(apple_id, apple_password)
    get_all_undownload_photos(api)
