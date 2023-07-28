from pyicloud import PyiCloudService
import os

# from feature_store import add_new_feature
from pathlib import Path

PATH = Path("/home/fergus/data/icloud_data/raw_icloud")


def get_and_connect_api_to_icloud(apple_id: str, apple_password: str):
    """Get connected icloud api.

    Returns:
        api (PyiCloudService) : logged in icloud api
    """
    api = PyiCloudService(apple_id, apple_password)
    if api.requires_2fa:
        print("2fa required.")
        code = input("Enter 2fa code: ")
        result = api.validate_2fa_code(code)
        print("Code validation result: %s" % result)
        if not result:
            print("Failed to verify security code")
            return

        if not api.is_trusted_session:
            print("Session is not trusted. Requesting trust...")
            result = api.trust_session()
            print("Session trust result %s" % result)

            if not result:
                print(
                    "Failed to request trust. You will likely be prompted for the code again in the coming weeks"
                )
    return api


def get_all_undownload_photos(api: PyiCloudService):
    """Get all photos from icloud that have not been downloaded yet.

    Args:
        api (PyiCloudService): api that has been authenticated to icloud
    """
    for i, photo in enumerate(api.photos.all):
        print(i)
        fn = f"{str(PATH)}/{photo.filename}"
        print(os.path.exists(fn))
        if os.path.exists(fn):
            print("already loaded")
            continue
        if "HEIC" not in fn:
            continue
        download = photo.download()
        with open(fn, "wb") as opened_file:
            opened_file.write(download.raw.read())
        os.system(f'convert {fn} {fn.replace("HEIC", "jpg")}')
        os.remove(fn)
        name = fn.replace(".jpg", "")
        # add_new_feature(name)
