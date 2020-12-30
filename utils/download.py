import requests
import os
from tqdm import tqdm

def download_file_from_google_drive(id, destination, fname=None):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination, fname)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination, fname):
    CHUNK_SIZE = 32768
    with open(destination, "wb") as f:
        for chunk in tqdm(response.iter_content(CHUNK_SIZE), desc=f"{fname}: "):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    data_root = "data"
    datapaths = ["freebase","processed_simplequestions_dataset"]

    files = {
        "processed_simplequestions_dataset": [
            {
                "file-id":"11gEKSVqJbU4kOc8fqkIxUhQnzVysWmF6",
                "filename":"train100.txt"
            },
            {
                "file-id": "1N2c_eiVpbigdNop-_KwU1YPUjtSnKlh4",
                "filename": "test100.txt"
            },
            {
                "file-id": "1gxRrkikHIm2ESp_lW3somQvFP0qzuEUJ",
                "filename": "valid100.txt"
            }
        ],
        "freebase": [
            {
                "file-id": "1ppg5dwvQ_j4tl_aahcDpRrIMpux-Y9IH",
                "filename": "names.trimmed.2M.txt"
            }
        ]
    }

    for apath in datapaths:
        if not os.path.exists(os.path.join(data_root,apath)):
            os.makedirs(os.path.join(data_root,apath))

    for folder,v in files.items():
        for afile in files[folder]:
            print(f"Downloading {afile['filename']}...")
            download_file_from_google_drive(afile["file-id"],destination=os.path.join(data_root,folder,afile["filename"]),fname=afile["filename"])
            print("Done !!")

    print("All files are downloaded successfully !!")