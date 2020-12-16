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
    file_ids = ['1-m20o6ss8hQepb8qcuIVtM1jEydQOuho','1R4vscrsUhn2cJCFHj3F8VoH9s1U88H8A','1ppg5dwvQ_j4tl_aahcDpRrIMpux-Y9IH']
    filenames = ['names_2M.pkl','reachability_2M.pkl','names.trimmed.2M.txt']
    data_root = "data"
    datapaths = ["FB2M","freebase"]

    if not os.path.exists("data/freebase/"):
        os.makedirs("data/freebase")

    for i, id in enumerate(file_ids):
        download_file_from_google_drive(id, f"data/freebase/{filenames[i]}", fname=filenames[i])

    print("All files downloaded succesfully !!")