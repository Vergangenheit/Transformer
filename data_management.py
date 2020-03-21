import os
import requests
from zipfile import ZipFile
import unicodedata
import re
import config


def maybe_download_and_read_file(url, filename):
    if not os.path.exists(filename):
        session = requests.Session()
        response = session.get(url, stream=True)

        CHUNK_SIZE = 32768
        with open(filename, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)

    zipf = ZipFile(filename)
    filename = zipf.namelist()
    with zipf.open('fra.txt') as f:
        lines = f.read()

    return lines


def create_raw_data() -> list:
    lines = maybe_download_and_read_file(config.URL, config.FILENAME)
    lines = lines.decode('utf-8')
    raw_data = []
    for line in lines.split('\n'):
        raw_data.append(line.split('\t'))

    return raw_data


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')


def normalize_string(s):
    s = unicode_to_ascii(s)
    s = re.sub(r'([!.?])', r' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    s = re.sub(r'\s+', r' ', s)
    return s


def split_raw_data(rawdata: list):
    raw_data_en, raw_data_fr = list(zip(*rawdata))
    raw_data_en = [normalize_string(data) for data in raw_data_en]
    raw_data_fr_in = ['<start> ' + normalize_string(data) for data in raw_data_fr]
    raw_data_fr_out = [normalize_string(data) + ' <end>' for data in raw_data_fr]

    return raw_data_en, raw_data_fr_in, raw_data_fr_out


# if __name__ == '__main__':
#     raw_data = create_raw_data()
#     raw_data_en, raw_data_fr_in, raw_data_fr_out = split_raw_data(rawdata=raw_data)
