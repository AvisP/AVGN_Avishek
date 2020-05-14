# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 15:02:51 2020

@author: Avishek Paul
"""

from avgn.downloading.download import download_tqdm
from avgn.utils.paths import DATA_DIR
from avgn.utils.general import unzip_file

data_urls = [
    ('https://ndownloader.figshare.com/articles/3470165/versions/1', 'all_files.zip'),
]
output_loc = DATA_DIR/"raw/koumura/"

for url, filename in data_urls:
    download_tqdm(url, output_location=output_loc)
    
    
from joblib import Parallel, delayed
from tqdm.tqdm import tqdm
import pandas as pd
pd.options.display.max_columns = None
import librosa
from datetime import datetime
import numpy as np

import avgn
from avgn.custom_parsing.koumura_bengalese_finch import generate_json, Koumura_Okanoya_parser
from avgn.utils.paths import DATA_DIR

DATASET_ID = 'koumura_bengalese_finch'

# create a unique datetime identifier for the files output by this notebook
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID

DSLOC = avgn.utils.paths.Path('I:/avgn_paper-vizmerge/data/raw/komura')
DSLOC

wav_list = list(DSLOC.glob('Bird*/Wave/*.wav'))
len(wav_list), np.sort(wav_list)[-2:]

annotation_files = list(DSLOC.glob('Bird*/Annotation.xml'))
len(annotation_files), np.sort(annotation_files)[-2:]

# Generate JSON for each wav
song_df = Koumura_Okanoya_parser(annotation_files, wav_list)
