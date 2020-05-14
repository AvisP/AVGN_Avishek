# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:56:10 2020

@author: AvishekPaul
"""
#####  SCRIPT 1

from avgn.utils.general import prepare_env
prepare_env()

from joblib import Parallel, delayed
import tqdm
import pandas as pd
pd.options.display.max_columns = None
import librosa
from datetime import datetime
import numpy as np

import avgn
from avgn.custom_parsing.zebra_finch_gardner import generate_json_wav_noise
from avgn.utils.paths import DATA_DIR

######## Load data in original Format

DATASET_ID = 'zebra_finch_gardner'

# create a unique datetime identifier for the files output by this notebook
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID

DSLOC = avgn.utils.paths.Path('I:/avgn_paper-vizmerge/data/ZebraFinch/download')
DSLOC

MATFILES = list((DSLOC).expanduser().glob('*/[!.]*.mat')) 
len(MATFILES), MATFILES[0]

####### parse MAT and create wav/JSON
import h5py as h5

for mat_file in tqdm.tqdm(MATFILES):
    print(mat_file)
    indv = mat_file.parent.stem
    # try loading the MAT file
    try:
        with h5.File(mat_file,'r') as f:
            songs = f["song"][()]
            nonsongs = f["nonsong"][()]
            rate = f["fs"][()]
    except:
        print(indv + " failed")
        continue
    
    for wav_num, (song, nonsong) in tqdm.tqdm(enumerate(zip(songs, nonsongs)), total=len(songs)):
        generate_json_wav_noise(indv, wav_num, song, nonsong, int(rate), DT_ID)

    # with Parallel(n_jobs=2, verbose=10) as parallel:
    #     parallel(
    #         delayed(generate_json_wav_noise)(indv, wav_num, song, nonsong, int(rate), DT_ID)
    #         for wav_num, (song, nonsong) in tqdm.tqdm(
    #             enumerate(zip(songs, nonsongs)), total=len(songs)
    #         )
    #     )
        
# for wav_num, (song, nonsong) in tqdm.tqdm(enumerate(zip(songs, nonsongs)), total=len(songs)):
#     generate_json_wav_noise(indv, wav_num, song, nonsong, int(rate), DT_ID)
        
### -------------------------------------------------------------------------------------------------------######
        
## SEGMENTATION USING DYNAMIC THRESHOLDING
        
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='once')

from avgn.utils.audio import load_wav, read_wav
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.song_segmentation.dynamic_thresholding import dynamic_threshold_segmentation
from avgn.song_segmentation.dynamic_thresholding import plot_segmented_spec
from avgn.visualization.spectrogram import plot_segmentations

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
from avgn.utils.hparams import HParams
from avgn.dataset import DataSet

hparams = HParams(
    n_fft = 4096,
    mel_lower_edge_hertz=500,
    mel_upper_edge_hertz=12000,
    butter_lowcut = 500,
    butter_highcut = 12000,
    ref_level_db = 20,
    min_level_db = -100,
    win_length_ms = 4,
    hop_length_ms = 1,
    num_mel_bins = 32,
    mask_spec = True,
    n_jobs = 1,  # Makes processing serial if set to 1, parallel processing giving errors
    verbosity=1,
    nex = -1
)

### segmentation parameters
n_fft=1024
hop_length_ms=1
win_length_ms=5
ref_level_db=15
pre=0.97
min_level_db=-120
min_level_db_floor = -20
db_delta = 5
silence_threshold = 0.01
min_silence_for_spec=0.001
max_vocal_for_spec=0.225,
min_syllable_length_s = 0.025
butter_min = 500
butter_max = 8000
spectral_range = [500, 8000]

hparams.n_fft = n_fft

hparams.win_length_ms = win_length_ms
hparams.hop_length_ms = hop_length_ms
hparams.butter_lowcut = butter_min
hparams.butter_highcut = butter_max
hparams.ref_level_db = ref_level_db
hparams.min_level_db = min_level_db

####
DATASET_ID = 'zebra_finch_gardner'
dataset = DataSet(DATASET_ID, hparams = hparams)
# Print sample dataset

dataset.sample_json
rate, data = load_wav(dataset.sample_json["wav_loc"])

hparams.sample_rate = rate

### Individual data file segmentation

import librosa
rate, data = load_wav(dataset.sample_json["wav_loc"])
np.min(data), np.max(data)

data = data / np.max(np.abs(data))
# filter data
data = butter_bandpass_filter(data, butter_min, butter_max, rate)

plt.plot(data)

# segment
results = dynamic_threshold_segmentation(
    data,
    rate,
    n_fft=n_fft,
    hop_length_ms=hop_length_ms,
    win_length_ms=win_length_ms,
    min_level_db_floor=min_level_db_floor,
    db_delta=db_delta,
    ref_level_db=ref_level_db,
    pre=pre,
    min_silence_for_spec=min_silence_for_spec,
    max_vocal_for_spec=max_vocal_for_spec,
    min_level_db=min_level_db,
    silence_threshold=silence_threshold,
    verbose=True,
    min_syllable_length_s=min_syllable_length_s,
    spectral_range=spectral_range,
)

plot_segmentations(
    results["spec"],
    results["vocal_envelope"],
    results["onsets"],
    results["offsets"],
    int(hparams.hop_length_ms),
    int(hparams.sample_rate),
    figsize=(15,5)
)
plt.show()

## Segment and plot
import joblib
import json
from avgn.utils.json import NoIndent, NoIndentEncoder


def segment_spec_custom(key, df, DT_ID, save=False, plot=False):
    # load wav
    rate, data = load_wav(df.data["wav_loc"])
    # filter data
    data = butter_bandpass_filter(data, butter_min, butter_max, rate)

    # segment
    results = dynamic_threshold_segmentation(
        data,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        min_level_db_floor=min_level_db_floor,
        db_delta=db_delta,
        ref_level_db=ref_level_db,
        pre=pre,
        min_silence_for_spec=min_silence_for_spec,
        max_vocal_for_spec=max_vocal_for_spec,
        min_level_db=min_level_db,
        silence_threshold=silence_threshold,
        verbose=True,
        min_syllable_length_s=min_syllable_length_s,
        spectral_range=spectral_range,
    )
    if results is None:
        return
    if plot:
        plot_segmentations(
            results["spec"],
            results["vocal_envelope"],
            results["onsets"],
            results["offsets"],
            hop_length_ms,
            rate,
            figsize=(15, 3)
        )
        plt.show()

    # save the results
    json_out = DATA_DIR / "processed" / (DATASET_ID + "_segmented") / DT_ID / "JSON" / (
        key + ".JSON"
    )

    json_dict = df.data.copy()

    json_dict["indvs"][list(df.data["indvs"].keys())[0]]["syllables"] = {
        "start_times": list(results["onsets"]),
        "end_times": list(results["offsets"]),
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)
    # save json
    if save:
        ensure_dir(json_out.as_posix())
        with open(json_out.as_posix(), "w") as json_file:
            json.dump(json_dict, json_file, cls=NoIndentEncoder, indent=2)
        json_file.close()
 #       print(json_txt, file=open(json_out.as_posix(), "w"))

    #print(json_txt)

    return results

indvs = np.array(['_'.join(list(i)) for i in dataset.json_indv])
np.unique(indvs)

DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
### Generate for two sample wav dataset
nex = 1
for indv in tqdm.tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv][:nex]

    joblib.Parallel(n_jobs=1, verbose=0)(
            joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], DT_ID, save=True, plot=True) 
                 for key in tqdm.tqdm(indv_keys, desc="files", leave=False)
        )

### Generate for full dataset
nex = -1
for indv in tqdm.tqdm(np.unique(indvs), desc="individuals"):
    print(indv)
    indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]

    joblib.Parallel(n_jobs=1, verbose=1)(
            joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], DT_ID, save=True, plot=False) 
                 for key in tqdm.tqdm(indv_keys, desc="files", leave=False)
        )   
### -----------------------------------------------------------------------------------------------######
# Create dataframe for zebra finch
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import tqdm
from joblib import Parallel, delayed
import umap
import pandas as pd

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir    
DATASET_ID = 'zebra_finch_gardner_segmented'   

from avgn.utils.hparams import HParams
from avgn.dataset import DataSet 

from avgn.signalprocessing.create_spectrogram_dataset import prepare_wav, create_label_df, get_row_audio

# Run the hparams code from previous segments
# hparams = HParams(
#     num_mel_bins = 32,
#     mel_lower_edge_hertz=250,
#     mel_upper_edge_hertz=12000,
#     butter_lowcut = 250,
#     butter_highcut = 12000,
#     ref_level_db = 20,
#     min_level_db = -50,
#     mask_spec = True,
#     win_length_ms = 10,
#     hop_length_ms = 2,
#     nex=-1,
#     n_jobs=-1,
#     verbosity = 1,
# )    

# create a dataset object based upon JSON segmented
dataset = DataSet(DATASET_ID, hparams = hparams)

from joblib import Parallel, delayed
n_jobs = 1; verbosity = 10

# See a sample dataset
dataset.sample_json
len(dataset.data_files)

# Create dataset based upon JSON
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(create_label_df)(
            dataset.data_files[key].data,
            hparams=dataset.hparams,
            labels_to_retain=[],
            unit="syllables",
            dict_features_to_retain = [],
            key = key,
        )
        for key in tqdm.tqdm(dataset.data_files.keys())
    )
syllable_df = pd.concat(syllable_dfs)
len(syllable_df)
# See dample dataset after json is converted to dataframe   
syllable_df[:3]  

# Get audio for dataset
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(get_row_audio)(
            syllable_df[syllable_df.key == key], 
            dataset.data_files[key].data['wav_loc'], 
            dataset.hparams
        )
        for key in tqdm.tqdm(syllable_df.key.unique())
    )
syllable_df = pd.concat(syllable_dfs)
len(syllable_df)
# See sample dataframe after adding audio data to dataframe
syllable_df[:3]  

# df_mask  = np.array([len(i) > 0 for i in tqdm.tqdm(syllable_df.audio.values)])
# syllable_df = syllable_df[np.array(df_mask)]
# syllable_df[:3]    # Sample dataframe

# sylls = syllable_df.audio.values
# import librosa
syllable_df['audio'] = [librosa.util.normalize(i) for i in syllable_df.audio.values]
sylls = syllable_df['audio'].values

### Plot time domain form of syllables
nrows = 5
ncols = 10
zoom = 2
fig, axs = plt.subplots(ncols=ncols, nrows = nrows,figsize = (ncols*zoom, nrows+zoom/1.5))
for i, syll in tqdm.tqdm(enumerate(sylls), total = nrows*ncols):
    ax = axs.flatten()[i]
    ax.plot(syll)
    if i == nrows*ncols -1:
        break
    
# Create Spectrograms
from avgn.visualization.spectrogram import draw_spec_set
from avgn.signalprocessing.create_spectrogram_dataset import make_spec, mask_spec, log_resize_spec, pad_spectrogram

syllables_wav = syllable_df.audio.values
syllables_rate = syllable_df.rate.values

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    # create spectrograms
    syllables_spec = parallel(
        delayed(make_spec)(
            syllable,
            rate,
            hparams=dataset.hparams,
            mel_matrix=dataset.mel_matrix,
            use_mel=True,
            use_tensorflow=False,
        )
        for syllable, rate in tqdm.tqdm(
            zip(syllables_wav, syllables_rate),
            total=len(syllables_rate),
            desc="getting syllable spectrograms",
            leave=False,
        )
    )   

hparams.ref_level_db = 80
spec_1 = make_spec(
    syllables_wav[10],
    syllables_rate[10],
    hparams=dataset.hparams,
    mel_matrix=dataset.mel_matrix,
    use_mel=True,
    use_tensorflow=False,
)

from matplotlib import cm
fig, ax = plt.subplots(figsize=(4, 5)) 
ax.matshow(spec_1,cmap=cm.afmhot)
ax.axis('off')

a = create_syllable_df(dataset,indvs[0],unit="syllables",log_scale_time=False)