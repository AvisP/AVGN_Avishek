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
from avgn.custom_parsing.bengalese_finch_sakata import generate_json_wav
from avgn.utils.paths import DATA_DIR

######## Load data in original Format

DATASET_ID = 'bengalese_finch_sakata'

# create a unique datetime identifier for the files output by this notebook
DT_ID = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
DT_ID

DSLOC = avgn.utils.paths.Path('I:/Avishek_segmentation/BrownBlue')
DSLOC

WAVFILES = list((DSLOC).expanduser().glob('*/[!.]*.wav')) 
len(WAVFILES), WAVFILES[0]

####### parse MAT and create wav/JSON
# import h5py as h5

last_indv = 'new_data'
wav_num = 0
for wav_file in tqdm.tqdm(WAVFILES):
    print(wav_file)
    indv = wav_file.parent.stem
    try:
        [song,rate] = librosa.core.load(wav_file)
    except:
        print(wav_file.parts[-1] + "failed")
        continue
    
    if indv != last_indv:
        wav_num = 0
        last_indv = indv
    else:
        wav_num += 1    
    
    generate_json_wav(indv, wav_file.parts[-1], wav_num, song, int(rate), DT_ID)
        
### -------------------------------------------------------------------------------------------------------######
        
## SEGMENTATION USING DYNAMIC THRESHOLDING
        
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='once')

from avgn.utils.audio import load_wav, read_wav
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.song_segmentation.dynamic_thresholding import dynamic_threshold_segmentation
from avgn.vocalsegmentation.dynamic_thresholding import plot_segmented_spec
from avgn.visualization.spectrogram import plot_segmentations

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
from avgn.utils.hparams import HParams
from avgn.dataset import DataSet

hparams = HParams(
    n_fft = 4096,
    mel_lower_edge_hertz=500,
    mel_upper_edge_hertz=11025,  # Should be sample_rate / 2 or less
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
win_length_ms=10
ref_level_db=50
pre=0.97
min_level_db=-120
min_level_db_floor = -20
db_delta = 5
silence_threshold = 0.01
min_silence_for_spec=0.001
max_vocal_for_spec=0.33,
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
hparams.min_level_db_floor = min_level_db_floor
hparams.db_delta = db_delta
hparams.min_silence_for_spec = min_silence_for_spec
hparams.max_vocal_for_spec = max_vocal_for_spec
hparams.silence_threshold = silence_threshold


syll_size = 128

hparams.mel_fiter = True # should a mel filter be used?
hparams.num_mels = syll_size, # how many channels to use in the mel-spectrogram (eg 128)
hparams.fmin =  300, # low frequency cutoff for mel filter
hparams.fmax = None, # high frequency cutoff for mel filter

hparams.power = 1.5;

### Add other parameters from here if required  ####
# # spectrogram inversion
#     'max_iters':200,
#     'griffin_lim_iters':60,
#     'power':1.5,

#     # Thresholding out noise
#     'mel_noise_filt' : 0.15, # thresholds out low power noise in the spectrum - higher numbers will diminish inversion quality

# # Vocal Envelope
#     'smoothing' : 'gaussian', # 'none', 
#     'envelope_signal' : "spectrogram", # spectrogram or waveform, what to get the vocal envelope from
#     'gauss_sigma_s': .0001,
#     'FOI_min': 4, # minimum frequency of interest for vocal envelope (in terms of mel)
#     'FOI_max': 24, # maximum frequency of interest for vocal envelope (in terms of mel)
    
#     # Silence Thresholding
#     'silence_threshold' : 0, # normalized threshold for silence
#     'min_len' : 5., # minimum length for a vocalization (fft frames)
#     'power_thresh': .3, # Threshold for which a syllable is considered to be quiet weak and is probably noise

#     # Syllabification
#     'min_syll_len_s' : 0.03, # minimum length for a syllable
#     'segmentation_rate': 0.0,#0.125, # rate at which to dynamically raise the segmentation threshold (ensure short syllables)
#     'threshold_max': 0.25,
#     'min_num_sylls': 20, # min number of syllables to be considered a bout
#     'slow_threshold':0.0,#0.02, # second slower threshold
#     'max_size_syll': syll_size, # the size of the syllable
#     'resize_samp_fr': int(syll_size*5.0), # (frames/s) the framerate of the syllable (in compressed spectrogram time components)
    
#     # Second pass syllabification
#     'second_pass_threshold_repeats':50, # the number of times to repeat the second pass threshold
#     'ebr_min': 0.05, # expected syllabic rate (/s) low 
#     'ebr_max':  0.2, # expected syllabic rate (/s) high 
#     'max_thresh':  0.02, # maximum pct of syllabic envelope to threshold at in second pass
#     'thresh_delta':  0.005, # delta change in threshold to match second pass syllabification
#     'slow_threshold': 0.005, # starting threshold for second pass syllabification
    
#     'pad_length' : syll_size, # length to pad spectrograms to 

####
DATASET_ID = 'bengalese_finch_sakata'
dataset = DataSet(DATASET_ID, hparams = hparams)
import numpy as np
# Print sample dataset

dataset.sample_json
rate, data = load_wav(dataset.sample_json["wav_loc"])

hparams.sample_rate = rate

### Individual data file segmentation

# import librosa
# rate, data = load_wav(dataset.sample_json["wav_loc"])
# np.min(data), np.max(data)
# data = data / np.max(np.abs(data))

# filter data
data = butter_bandpass_filter(data, butter_min, butter_max, rate)

plt.plot(data)

# from avgn.visualization.spectrogram import plot_spec, visualize_spec
# from avgn.song_segmentation.spectrogramming import spectrogram_nn
# from avgn.song_segmentation.dynamic_thresholding import _normalize

# hparams.ref_level_db = 10


# spec_orig = spectrogram_nn(data,
#                            hparams)



# plot_spec(
#     spec_orig,
#     fig=None,
#     ax=None,
#     rate=None,
#     hop_len_ms=None,
#     cmap=plt.cm.afmhot,
#     show_cbar=True,
#     figsize=(20, 6),
# )

# fft_rate = 1000 / hop_length_ms

# # def norm(x, _type="zero_one"):
# #     return (x - np.min(x)) / (np.max(x) - np.min(x))

# # hparams.min_level_db = 50
# # spec = norm(_normalize(spec_orig, hparams.min_level_db))

# plot_spec(
#     spec,
#     fig=None,
#     ax=None,
#     rate=None,
#     hop_len_ms=None,
#     cmap=plt.cm.afmhot,
#     show_cbar=True,
#     figsize=(20, 6),
# )



# segment
results = dynamic_threshold_segmentation(data,
                                          hparams,
                                          verbose=True,
                                          min_syllable_length_s=min_syllable_length_s,
                                          spectral_range=spectral_range)

# from avgn.vocalsegmentation.dynamic_thresholding import dynamic_threshold_segmentation
# from avgn.vocalsegmentation.utils import spectrogram_nn
# from avgn.visualization.spectrogram import plot_spec

# spec_orig = spectrogram_nn(
#         data,
#         rate,
#         n_fft=n_fft,
#         hop_length_ms=hop_length_ms,
#         win_length_ms=win_length_ms,
#         ref_level_db=ref_level_db,
#         pre=pre,
#     )
# fft_rate = 1000 / hop_length_ms
    
# plot_spec(
#     spec_orig,
#     fig=None,
#     ax=None,
#     rate=rate,
#     hop_len_ms=hop_length_ms,
#     cmap=plt.cm.afmhot,
#     show_cbar=True,
#     figsize=(20, 6),
# )


# results = dynamic_threshold_segmentation(
#     data,
#     rate,
#     n_fft=n_fft,
#     hop_length_ms=hop_length_ms,
#     win_length_ms=win_length_ms,
#     min_level_db_floor=min_level_db_floor,
#     db_delta=db_delta,
#     ref_level_db=ref_level_db,
#     pre=pre,
#     min_silence_for_spec=min_silence_for_spec,
#     max_vocal_for_spec=max_vocal_for_spec,
#     min_level_db=min_level_db,
#     silence_threshold=silence_threshold,
#     verbose=True,
#     min_syllable_length_s=min_syllable_length_s,
#     spectral_range=spectral_range,
# )

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