# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:56:10 2020

@author: AvishekPaul
"""
#####  SCRIPT 1

from avgn.utils.general import prepare_env
prepare_env()

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
import numpy as np
import warnings
warnings.filterwarnings(action='once')

from avgn.utils.audio import load_wav, read_wav
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.signalprocessing.spectrogramming import spectrogram
from avgn.song_segmentation.dynamic_thresholding import norm, dynamic_threshold_segmentation
from avgn.vocalsegmentation.dynamic_thresholding import plot_segmented_spec
from avgn.visualization.spectrogram import plot_segmentations
from avgn.visualization.spectrogram import plot_spec, visualize_spec

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
max_vocal_for_spec=0.49,
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

# Print sample dataset

dataset.sample_json
rate, data = load_wav(dataset.sample_json["wav_loc"])

mypath = r'I:\avgn_paper-vizmerge\data\processed\bengalese_finch_sakata\2020-04-29_21-12-51\WAV'
# file_current = 'br81bl41_0016.wav'
file_current = 'br82bl42_0016.wav'
file_current = 'tutor_bl5w5_0017.WAV'


rate, data_loaded = load_wav(mypath+'\\'+file_current)
data = data_loaded
times = np.linspace(0,len(data)/rate,len(data));


# filter data
data = butter_bandpass_filter(data, butter_min, butter_max, rate)
plt.plot(times,data)


hparams.ref_level_db = 90
spec_orig = spectrogram(data,
                            rate,
                            hparams)
plot_spec(
    norm(spec_orig),
    fig=None,
    ax=None,
    rate=None,
    hop_len_ms=None,
    cmap=plt.cm.afmhot,
    show_cbar=True,
    figsize=(20, 6),
)

# segment
results = dynamic_threshold_segmentation(data,
                                          hparams,
                                          verbose=True,
                                          min_syllable_length_s=min_syllable_length_s,
                                          spectral_range=spectral_range)

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

# Function for batch processing all segments
import joblib
import json
from avgn.utils.json import NoIndent, NoIndentEncoder

def segment_spec_custom(key, df, DT_ID, save=False, plot=False):
    # load wav
    rate, data = load_wav(df.data["wav_loc"])
    # filter data
    data = butter_bandpass_filter(data, butter_min, butter_max, rate)

    # segment
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
    
    results = dynamic_threshold_segmentation(data,
                                          hparams,
                                          verbose=True,
                                          min_syllable_length_s=min_syllable_length_s,
                                          spectral_range=spectral_range)
    
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
            joblib.delayed(segment_spec_custom)(key, dataset.data_files[key], DT_ID, save=True, plot=False) 
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
    
# If some of the files did not preocess properly and want to re run them
# Figure out indidual key first from indv_keys 
# Find out the indv string by executing
for indv in tqdm.tqdm(np.unique(indvs), desc="individuals"):
    print(indv)    
indv_keys = np.array(list(dataset.data_files.keys()))[indvs == indv]
# USing the indv_keys run the following command
key = 'tutor_bl5w5_0030'
segment_spec_custom(key, dataset.data_files[key], DT_ID, save=True, plot=False)

# DT_ID = '2020-05-07_16-26-29'
   
# Create dataframe for zebra finch
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import tqdm
from joblib import Parallel, delayed
import umap
import pandas as pd

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir    
DATASET_ID = 'bengalese_finch_sakata_segmented'   

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
import librosa
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

### Section for checking how sample spectrograms looks like with hparams settings
### and change them as necessary
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

###### 

## Run sepctrograms with these hparams settings for whole dataset

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
    
# Rescale spectrogram (using log scaling)
log_scaling_factor = 4
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
        for spec in tqdm.tqdm(syllables_spec, desc="scaling spectrograms", leave=False)
    )    
# Check to see how the specrograms look like after rescaling
draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25)

# Pad spectrograms
syll_lens = [np.shape(i)[1] for i in syllables_spec]
pad_length = np.max(syll_lens)
syllable_df[:3]

import seaborn as sns
with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:

    syllables_spec = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm.tqdm(
            syllables_spec, desc="padding spectrograms", leave=False
        )
    )
# Check to see how the specrograms look like after padding
draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25)

np.shape(syllables_spec)
syllable_df['spectrogram'] = syllables_spec
syllable_df[:3]

# View syllables per individual

for indv in np.sort(syllable_df.indv.unique()):
    print(indv, np.sum(syllable_df.indv == indv))
    specs = np.array([i/np.max(i) for i in syllable_df[syllable_df.indv == indv].spectrogram.values])
    specs[specs<0] = 0
    draw_spec_set(specs, zoom=2,
                  maxrows=16, 
                  colsize=25,
                  fig_title=indv,
                  num_indv=str(np.sum(syllable_df.indv == indv)))

save_loc = DATA_DIR / 'syllable_dfs' / DATASET_ID / 'bf_sakata_Bluebrown.pickle'
ensure_dir(save_loc)
syllable_df.to_pickle(save_loc)