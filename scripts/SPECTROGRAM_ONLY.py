# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:56:10 2020

@author: AvishekPaul
"""
        
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from avgn.utils.audio import load_wav, read_wav
from avgn.signalprocessing.filtering import butter_bandpass_filter
from avgn.signalprocessing.spectrogramming import spectrogram, norm
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
max_vocal_for_spec=0.483,
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

mypath = 'F:\\data_for_avishek\\blk12'
# file_current = 'blk12_undir1_new.wav'
file_current = 'motif.18722.3583blk12_dir1_new.wav'
file_current = 'motif.5238.458blk12_undir3_new.wav'
file_current = 'motif.1454.1723blk12_dir5_new.wav'

mypath = r'F:\data_for_avishek\blkorng_new'
file_current = 'motif.3044.8299dir5_b_new.wav'

mypath ='F:\data_for_avishek\Stimuli\yel55red72\hctsa data'
file_current = 'dirmotif_yelred.20080611.0020.wav'
# file_current = 'motif.9231.1565yelred_dir1.wav'

mypath = 'F:\data_for_avishek\orngpnk23-24_stim'
file_current = 'motif.1489.9546o23p24_dir1.wav'
file_current = 'motif.1592.3583o23p24_dir9.wav'

mypath= 'F:\data_for_avishek\Dir_Undir\done\prpred'
file_current = 'motif.1883.8095prpred47-36_dir6_new.wav'
file_current = 'motif.578.7755prpred47-36_dir4_new.wav'
file_current = 'motif.1467.551prpred47-36_dir1_new.wav'

mypath= 'F:\data_for_avishek\Dir_Undir\done'
file_current = 'motif.10384.8299prpred47-36_dir1_new.wav'

mypath = 'F:\data_for_avishek\Dir_Undir\done\o122p123'
file_current = 'motif.1614.898o22p23_dir5.wav'

mypath = 'F:\data_for_avishek\Dir_Undir\done\o121p122'
file_current = 'motif.2210.9524o21p22_dir7_new.wav'

mypath = r'F:\data_for_avishek\motif_all_dataset'
file_current = 'motif.2636.5533blublu50-58_dir4_new.wav'
file_current = 'motif.53.8549blublu50-58_dir5_new.wav'

mypath  = r'F:\data_for_avishek\Stimuli\blu19org61\hctsadata'
file_current = 'directed_b19o61.3-041103.385.wav'

#rate, data_loaded = wavfile.read(mypath+'\\'+file_current)
rate, data = load_wav(mypath+'\\'+file_current)

# filter data
data = butter_bandpass_filter(data, butter_min, butter_max, rate)
plt.plot(data)


hparams.ref_level_db = 70
spec_orig = spectrogram(data,
                            rate,
                            hparams)

figsize = (5,3)
# norm = matplotlib.colors.Normalize(vmin=np.min(spec_dir), vmax=np.max(spec_dir))

# current_map = cm.get_cmap('RdPu',512)
current_map = cm.get_cmap('Blues',512)
# current_map = cm.get_cmap('Reds',512)
# current_map = cm.get_cmap('Oranges',512)
boxcolor = current_map(512)

fig, ax = plt.subplots(figsize=figsize)
# fig.patch.set_facecolor('black')
plot_spec(spec_orig, fig, ax,cmap=current_map,show_cbar=False);
ax.axvline(x=0,linewidth=1,color=boxcolor)
ax.axvline(x=spec_orig.shape[1]-2,linewidth=1,color=boxcolor)
ax.axhline(y=10,linewidth=1,color=boxcolor)
ax.axhline(y=spec_orig.shape[0]-10,linewidth=1,color=boxcolor)
ax.axis('off')
