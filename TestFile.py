# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:13:19 2020

@author: SakataWoolley
"""

from tqdm.autonotebook import tqdm
from avgn.song_segmentation.spectrogramming import spectrogram_nn, _normalize
from avgn.signalprocessing.spectrogramming import spectrogram
from avgn.song_segmentation.dynamic_thresholding import onsets_offsets
import numpy as np
from scipy import ndimage
from scipy.signal import butter, lfilter
from avgn.utils.hparams import HParams
import librosa
from avgn.utils.audio import load_wav, read_wav
from avgn.visualization.spectrogram import plot_spec
from avgn.signalprocessing.filtering import butter_bandpass_filter
import matplotlib.pyplot as plt
from avgn.visualization.spectrogram import plot_segmentations


### General


def int16tofloat32(data):
    return np.array(data / 32768).astype("float32")


def norm(x, _type="zero_one"):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def dynamic_threshold_segmentation(
    vocalization,
    hparams,    
    verbose=False,
    min_syllable_length_s=0.1,
    spectral_range=None):
    """
    computes a spectrogram from a waveform by iterating through thresholds
         to ensure a consistent noise level
    
    Arguments:
        vocalization {[type]} -- waveform of song
        rate {[type]} -- samplerate of datas
    
    Keyword Arguments:
        min_level_db {int} -- default dB minimum of spectrogram (threshold anything below) (default: {-80})
        min_level_db_floor {int} -- highest number min_level_db is allowed to reach dynamically (default: {-40})
        db_delta {int} -- delta in setting min_level_db (default: {5})
        n_fft {int} -- FFT window size (default: {1024})
        hop_length_ms {int} -- number audio of frames in ms between STFT columns (default: {1})
        win_length_ms {int} -- size of fft window (ms) (default: {5})
        ref_level_db {int} -- reference level dB of audio (default: {20})
        pre {float} -- coefficient for preemphasis filter (default: {0.97})
        min_syllable_length_s {float} -- shortest expected length of syllable (default: {0.1})
        min_silence_for_spec {float} -- shortest expected length of silence in a song (used to set dynamic threshold) (default: {0.1})
        silence_threshold {float} -- threshold for spectrogram to consider noise as silence (default: {0.05})
        max_vocal_for_spec {float} -- longest expected vocalization in seconds  (default: {1.0})
        spectral_range {[type]} -- spectral range to care about for spectrogram (default: {None})
        verbose {bool} -- display output (default: {False})
    
    
    Returns:
        [results] -- [dictionary of results]
    """
    rate = hparams.sample_rate
    n_fft = hparams.n_fft
    hop_length_ms=hparams.hop_length_ms
    win_length_ms=hparams.win_length_ms
    min_level_db=hparams.min_level_db
    min_level_db_floor=hparams.min_level_db_floor
    db_delta=hparams.db_delta   
    ref_level_db=hparams.ref_level_db
    pre=hparams.preemphasis
    min_silence_for_spec = hparams.min_silence_for_spec
    max_vocal_for_spec = hparams.max_vocal_for_spec
    silence_threshold = hparams.silence_threshold

    # does the envelope meet the standards necessary to consider this a bout
    envelope_is_good = False

    # make a copy of the hyperparameters

    # make a copy of the original spectrogram
    # spec_orig = spectrogram_nn(
    #     vocalization,
    #     rate,
    #     n_fft=n_fft,
    #     hop_length_ms=hop_length_ms,
    #     win_length_ms=win_length_ms,
    #     ref_level_db=ref_level_db,
    #     pre=pre,
    # )
    # spec_orig = spectrogram_nn(vocalization,
    #             hparams)
    
    spec_orig = spectrogram(vocalization,
                            rate,
                            hparams)
    
    
    fft_rate = 1000 / hop_length_ms

    if spectral_range is not None:
        spec_bin_hz = (rate / 2) / np.shape(spec_orig)[0]
        spec_orig = spec_orig[
            int(spectral_range[0] / spec_bin_hz) : int(spectral_range[1] / spec_bin_hz),
            :,
        ]

    # loop through possible thresholding configurations starting at the highest
    for _, mldb in enumerate(
        tqdm(
            np.arange(min_level_db, min_level_db_floor, db_delta),
            leave=False,
            disable=(not verbose),
        )
    ):
        # set the minimum dB threshold
        min_level_db = mldb
        # normalize the spectrogram
        # spec = norm(_normalize(spec_orig, min_level_db=min_level_db))
        spec = norm(spec_orig)

        # subtract the median
        spec = spec - np.median(spec, axis=1).reshape((len(spec), 1))
        spec[spec < 0] = 0

        # get the vocal envelope
        vocal_envelope = np.max(spec, axis=0) * np.sqrt(np.mean(spec, axis=0))
        # normalize envelope
        vocal_envelope = vocal_envelope / np.max(vocal_envelope)

        # Look at how much silence exists in the signal
        onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold) / fft_rate
        onsets_sil, offsets_sil = (
            onsets_offsets(vocal_envelope <= silence_threshold) / fft_rate
        )

        # if there is a silence of at least min_silence_for_spec length,
        #  and a vocalization of no greater than max_vocal_for_spec length, the env is good
        if len(onsets_sil) > 0:
            # frames per second of spectrogram

            # longest silences and periods of vocalization
            max_silence_len = np.max(offsets_sil - onsets_sil)
            max_vocalization_len = np.max(offsets - onsets)
            if verbose:
                print("longest silence", max_silence_len)
                print("longest vocalization", max_vocalization_len)

            if max_silence_len > min_silence_for_spec:
                if max_vocalization_len < max_vocal_for_spec:
                    envelope_is_good = True
                    break
        if verbose:
            print("Current min_level_db: {}".format(min_level_db))

    if not envelope_is_good:
        return None

    onsets, offsets = onsets_offsets(vocal_envelope > silence_threshold) / fft_rate

    # threshold out short syllables
    length_mask = (offsets - onsets) >= min_syllable_length_s

    return {
        "spec": spec,
        "vocal_envelope": vocal_envelope.astype("float32"),
        "min_level_db": min_level_db,
        "onsets": onsets[length_mask],
        "offsets": offsets[length_mask],
    }

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

rate, data = load_wav('I:/avgn_paper-vizmerge/data/processed/bengalese_finch_sakata/2020-04-29_21-12-51/WAV/br82bl42_0001.WAV')
rate, data = load_wav('I:/avgn_paper-vizmerge/data/processed/bengalese_finch_sakata/2020-04-29_21-12-51/WAV/tutor_bl5w5_0000.WAV')
hparams.sample_rate = rate

data = butter_bandpass_filter(data, butter_min, butter_max, rate)
plt.plot(data)

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

# # segment
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


