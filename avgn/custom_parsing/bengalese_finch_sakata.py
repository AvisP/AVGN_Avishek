from avgn.utils.json import NoIndentEncoder
import librosa
import json
import avgn
from avgn.utils.paths import DATA_DIR, ensure_dir
import os
# import pathlib
import pathlib2


DATASET_ID = 'bengalese_finch_sakata'

def generate_json_wav(indv, actual_filename, wav_num, song, sr, DT_ID):
    
    wav_duration = len(song) / sr

    wav_stem = indv + "_" + str(wav_num).zfill(4)

    json_out = ( DATA_DIR / "processed" / DATASET_ID / DT_ID / "JSON" / (wav_stem + ".JSON") )
    # json_out = pathlib2.PureWindowsPath(DATA_DIR).joinpath("processed",DATASET_ID,DT_ID,"JSON",(wav_stem+".JSON"))
    # head,tail = os.path.split(json_out)
    # ensure_dir(pathlib.PureWindowsPath(json_out).parents[0])
    ensure_dir(json_out)
    
        
    # wav_out = pathlib2.PureWindowsPath(DATA_DIR).joinpath("processed",DATASET_ID,DT_ID,"WAV",(wav_stem+".WAV"))
    wav_out = ( DATA_DIR / "processed" / DATASET_ID / DT_ID / "WAV" / (wav_stem + ".WAV"))
    # head,tail = os.path.split(wav_out)
    # ensure_dir(pathlib.PureWindowsPath(wav_out).parents[0])
    # print('Wave Out directory')
    ensure_dir(wav_out)

    # make json dictionary
    json_dict = {}
    # add species
    json_dict["species"] = "Lonchura striata domestica"
    json_dict["common_name"] = "Bengalese finch"
    json_dict["original_filename"] = actual_filename
    json_dict["wav_loc"] = wav_out.as_posix()
    # json_dict["noise_loc"] = noise_out.as_posix()

    # rate and length
    json_dict["samplerate_hz"] = sr
    json_dict["length_s"] = wav_duration
    json_dict["wav_num"] = wav_num

    # add syllable information
    json_dict["indvs"] = {
        indv: {"motifs": {"start_times": [0.0], "end_times": [wav_duration]}}
    }

    json_txt = json.dumps(json_dict, cls=NoIndentEncoder, indent=2)

    # save wav file
    print(wav_out)
    avgn.utils.paths.ensure_dir(wav_out)
    librosa.output.write_wav(wav_out, y=song, sr=int(sr), norm=True)

    # save json
    avgn.utils.paths.ensure_dir(json_out.as_posix())
    print(json_txt, file=open(json_out.as_posix(), "w"))