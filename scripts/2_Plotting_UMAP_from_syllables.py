# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 20:02:36 2020

@author: Avishek Paul
"""

### Script 2 Plotting UMAPS

# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm.autonotebook import tqdm
# import pandas as pd
# # from cuml.manifold.umap import UMAP as cumlUMAP
# from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
# from avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms

# DATASET_ID = 'zebra_finch_gardner_segmented'

# from avgn.visualization.projections import (
#     scatter_projections,
#     draw_projection_transitions,
# )

# DATASET_ID = 'zebra_finch_gardner_segmented'
# df_loc =  DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf.pickle'

# syllable_df = pd.read_pickle(df_loc)
# del syllable_df['audio']

# syllable_df[:3]
# np.shape(syllable_df.spectrogram.values[0])

# ensure_dir(DATA_DIR / 'embeddings' / DATASET_ID / 'full')

import numpy as np
import matplotlib.pyplot as plt
import tqdm
from joblib import Parallel, delayed
import pandas as pd

import warnings; warnings.simplefilter('ignore')

from avgn.utils.paths import DATA_DIR, most_recent_subdirectory, ensure_dir
from avgn.signalprocessing.create_spectrogram_dataset import flatten_spectrograms
from avgn.visualization.spectrogram import draw_spec_set

DATASET_ID = 'zebra_finch_gardner_segmented'

from avgn.visualization.projections import (
    scatter_projections,
    draw_projection_transitions,
)

df_loc =  DATA_DIR / 'syllable_dfs' / DATASET_ID / 'zf.pickle'
df_loc

syllable_df = pd.read_pickle(df_loc)
syllable_df[:3]

len(syllable_df)

# Histograms of syllables
fig, axs = plt.subplots(ncols=4, figsize=(24,6))
axs[0].hist([np.max(i) for i in syllable_df.spectrogram.values], bins=50);axs[0].set_title('max')
axs[1].hist([np.sum(i) for i in syllable_df.spectrogram.values], bins=50);axs[1].set_title('sum')
axs[2].hist((syllable_df.end_time - syllable_df.start_time).values, bins = 50); axs[2].set_title('len')
axs[3].hist([np.min(i) for i in syllable_df.spectrogram.values], bins=50);axs[3].set_title('min')

# Cluster 
len(syllable_df)

def norm(x):
    return (x-np.min(x)) / (np.max(x) - np.min(x))

specs = list(syllable_df.spectrogram.values)
specs = [norm(i) for i in specs]
specs_flattened = flatten_spectrograms(specs)
np.shape(specs_flattened)

# Variation across individuals ( Not complete)
# syllable_df.indv.unique()
# from cuml.manifold.umap import UMAP as cumlUMAP
import umap
from avgn.visualization.projections import scatter_spec
from avgn.utils.general import save_fig
from avgn.utils.paths import FIGURE_DIR, ensure_dir
from avgn.visualization.quickplots import draw_projection_plots
ensure_dir(FIGURE_DIR / 'zf')

    
indv_dfs = {}    
for indvi, indv in enumerate(tqdm.tqdm(syllable_df.indv.unique())):
    indv_dfs[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs[indv] = indv_dfs[indv].sort_values(by=["key", "start_time"])
    print(indv, len(indv_dfs[indv]))
    specs = [norm(i) for i in indv_dfs[indv].spectrogram.values]
    
    # sequencing
    indv_dfs[indv]["syllables_sequence_id"] = None
    indv_dfs[indv]["syllables_sequence_pos"] = None
    for ki, key in enumerate(indv_dfs[indv].key.unique()):
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_id"] = ki
        indv_dfs[indv].loc[indv_dfs[indv].key == key, "syllables_sequence_pos"] = np.arange(
            np.sum(indv_dfs[indv].key == key)
        )
        
    # umap
    specs_flattened = flatten_spectrograms(specs)
#    cuml_umap = cumlUMAP(min_dist=0.5)
#    z = list(cuml_umap.fit_transform(specs_flattened))
    fit = umap.UMAP(n_neighbors=30,
        #min_dist=0.0,
        n_components=2,
        random_state=42,)
    
    z = list(fit.fit_transform(specs_flattened))
    indv_dfs[indv]["umap"] = z
    
import hdbscan
    
for indv in tqdm.tqdm(indv_dfs.keys()):
    ### cluster
    #break
    z = list(indv_dfs[indv]["umap"].values)
    # HDBSCAN UMAP
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(len(z) * 0.01), # the smallest size we would expect a cluster to be
        min_samples=1, # larger values = more conservative clustering
    )
    clusterer.fit(z);
    indv_dfs[indv]['hdbscan_labels'] = clusterer.labels_   
    
for indv in tqdm.tqdm(indv_dfs.keys()):
    draw_projection_plots(indv_dfs[indv], label_column="hdbscan_labels")
import pickle    
with open('Indv_df.pickle', 'wb') as handle:
    pickle.dump(indv_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
    
---------------------------------------------------------------------------------------------------------  

for indv in tqdm.tqdm(syllable_df.indv.unique()):
    indv_df = syllable_df[syllable_df.indv == indv]
    print(indv, len(indv_df))

    specs = list(indv_df.spectrogram.values)
    draw_spec_set(specs, zoom=1, maxrows=10, colsize=25)
    
    specs_flattened = flatten_spectrograms(specs)
    
    # cuml_umap = cumlUMAP(min_dist=0.25)
    # z = list(cuml_umap.fit_transform(specs_flattened))
    
    fit = umap.UMAP(n_neighbors=30,
        #min_dist=0.0,
        n_components=2,
        random_state=42,)
    z = list(fit.fit_transform(specs_flattened))
    indv_df["umap"] = z

    indv_df["syllables_sequence_id"] = None
    indv_df["syllables_sequence_pos"] = None
    indv_df = indv_df.sort_values(by=["key", "start_time"])
    for ki, key in enumerate(indv_df.key.unique()):
        indv_df.loc[indv_df.key == key, "syllables_sequence_id"] = ki
        indv_df.loc[indv_df.key == key, "syllables_sequence_pos"] = np.arange(
            np.sum(indv_df.key == key)
        )

    draw_projection_plots(indv_df, label_column="indv")
    
    scatter_spec(
        np.vstack(z),
        specs,
        column_size=15,
        #x_range = [-5.5,7],
        #y_range = [-10,10],
        pal_color="hls",
        color_points=False,
        enlarge_points=20,
        figsize=(10, 10),
        scatter_kwargs = {
            #'labels': list(indv_df.labels.values),
            'alpha':0.25,
            's': 1,
            'show_legend': False
        },
        matshow_kwargs = {
            'cmap': plt.cm.Greys
        },
        line_kwargs = {
            'lw':1,
            'ls':"solid",
            'alpha':0.25,
        },
        draw_lines=True
    );
    
    save_fig(FIGURE_DIR / 'zf' / ('zf_'+indv), dpi=300, save_png=True, save_jpg=False)

    plt.show()
