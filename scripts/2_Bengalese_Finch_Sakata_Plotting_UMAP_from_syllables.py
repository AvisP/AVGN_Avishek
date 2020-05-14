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

DATASET_ID = 'bengalese_finch_sakata_segmented'

from avgn.visualization.projections import (
    scatter_projections,
    draw_projection_transitions,
)
from avgn.visualization.network_graph import plot_network_graph

df_loc =  DATA_DIR / 'syllable_dfs' / DATASET_ID / 'bf_sakata_Bluebrown.pickle'
df_loc

syllable_df = pd.read_pickle(df_loc)
syllable_df[:3]

len(syllable_df)

# Histograms of syllables
fig, axs = plt.subplots(ncols=4, figsize=(24,6))
axs[0].hist([np.max(i) for i in syllable_df.spectrogram.values], bins=50);axs[0].set_title('max spec value' )
axs[1].hist([np.sum(i) for i in syllable_df.spectrogram.values], bins=50);axs[1].set_title('sum spec value')
axs[2].hist((syllable_df.end_time - syllable_df.start_time).values, bins = 50); axs[2].set_title('syllable len')
axs[3].hist([np.min(i) for i in syllable_df.spectrogram.values], bins=50);axs[3].set_title('min spec value')

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
ensure_dir(FIGURE_DIR / 'bf_sakata_Bluebrown')

    
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
    
indv_dfs_backup = indv_dfs


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
    
#### Save the dataframe    
import pickle 
filename_pickle = 'bf_sakata_Bluebrown.pickle'
   
with open(filename_pickle, 'wb') as handle:
    pickle.dump(indv_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
    
#### Do different plots
    
for indv in tqdm.tqdm(indv_dfs.keys()):
    # print(indv)
    draw_projection_plots(indv_dfs[indv], label_column="indv")
 
indv = 'tutor_bl5w5' 
draw_projection_plots(indv_dfs[indv], label_column="hdbscan_labels")

indv_df = indv_dfs[indv]
fig, axs, main_ax = scatter_spec(
        np.vstack(indv_df.umap.values),
        indv_df.spectrogram.values,
        column_size=8,
        #x_range = [-5.5,7],
        #y_range = [-10,10],
        pal_color="hls",
        color_points=True,
        enlarge_points=0,
        figsize=(10, 10),
        scatter_kwargs = {
            #'labels': list(indv_df.phrase.values),
            'alpha':0.25,
            's': 1,
            'show_legend': False,
        },
        matshow_kwargs = {
            'cmap': plt.cm.afmhot
        },
        line_kwargs = {
            'lw':3,
            'ls':"dashed",
            'alpha':0.25,
        },
        draw_lines=True,
        n_subset= 1000,
        border_line_width = 3,
        show_scatter=True,
    );
fig.suptitle(indv_df['indv'].values[0])
    
import pickle    
with open('Indv_df.pickle', 'wb') as handle:
    pickle.dump(indv_dfs, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('filename.pickle', 'rb') as handle:
    b = pickle.load(handle)
  
#### Separate diagram plotting
    
# indv = 'tutor_bl5w5'
# indv = 'br81bl41'
indv = 'br82bl42'

# syllable_df_plot = indv_dfs[indv]

syllable_df_plot = indv_dfs2[indv] # indv_df2 for umapped together
                                    # indv_df for separate umaping
projection_column = "umap"
label_column="hdbscan_labels"
figsize=(10,10)

fig, axs = plt.subplots(ncols=1, figsize=figsize)
    # plot scatter
    # ax = axs[0]
scatter_projections(
    projection=np.array(list(syllable_df_plot[projection_column].values)),
    labels=syllable_df_plot[label_column].values,
    ax=axs,
    color_palette = "cubehelix",
    alpha=0.7,
    show_legend = False,
) 
axs.set_title(syllable_df_plot['indv'].values[0])
axs.set_xlim(xmin=-8.5,xmax=8.5)
axs.set_ylim(ymin=-8,ymax=8)

fig, axs = plt.subplots(ncols=1, figsize=figsize)

draw_projection_transitions(
        projections=np.array(list(syllable_df_plot[projection_column].values)),
        sequence_ids=syllable_df_plot["syllables_sequence_id"],
        sequence_pos=syllable_df_plot["syllables_sequence_pos"],
        ax=axs,
        alpha=0.2,
        range_pad = 0.05,
    )
axs.set_title(syllable_df_plot['indv'].values[0])
axs.set_xlim(xmin=-8.5,xmax=8.5)
axs.set_ylim(ymin=-8,ymax=8)

fig, axs = plt.subplots(ncols=1, figsize=figsize)

elements = syllable_df_plot[label_column].values
projections = np.array(list(syllable_df_plot[projection_column].values))
sequence_ids = np.array(syllable_df_plot["syllables_sequence_id"])
plot_network_graph(
    elements, 
    projections, 
    sequence_ids, 
    color_palette="cubehelix", 
    ax=axs,
)
axs.set_title(syllable_df_plot['indv'].values[0])
axs.set_xlim(xmin=-8.5,xmax=8.5)
axs.set_ylim(ymin=-8,ymax=8)
    # ax.axis("off")
  
##### UMAP taking all together

specs_all = [norm(i) for i in syllable_df.spectrogram.values]  
specs_flattened_all = flatten_spectrograms(specs_all)
fit_all = umap.UMAP(n_neighbors=30,
        #min_dist=0.0,
        n_components=2,
        random_state=42,)
z_all = list(fit_all.fit_transform(specs_flattened_all))
syllable_df['umap'] = z_all

indv_dfs2 = {}  
for indvi, indv in enumerate(tqdm.tqdm(syllable_df.indv.unique())):
    indv_dfs2[indv] = syllable_df[syllable_df.indv == indv]
    indv_dfs2[indv] = indv_dfs2[indv].sort_values(by=["key", "start_time"])
    for ki, key in enumerate(indv_dfs2[indv].key.unique()):
        indv_dfs2[indv].loc[indv_dfs2[indv].key == key, "syllables_sequence_id"] = ki
        indv_dfs2[indv].loc[indv_dfs2[indv].key == key, "syllables_sequence_pos"] = np.arange(
            np.sum(indv_dfs2[indv].key == key)
        )

import hdbscan
    
for indv in tqdm.tqdm(indv_dfs2.keys()):
    ### cluster
    #break
    z = list(indv_dfs2[indv]["umap"].values)
    # HDBSCAN UMAP
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(len(z) * 0.01), # the smallest size we would expect a cluster to be
        min_samples=1, # larger values = more conservative clustering
    )
    clusterer.fit(z);
    indv_dfs2[indv]['hdbscan_labels'] = clusterer.labels_  
    
# ---------------------------------------------------------------------------------------------------------  

# for indv in tqdm.tqdm(syllable_df.indv.unique()):
#     indv_df = syllable_df[syllable_df.indv == indv]
#     print(indv, len(indv_df))

#     specs = list(indv_df.spectrogram.values)
#     draw_spec_set(specs, zoom=1, maxrows=10, colsize=25)
    
#     specs_flattened = flatten_spectrograms(specs)
    
#     # cuml_umap = cumlUMAP(min_dist=0.25)
#     # z = list(cuml_umap.fit_transform(specs_flattened))
    
#     fit = umap.UMAP(n_neighbors=30,
#         #min_dist=0.0,
#         n_components=2,
#         random_state=42,)
#     z = list(fit.fit_transform(specs_flattened))
#     indv_df["umap"] = z

#     indv_df["syllables_sequence_id"] = None
#     indv_df["syllables_sequence_pos"] = None
#     indv_df = indv_df.sort_values(by=["key", "start_time"])
#     for ki, key in enumerate(indv_df.key.unique()):
#         indv_df.loc[indv_df.key == key, "syllables_sequence_id"] = ki
#         indv_df.loc[indv_df.key == key, "syllables_sequence_pos"] = np.arange(
#             np.sum(indv_df.key == key)
#         )

#     draw_projection_plots(indv_df, label_column="indv")
    
#     scatter_spec(
#         np.vstack(z),
#         specs,
#         column_size=15,
#         #x_range = [-5.5,7],
#         #y_range = [-10,10],
#         pal_color="hls",
#         color_points=False,
#         enlarge_points=20,
#         figsize=(10, 10),
#         scatter_kwargs = {
#             #'labels': list(indv_df.labels.values),
#             'alpha':0.25,
#             's': 1,
#             'show_legend': False
#         },
#         matshow_kwargs = {
#             'cmap': plt.cm.Greys
#         },
#         line_kwargs = {
#             'lw':1,
#             'ls':"solid",
#             'alpha':0.25,
#         },
#         draw_lines=True
#     );
    
#     save_fig(FIGURE_DIR / 'bf' / ('bf_sakata_Bluebrown_'+indv), dpi=300, save_png=True, save_jpg=False)

#     plt.show()
