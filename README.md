# AVGN_Avishek
AVGN code by Tim Modified for Sakata Lab data

Code execution tested on Python 3.7.5 in spyder IDE on Windows

- Avoids issues with SIGALRM attribute in Windows
- Parallel processing avoided wherever necessary as it leads to error
- Import of tqdm module and application modified to make it properly run in IDE
- Resource Allocation warning as a result of not closing of json files addressed
- Dependancy on vocalization-segmentation package for dynamic segmentation removed

A sample dataset consisting of 3 birds with multiple renditions 
* Tutor_Bl5w5 (31 files)
* Br81Bl41 (28 files)
* Br82Bl42 (19 files)

Results after execution on this dataset are shown 

**Audio File**

![Sample_audio_file](https://github.com/AvisP/AVGN_Avishek/blob/master/figures/BlackBrown/Sample_audio.png "Sample Audio File")

**Dynamic Segmentation**

![Sample_segmented_Spectrogram](https://github.com/AvisP/AVGN_Avishek/blob/master/figures/BlackBrown/Sample_segmentation.png "Sample Dynamic Segmentation")

**Summary of segmentation**
<img src="https://github.com/AvisP/AVGN_Avishek/blob/master/figures/BlackBrown/Histograms.png" width="140%">

**Spectrograms**

<p align="center"><img src="https://github.com/AvisP/AVGN_Avishek/blob/master/gifs/Spec_scatter2.gif" width="80%" title="Spectrogram Scatter" /></p>

**UMAP**
<p align="center"><img src="https://github.com/AvisP/AVGN_Avishek/blob/master/gifs/UMAP.gif" width="80%" title="UMAP"/></p>

**Transition Diagram**
<p align="center"><img src="https://github.com/AvisP/AVGN_Avishek/blob/master/gifs/TransitionDiagram.gif" width="80%" title="UMAP"/></p>

**Network Diagram**
<p align="center"><img src="https://github.com/AvisP/AVGN_Avishek/blob/master/gifs/NetworkDiagram.gif" width="80%" title="UMAP"/></p>
