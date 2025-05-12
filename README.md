# STREAM-4D
### Surface to Tractography Real-time EEG Activation Mapping in 4 Dimensions 

![alt text](https://github.com/ComaRecoveryLab/STREAM-4D/blob/main/resources/STREAM-4D_banner.png "STREAM-4D Left Premotor Stimulation")

Verson 1.0.0

To enhance mechanistic understanding of effective connectivity in the human brain, we created a tool that links high-temporal resolution transcranial magnetic stimulation electroencephalography (TMS-EEG) with high-spatial resolution diffusion MRI. This tool, Surface to Tractography Real-time EEG Activation Mapping in 4 Dimensions (STREAM-4D), integrates electrophysiologic source estimation models from TMS-evoked potentials (TEPs) with structural connectivity models from diffusion MRI tractography. Our proof-of-principle application of this pipeline is described in ["Visualizing Effective Connectivity in the Human Brain"](https://doi.org/10.1101/2025.03.06.641642).

The current pipeline is built for use with .src source estimation files (the output of ["MNE Python"](https://mne.tools/stable/index.html) source localization), .tck tractograms (the output of ["MRTrix"](https://www.mrtrix.org) tckgen), and ["Freesurfer"](https://surfer.nmr.mgh.harvard.edu) surface outputs.

The pipeline consists of two commands: First generating the associations between streamlines and surface activation, and then loading, animating, and rendering using ["Blender"](https://www.blender.org).

```
python stream4d.py [-t TRACTOGRAPHY_PATH] [-e SOURCE_ESTIMATE_PATH] [-w SIFT_WEIGHT_PATH] [-f FREESURFER_DIR] [-s SUBJECT] [-l LABEL] [-o OUTPUT_DIR] [--no-connectome]

python render.py [-s SUBJECT] [-l LABEL] [-o OUTPUT_DIR]
```
