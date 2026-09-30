# STREAM-4D
### Surface to Tractography time-Resolved EEG Activation Mapping in 4 Dimensions 

![alt text](https://github.com/ComaRecoveryLab/STREAM-4D/blob/main/resources/STREAM-4D_premotor.gif "STREAM-4D Left Premotor Stimulation")

Version 2.0.0

To enhance mechanistic understanding of effective connectivity in the human brain, we created a tool that links high-temporal resolution transcranial magnetic stimulation electroencephalography (TMS-EEG) with high-spatial resolution diffusion MRI. This tool, Surface to Tractography Real-time EEG Activation Mapping in 4 Dimensions (STREAM-4D), integrates electrophysiologic source estimation models from TMS-evoked potentials (TEPs) with structural connectivity models from diffusion MRI tractography. Our proof-of-principle application of this pipeline is described in ["Visualizing Effective Connectivity in the Human Brain"](https://doi.org/10.1016/j.brs.2025.06.015).

The pipeline is built for use with `.stc` source estimation files (the output of [MNE Python](https://mne.tools/stable/index.html) source localization), `.tck` tractograms (the output of [MRTrix](https://www.mrtrix.org) tckgen), and [FreeSurfer](https://surfer.nmr.mgh.harvard.edu) surface outputs. STREAM-4D requires a Blender version > 4.2.

## Pipeline

The pipeline consists of two commands: first generating the associations between streamlines and surface activation, and then loading, keyframing, and rendering using [Blender](https://www.blender.org). For `render.py` to execute successfully, Blender must be accessible from the command line.

```
python stream4d.py [-t TRACTOGRAPHY_PATH] [-e SOURCE_ESTIMATE_PATH] [-w SIFT_WEIGHT_PATH] [-f FREESURFER_DIR] [-s SUBJECT] [-l LABEL] [-o OUTPUT_DIR] [--dist_threshold THRESHOLD_MM] [--no-connectome]

python render.py [-s SUBJECT] [-l LABEL] [-o OUTPUT_DIR]
```

### Geometry-only mode

The `-e` source estimate argument is optional. When omitted, `stream4d.py` runs only the geometry stage: surface conversion, tractography sampling, and construction of the vertex-streamline association matrix. This is useful for pre-computing structural relationships before source estimates are available, or for reuse across multiple sessions of the same subject.

### Vertex-streamline association matrix

`stream4d.py` saves a sparse boolean vertex-streamline association matrix to `OUTPUT_DIR/connectome/vertex_streamline_associations.npz`. The matrix has shape `(n_streamlines, n_vertices)` and encodes, for each streamline in the full tractography, the cortical surface vertices whose locations on the white-matter surface fall within the configured distance threshold of either streamline endpoint. On subsequent runs this file is loaded directly, skipping the association computation. The matrix can be used independently for structural connectivity analyses.

Streamline endpoints are associated against the white-matter surface (`?h.white`) rather than the pial surface, since standard GMWM-boundary-seeded tractography terminates at the white-matter surface. A tighter distance threshold is therefore appropriate (a default of 1.5–2 mm is recommended for GMWM-seeded tractography); the `--dist_threshold` flag accepts a value in millimeters. When omitted, the threshold defaults to the subject's average cortical thickness, which is appropriate when streamlines may extend into the cortical ribbon.

### Blender scene assembly

The Blender scene (`blender/scene_template.py`) is fully procedural. All required materials (`streamlines`, `wireframe`, `tms_map`) and geometry node groups (`streamline_geometry`, `white_matter`) are constructed in Python from modules under `blender/blender_nodes/`. No `.blend` template is required.

## Repository Layout

```
stream4d.py                 Pipeline entry point
render.py                   Blender orchestration entry point
blender/
  scene_template.py         Scene assembly (procedural materials + node groups)
  keyframe_tractography.py  Per-streamline activation keyframing
  source_estimation_animation.py  Per-frame vertex intensity update
  blender_nodes/            Python definitions of all materials and node groups
    materials/              wireframe, tms_map, streamlines
    geometry/               white_matter, streamline_geometry
resources/                  FreeSurfer label LUTs used during connectome construction
```

## Repo Contributors

- Alexander Atalay ([asatalay@stanford.edu](mailto:asatalay@stanford.edu))

## Acknowledgments

This work was supported by the Chen Institute Mass General Research Scholar Award, Mass General Neuroscience Transformative Scholar Award, Tiny Blue Dot Foundation, NIH National Institute of Neurological Disorders and Stroke (R01NS138257), NIH Director's Office (DP2HD101400), and MIT/MGH Brain Arousal State Control Innovation Center (BASCIC) project.
