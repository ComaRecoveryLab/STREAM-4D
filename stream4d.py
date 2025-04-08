import argparse
import json
import os
import random
import shutil

import nibabel as nib
import mne
import numpy as np
import pywavefront
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.stats import norm
from scipy.sparse.csgraph import shortest_path

def associate_vertices_to_streamlines(vertices, streamline_endpoints, distance_threshold):
    """Associate surface vertices with streamlines that have an endpoint within the proximity threshold."""
    # Extract start and end points of streamlines
    streamline_starts = streamline_endpoints[:,0]
    streamline_ends = streamline_endpoints[:,1]

    # Keep track of streamline endpoint indices
    all_streamline_points = np.vstack((streamline_starts, streamline_ends))
    streamline_indices = np.arange(len(streamline_endpoints))
    streamline_indices = np.concatenate([streamline_indices, streamline_indices])

    # Construct a k-d tree for the start and end points
    tree = cKDTree(all_streamline_points)

    # Instantiate list of lists to store associations
    vertex_to_streamlines = [[] for _ in range(len(vertices))]

    # Query the k-d tree for each vertex to find close streamline start/end points
    for i, vertex in enumerate(vertices):
        # Find all streamline points within the distance threshold from the vertex
        nearby_indices = tree.query_ball_point(vertex, distance_threshold)
        # Ensure streamlines are unique
        unique_streamline_indices = set(streamline_indices[idx] for idx in nearby_indices)
        vertex_to_streamlines[i] = list(unique_streamline_indices)

    return vertex_to_streamlines

def get_sparse_connectivity(faces, num_vertices):
    """Constructs a sparse vertex connectivity matrix from faces to use for source activation smoothing"""
    row, col = [], []
    for face in faces:
        # Ensure the face has at least 2 vertices (skip otherwise)
        if len(face) < 2:
            continue
        
        # For each vertex in the face, connect it to the other vertices
        for i, v in enumerate(face):
            if v < num_vertices:  # Ensure the vertex index is within bounds
                linked_neighbors = [neighbor for neighbor in face if neighbor != v and neighbor < num_vertices]
                for neighbor in linked_neighbors:
                    row.append(v)
                    col.append(neighbor)
    
    # Set all entries to 1 (indicating a connection)
    data = np.ones(len(row)) 
    connectivity = csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    return connectivity

def interpolate_surface_values_timeseries(vertices, stc_indices, stc_values_timeseries):
    """Interpolates missing values in full_scalar_values for each timepoint in a timeseries using nearest neighbors."""
    n_timepoints = stc_values_timeseries.shape[1]
    full_scalar_values = np.zeros((vertices.shape[0], n_timepoints))
    
    # Set known scalar values for all timepoints
    full_scalar_values[stc_indices, :] = stc_values_timeseries
    
    missing_indices = np.where(full_scalar_values[:, 0] == 0)[0]  # assumes missing values are 0 in all timepoints
    
    # Use KDTree to find the nearest known vertex for each missing vertex
    tree = cKDTree(vertices[stc_indices])
    _, nearest_known_indices = tree.query(vertices[missing_indices])
    
    # Assign the scalar values of the nearest known vertex to each missing vertex for each timepoint
    full_scalar_values[missing_indices, :] = full_scalar_values[stc_indices][nearest_known_indices, :]
    
    return full_scalar_values

def smooth_values_sparse_timeseries(connectivity, scalar_values_timeseries, num_passes=10):
    """Smooths timeseries of scalar values using a sparse connectivity matrix, ignoring zero vertices in averaging."""
    smoothed_values = scalar_values_timeseries.copy()
    for _ in range(num_passes):
        neighbor_sums = connectivity.dot(smoothed_values)
        
        non_zero_neighbors = connectivity.dot((smoothed_values != 0).astype(float))
        
        # Avoid division by zero by only averaging where there are non-zero neighbors
        mask = non_zero_neighbors > 0
        smoothed_values[mask] = neighbor_sums[mask] / non_zero_neighbors[mask]
    
    return smoothed_values

def srf_to_wavefront(freesurfer_dir, subject, wavefront_output_dir):
    """Convert FreeSurfer srf surface output to Blender compatible wavefront object"""
    os.makedirs(wavefront_output_dir, exist_ok=True)
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    
    # for tissue in ['pial', 'white']:
    for tissue in ['pial']:
        all_sphere_vertices = []
        all_vertices = []
        all_faces = []
    
        for hemi in ['lh', 'rh']:
            input_surface = f'{freesurfer_dir}/{subject}/surf/{hemi}.{tissue}'
            adj = {'lh':-60, 'rh':60}
    
            # Read surface geometry and CRAS metadata
            vertices, faces, metadata = nib.freesurfer.read_geometry(input_surface, read_metadata=True)
            cras_offset = metadata.get('cras', np.zeros(3))
            vertices += cras_offset

            sphere_vertices, _ = nib.freesurfer.read_geometry(f'{freesurfer_dir}/{subject}/surf/{hemi}.sphere')
            sphere_vertices[:,0] += adj[hemi]
            all_sphere_vertices.append(sphere_vertices)
    
            # Adjust face indices by current offset
            faces += vertex_offset
            all_vertices.append(vertices)
            all_faces.append(faces)
            
            if hemi=='lh':
                vertex_offset += vertices.shape[0]
    
        # Combine both hemispheres
        all_vertices = np.vstack(all_vertices)
        all_faces = np.vstack(all_faces)
        all_sphere_vertices = np.vstack(all_sphere_vertices)
        
        # Write to .obj file
        output_obj = os.path.join(wavefront_output_dir, f'{tissue}.obj')
        with open(output_obj, 'w') as obj:
            for v in all_vertices:
                obj.write(f'v {v[0]} {v[1]} {v[2]}\n')
            for f in all_faces:
                obj.write(f'f {f[0]+1} {f[1]+1} {f[2]+1}\n')

        # Return geometry dictionary for further use
        return({'vertices':all_vertices,
               'sphere_vertices':all_sphere_vertices,
               'faces':all_faces,
               'n_vertices':len(all_vertices),
               'vertex_offset':vertex_offset})

def min_max_norm(a, min=None):
    """min-max normailization function with option to set minimum threshold"""
    if min:
        return(a - min)/(np.max(a) - min)
    else:
        return(a - np.min(a))/(np.max(a) - np.min(a))

def get_streamline_subset(tractography_file, output_dir, n=15000, force=False):
    """Generate subset of streamlines for visualization purposes"""
    # If no subsample has been saved or if force generation is applied, load in full tractography output and generate sample
    if (not os.path.exists(f'{output_dir}/streamline_subset.txt')) or force:
        raw_streamlines = nib.streamlines.load(tractography_file)
        streamlines = list(raw_streamlines.streamlines)
        streamline_subset = random.sample(streamlines, n)

        output_struct = [streamline.tolist() for streamline in streamline_subset]
        with open(f'{output_dir}/streamline_subset.txt', 'w') as file:
            json.dump(output_struct, file)

        del raw_streamlines

    else:
        # Read in preexisting streamline subset
        with open(f'{output_dir}/streamline_subset.txt', 'r') as file:
            streamline_subset_raw = json.load(file)
            streamline_subset = [np.array(streamline) for streamline in streamline_subset_raw]
    
    return(streamline_subset)

def link_streamline_activation(scalars, vertex_associations, streamlines, output_dir, label = ''):
    """Assign normalized and thresholded surface activation intensities to associated streamlines"""
    # min-max normalize source estimation values from  0-1
    if label:
        label = label + '_'

    soft_max = np.percentile(scalars, 99.95)
    thresh = np.percentile(scalars[scalars > 0], 90)
    normalized_scalars = (scalars - thresh) / (soft_max - thresh)
    normalized_scalars = np.where(normalized_scalars > 0, normalized_scalars, 0)

    # Save normalized time series estimate for Blender vertex shading
    np.save(f'{output_dir}/source_estimates/normalized/{label}normalized.npy', normalized_scalars)
    
    # Loop through vertex associations and assign activation values to streamlines 
    active_streamlines = np.zeros((len(streamlines), normalized_scalars.shape[-1]))    
    for t in range(normalized_scalars.shape[-1]):
        active_vertices = normalized_scalars[:,t] > 0
        for vert_index, streamline_indices in enumerate(vertex_associations):
            if active_vertices[vert_index]:
                for streamline_index in streamline_indices:
                    # Streamline activation value is the maximum between all associated vertices
                    active_streamlines[streamline_index, t] = np.max([active_streamlines[streamline_index, t], normalized_scalars[vert_index, t]])
    
    np.save(f'{output_dir}/tractography/{label}streamline_activation_timeseries.npy', active_streamlines)
    return(active_streamlines)

def weighted_connectome_analysis(streamlines, active_streamlines, sift_weight_path, parcels_path):
    with open(sift_weight_path,'r') as weight_file:
        anatomical_weights = np.array(weight_file.readlines()[-1].split(' ')).astype(np.float64)

    streamline_activations_integrated = np.sum(active_streamlines, axis=1)
    streamline_activations_integrated_anat_weighted = streamline_activations_integrated*anatomical_weights
    
    streamline_mask = [sl for (sl, val) in zip(streamlines, streamline_activations_integrated) if val]
    streamline_scalars_mask = min_max_norm(streamline_activations_integrated_anat_weighted[streamline_activations_integrated_anat_weighted>0])*10
    
    np.savetxt(f'{output_dir}/connectome/activation_integration_weights.txt',streamline_scalars_mask,delimiter=' ',newline=' ')

    tractogram = nib.streamlines.Tractogram(streamline_mask)
    nib.streamlines.save(tractogram, f'{output_dir}/connectome/active_streamlines.tck')

    os.system(f'tck2connectome -symmetric -zero_diagonal \
    -scale_invnodevol -tck_weights_in {output_dir}/connectome/activation_integration_weights.txt \
    {output_dir}/connectome/active_streamlines.tck {parcels_path} \
    {output_dir}/connectome/parcels.csv \
    -out_assignment {output_dir}/connectome/assignments_parcels.csv -force')

def threshold_stc(source_estimate_path, surface_geometry, output_dir, stim_onset, time_range):
    """Threshold, interpolate, and smooth source estimate activation values temporally and spatially"""
    source_estimate = mne.read_source_estimate(source_estimate_path)
    stc_data = source_estimate.data

    lh_vertno = source_estimate.lh_vertno
    rh_vertno = source_estimate.rh_vertno + surface_geometry['vertex_offset']
    vertno = np.hstack([lh_vertno, rh_vertno])
    
    baseline_data = stc_data[:,:(stim_onset-10)]
    baseline_means = np.mean(baseline_data, axis=1)
    baseline_data_adj = baseline_data - baseline_means[:, np.newaxis]
    baseline_stds = np.std(baseline_data_adj, axis=1)
    
    stc_data_adj = stc_data - baseline_means[:, np.newaxis]
    
    p_value = 0.05/len(baseline_means)
    z_score = norm.ppf(1 - p_value / 2) * 2
    thresholds = z_score * baseline_stds
    
    stc_data_thresh = np.where(stc_data_adj > (baseline_means[:, np.newaxis] + thresholds[:, np.newaxis]), stc_data_adj, 0)
    
    time_index_range = stim_onset + time_range
    n_frames = len(time_range)

    hemisphere_scalars = np.zeros((n_frames, surface_geometry['n_vertices']))
    interpolated_estimate = interpolate_surface_values_timeseries(surface_geometry['sphere_vertices'], vertno, stc_data_thresh[:, time_index_range])
    temporally_smoothed_estimate = gaussian_filter(interpolated_estimate, sigma=[0, 2])
    smoothed_estimate = smooth_values_sparse_timeseries(surface_geometry['connectivity'], temporally_smoothed_estimate)

    np.save(f'{output_dir}/source_estimates/smoothed/smoothed.npy', smoothed_estimate)
    return(smoothed_estimate)

def setup_blender(subject, output_dir):
    """Set up Blender for rendering"""
    cwd = os.getcwd()
    os.system(f"blender --background --python {cwd}/scene_template.py -- --subject {subject} --output {output_dir}")

def render_output(subject, render_output_dir):
    blender_file = f'{subject}/{subject}_stream3d.blend'
    render_output_dir = f'./render/frame_####.png'

    os.system(f'''\
    blender -b {blender_file} \
    --python-expr "import bpy; bpy.data.texts['source_estimation_import'].as_module().run()" \
    --engine BLENDER_EEVEE \
    --render-output {output_dir}/render/frame_####.png \
    --render-anim\
    ''')

def run_stream4d(freesurfer_dir, subject, tractography_path, source_estimate_path, output_dir):
    print('------------------------------')
    print('          STREAM-4D           ')
    print('------------------------------')

    print('setting up directories')
    for subdir in ['wavefront', 'tractography', 'connectome', 'source_estimates/raw', 'source_estimates/smoothed', 'source_estimates/normalized', 'render']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    os.system(f'ln -s {tractography_path} {output_dir}/tractography/{os.path.basename(tractography_path)}')
    os.system(f'ln -s {source_estimate_path} {output_dir}/source_estimates/raw/{os.path.basename(source_estimate_path)}')

    print('\nLoading FreeSurfer Surface Geometry\n')
    surface_geometry = srf_to_wavefront(freesurfer_dir, subject, os.path.join(output_dir, 'wavefront'))
    surface_geometry['connectivity'] = get_sparse_connectivity(surface_geometry['faces'], surface_geometry['n_vertices'])

    print('Loading Tractography Data\n')
    streamlines = get_streamline_subset(tractography_path, os.path.join(output_dir, 'tractography'))
    streamline_endpoints = np.zeros((len(streamlines),2,3))

    print('Extracting Streamline Endpoints\n')
    for i, streamline in enumerate(streamlines):
        streamline_endpoints[i] = [streamline[0], streamline[-1]]
        
    print('Thresholding Source Estimate\n')
    scalars = threshold_stc(source_estimate_path=source_estimate_path, surface_geometry=surface_geometry, output_dir=output_dir, stim_onset=500, time_range=np.arange(-25, 75))

    print('Associating Streamlines to Vertices\n')
    vertex_associations = associate_vertices_to_streamlines(surface_geometry['vertices'], streamline_endpoints, 3)

    print('Linking Activation Timeseries\n')
    link_streamline_activation(scalars, vertex_associations, streamlines, output_dir)
    print('Associations Complete!')

    print("\nSetting up Blender for rendering\n")
    setup_blender(subject, output_dir)
    print("Blender setup complete\n")

def main():
    parser = argparse.ArgumentParser(description="Assign streamline activation intensity from source estimate")
    parser.add_argument("-t", "--tractography_path", type=str, help=".tck tractography file containing streamline data")
    parser.add_argument("-e", "--source_estimate_path", type=str, help=".stc source estimation path (MNE output, compatibility to improve with future releases)")
    parser.add_argument("-s", "--subject", type=str, help="FreeSurfer Reconall subject")
    parser.add_argument("-f", "--freesurfer_dir", type=str, help="Path to the directory containing FreeSurfer's 'recon-all' output.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory path")
    args = parser.parse_args()

    run_stream4d(args.freesurfer_dir, args.subject, args.tractography_path, args.source_estimate_path, args.output_dir)

if __name__ == "__main__":
    main()