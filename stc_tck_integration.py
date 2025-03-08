import argparse
import json
import os
import random
import mne
import nibabel as nib
import numpy as np
import pywavefront
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix


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
    
    data = np.ones(len(row))  # Set all entries to 1 (indicating a connection)
    connectivity = csr_matrix((data, (row, col)), shape=(num_vertices, num_vertices))
    return connectivity

def interpolate_surface_values_timeseries(vertices, stc_indices, stc_values_timeseries):
    """Fills source estimate values for each vertex given a partial source space"""
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
    for hemi in ['rh', 'lh']:    
    for tissue in ['pial','white']:
        input_surface = f'{freesurfer_dir}/{subject}/surf/{hemi}.{tissue}'
        output_obj = f'{wavefront_output_dir}/{hemi}.{tissue}.obj'

        # Read in srf surface data
        vertices, faces, metadata = nib.freesurfer.read_geometry(input_surface, read_metadata=True)
        cras_offset = metadata.get('cras')

        # Write wavefront object using vertex coordinates and face associations
        with open(output_obj, 'w') as obj:
            for vertex in vertices:
                x,y,z = vertex[:3] + cras_offset
                obj.write(f"v {x} {y} {z}\n")
                
            for face in faces:
                idx1, idx2, idx3 = face + 1
                obj.write(f"f {idx1} {idx2} {idx3}\n")

def min_max_norm(a, min=None):
    """min-max normailization function with option to set minimum threshold"""
    if min:
        return(a - min)/(np.max(a) - min)
    else:
        return(a - np.min(a))/(np.max(a) - np.min(a))

def get_streamline_subset(tractography_file, tractography_output_dir, n=15000, force=False):
    """Generate subset of streamlines for visualization purposes"""
    # If no subsample has been saved or if force generation is applied, load in full tractography output and generate sample
    if (not os.path.exists(f'{tractography_output_dir}/streamline_subset.txt')) or force:
        raw_streamlines = nib.streamlines.load(tractography_file)
        streamlines = list(raw_streamlines.streamlines)
        streamline_subset = random.sample(streamlines, n)

        output_struct = [streamline.tolist() for streamline in streamline_subset]
        with open(f'{output_dir}/tractography/streamline_subset.txt', 'w') as file:
            json.dump(output_struct, file)

    else:
        # Read in preexisting streamline subset
        with open(f'{output_dir}/tractography/streamline_subset.txt', 'r') as file:
            streamline_subset_raw = json.load(file)
            streamline_subset = [np.array(streamline) for streamline in streamline_subset_raw]
    
    del raw_streamlines
    return(streamline_subset)

def link_streamline_activation(lh_scalars, rh_scalars, lh_vertex_associations, rh_vertex_associations, streamlines, threshold=0.6, soft_max=True:
    """Assign normalized and thresholded surface activation intensities to associated streamlines"""
    # min-max normalize source estimation values from  0-1
    
    if soft_max:
        # Uses "soft" percentile near-maximum to clip outliers in visualization
        stc_max = np.percentile(np.concatenate([rh_scalars.flatten(), lh_scalars.flatten()]), 99.9) 
    else:
        stc_max = np.max(np.concatenate([rh_scalars.flatten(), lh_scalars.flatten()])) 

    lh_norm = (lh_scalars - threshold*stc_max) / (stc_max - threshold*stc_max)
    rh_norm = (rh_scalars - threshold*stc_max) / (stc_max - threshold*stc_max)

    # Save normalized time series estimate for Blender vertex shading
    for hemi, scalars in [('lh', lh_norm), ('rh', rh_norm)]:
        np.save(f'{output_dir}/source_estimates/normalized/normalized_{hemi}.npy', scalars)
    
    # Soop through vertex associations and assign activation values to streamlines 
    active_streamlines = np.zeros((len(streamlines), lh_norm.shape[-1]))    
    for normalized_values, vertex_associations in [(rh_norm,rh_vertex_associations),(lh_norm,lh_vertex_associations)]:
        for t in range(normalized_values.shape[-1]):
            active_vertices = normalized_values[:,t] > 0
            for vert_index, streamline_indices in enumerate(vertex_associations):
                if active_vertices[vert_index]:
                    for streamline_index in streamline_indices:
                        # Streamline activation value is the maximum between all associated vertices
                        active_streamlines[streamline_index, t] = np.max([active_streamlines[streamline_index, t], normalized_values[vert_index, t]])
    
    np.save(f'{output_dir}/tractography/streamline_activation_timeseries.npy', active_streamlines)
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

def thresh_interpolate_smooth_stc(source_estimate_path, stim_onset=None, time_range=None):
    """Threshold, interpolate, and smooth source estimate activation values temporally and spatially"""
    source_estimate = mne.read_source_estimate(source_estimate_path)
        
    stc_data = np.concatenate([source_estimate.rh_data, source_estimate.lh_data])    
    
    baseline_data = stc_data[:,:(stim_onset-10)]
    baseline_means = np.mean(baseline_data, axis=1)
    baseline_data_adj = baseline_data - baseline_means[:, np.newaxis]
    baseline_stds = np.std(baseline_data_adj, axis=1)
    
    stc_data_adj = stc_data - baseline_means[:, np.newaxis]
    
    p_value = 0.05/len(baseline_means)
    z_score = norm.ppf(1 - p_value / 2) * 2
    thresholds = z_score * baseline_stds
    
    stc_data_thresh = np.where(stc_data_adj > (baseline_means[:, np.newaxis] + thresholds[:, np.newaxis]), stc_data_adj, 0)
    
    rh_data_thresh = stc_data_thresh[:source_estimate.rh_data.shape[0],:]
    lh_data_thresh = stc_data_thresh[source_estimate.rh_data.shape[0]:,:]
    
    estimate_dict = {'lh':(source_estimate.lh_vertno, lh_data_thresh),
                     'rh':(source_estimate.rh_vertno, rh_data_thresh)}
    
    if stim_onset and time_range:
        time_index_range = stim_onset + time_range
        n_frames = len(time_range)
    else:
        n_frames = len(estimate_dict['lh'][1].shape[1])
        time_index_range = np.arange(n_frames)
    
    output = []
    for hemi in ('rh','lh'):
        hemisphere_scalars = np.zeros((n_frames, surface_geometry[hemi]['n_vertices']))
        vertno, data = estimate_dict[hemi]
        interpolated_estimate = interpolate_surface_values_timeseries(surface_geometry[hemi]['vertices'], vertno, data[:, time_index_range])
        temporally_smoothed_estimate = gaussian_filter(interpolated_estimate, sigma=[0, 2])
        np.save(f'{output_dir}/source_estimates/smoothed/smoothed_{hemi}.npy', temporally_smoothed_estimate)
        output.append(temporally_smoothed_estimate)
    
    return(output)

    
def main():
    parser = argparse.ArgumentParser(description="Assign streamline activation intensity from source estimate")
    parser.add_argument("-t", "--tractography_path", type=str, help=".tck tractography file containing streamline data")
    parser.add_argument("-e", "--source_estimate_path", type=str, help=".stc source estimation path (MNE output, compatibility to improve with future releases)")
    parser.add_argument("-s", "--subject", type=str, help="FreeSurfer Reconall subject")
    parser.add_argument("-f", "--freesurfer_dir", type=str, help="Path to the directory containing FreeSurfer's 'recon-all' output.")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory path")
    args = parser.parse_args()

    subject=args.subject
    freesurfer_dir=args.freesurfer_dir
    tractography_path=args.tractography_path
    source_estimate_path=args.source_estimate_path
    output_dir=args.output_dir

    for subdir in ['wavefront', 'tractography', 'connectome', 'source_estimates/raw', 'source_estimates/smoothed', 'source_estimates/normalized', 'render']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    os.system(f'ln -s {tractography_path} {output_dir}/tractography/{os.path.basename(tractography_path)}')
    os.system(f'ln -s {source_estimate_path} {output_dir}/source_estimates/raw/{os.path.basename(source_estimate_path)}')

    srf_to_wavefront(freesurfer_dir, subject, os.path.join(output_dir, 'wavefront'))

    if sample:
        streamlines = get_streamline_subset(tractography_path, os.path.join(output_dir, 'tractography'))
    else:
        streamlines = list(nib.streamlines.load(tractography_path).streamlines)

    streamline_endpoints = np.zeros((len(streamlines),2,3))
    for i, streamline in enumerate(streamlines):
        streamline_endpoints[i] = [streamline[0], streamline[-1]]

    if interpolate_stc:
        rh_scalars, lh_scalars = interpolate_and_smooth_stc(source_estimate_path)
    else:
        raw_stc = mne.read_source_estimate(source_estimate_path)
        rh_scalars, lh_scalars = raw_stc.rh_data, raw_stc.lh_data

    lh_surface_verts = pywavefront.Wavefront(f'{output_dir}/wavefront/lh.pial.obj').vertices
    rh_surface_verts = pywavefront.Wavefront(f'{output_dir}/wavefront/rh.pial.obj').vertices

    lh_vertex_associations = associate_vertices_to_streamlines(lh_surface_verts, streamline_endpoints, 3)
    rh_vertex_associations = associate_vertices_to_streamlines(rh_surface_verts, streamline_endpoints, 3)

    link_streamline_activation(lh_scalars, rh_scalars, lh_vertex_associations, rh_vertex_associations, streamlines, threshold=0.6)

    if __name__ == "__main__":
    main()