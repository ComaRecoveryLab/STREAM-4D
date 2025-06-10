import argparse
import json
import os
import subprocess
import time
import random

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

import mne
import mne_connectivity
import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.sparse import csr_matrix
from scipy.spatial import KDTree
from scipy.stats import norm

# Utility
def min_max_norm(a, min=None):
    """
    Normalizes an array to [0, 1] using min-max normalization.

    Parameters:
        a (ndarray): Array to normalize.
        min (float, optional): Minimum value to use instead of np.min(a).

    Returns:
        ndarray: Normalized array.
    """
    if min is not None:
        return(a - min)/(np.max(a) - min)
    else:
        return(a - np.min(a))/(np.max(a) - np.min(a))

def get_sparse_connectivity(faces, num_vertices):
    """
    Constructs a sparse vertex adjacency matrix based on triangular mesh faces.

    Parameters:
        faces (ndarray): M x 3 array of triangle vertex indices.
        num_vertices (int): Total number of vertices.

    Returns:
        csr_matrix: Sparse connectivity matrix (num_vertices x num_vertices).
    """
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

# Geometry
def get_streamline_endpoints(streamlines):
    streamline_endpoints = np.zeros((len(streamlines),2,3))
    for i, streamline in enumerate(streamlines):
        streamline_endpoints[i] = [streamline[0], streamline[-1]]
    return streamline_endpoints

def srf_to_wavefront(freesurfer_dir, subject, wavefront_output_dir):
    """
    Converts FreeSurfer surface geometry to OBJ format for Blender visualization.

    Parameters:
        freesurfer_dir (str): Path to FreeSurfer recon-all directory.
        subject (str): Subject identifier.
        wavefront_output_dir (str): Directory to save .obj output.

    Returns:
        dict: Contains 'vertices', 'sphere_vertices', 'faces', 'n_vertices', and 'vertex_offset'.
    """
    os.makedirs(wavefront_output_dir, exist_ok=True)
    output_dict = {}
    
    for tissue in ['white', 'pial']:
        all_sphere_vertices = []
        all_vertices = []
        all_faces = []
        vertex_offset = 0
    
        for hemi in ['lh', 'rh']:
            input_surface = os.path.join(freesurfer_dir, subject, 'surf', f'{hemi}.{tissue}')
            adj = {'lh':-60, 'rh':60}
    
            # Read surface geometry and CRAS metadata
            vertices, faces, metadata = nib.freesurfer.read_geometry(input_surface, read_metadata=True)
            cras_offset = metadata.get('cras', np.zeros(3))
            vertices += cras_offset

            sphere_vertices, _ = nib.freesurfer.read_geometry(os.path.join(freesurfer_dir, subject, 'surf', f'{hemi}.sphere'))
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
        if tissue == 'pial':
            output_dict = {'vertices':all_vertices,
                           'sphere_vertices':all_sphere_vertices,
                           'faces':all_faces,
                           'n_vertices':len(all_vertices),
                           'vertex_offset':vertex_offset}

    return(output_dict)

def create_connectome_parcels(freesurfer_dir, subject, output_dir):
    """
    Converts a subject's FreeSurfer aparc+aseg.mgz parcellation to a .mif file using MRtrix's labelconvert.

    Parameters:
        freesurfer_dir (str): Path to the FreeSurfer recon-all directory.
        subject (str): Freesurfer sbject identifier.
        output_dir (str): Directory to save the converted .mif file.

    Returns:
        str: Path to the generated .mif parcellation file.
    """
    os.system(
        f'labelconvert {freesurfer_dir}/{subject}/mri/aparc+aseg.mgz '
        f'{os.path.dirname(os.path.realpath(__file__))}/resources/FreeSurferColorLUT.txt {os.path.dirname(os.path.realpath(__file__))}/resources/fs_default.txt '
        f'{output_dir}/{subject}.mif -force'
    )
    return f'{output_dir}/{subject}.mif'

# Streamline Management
def get_streamline_subset(tractography_file, output_dir, n=15000, force=False):
    """
    Loads a random subset of streamlines for visualization or analysis.

    Parameters:
        tractography_file (str): Path to .tck streamline file.
        output_dir (str): Directory to save or load subset from.
        n (int): Number of streamlines to sample.
        force (bool): Whether to overwrite existing subset.

    Returns:
        list of ndarray: Subset of streamlines.
    """
    # If no subsample has been saved or if force generation is applied, load in full tractography output and generate sample
    if not os.path.exists(os.path.join(output_dir, 'streamline_subset.txt')) or force:
        raw_streamlines = nib.streamlines.load(tractography_file)
        streamlines = list(raw_streamlines.streamlines)
        streamline_subset = random.sample(streamlines, n)

        output_struct = [streamline.tolist() for streamline in streamline_subset]
        with open(os.path.join(output_dir, 'streamline_subset.txt'), 'w') as file:
            json.dump(output_struct, file)

        del raw_streamlines

    else:
        # Read in preexisting streamline subset
        with open(os.path.join(output_dir, 'streamline_subset.txt'), 'r') as file:
            streamline_subset_raw = json.load(file)
            streamline_subset = [np.array(streamline) for streamline in streamline_subset_raw]
    
    return(streamline_subset)

def associate_vertices_to_streamlines(vertices, streamline_endpoints, distance_threshold):
    """
    Associates surface vertices with streamlines whose endpoints are within a given distance threshold.

    Parameters:
        vertices (ndarray): N x 3 array of vertex coordinates.
        streamline_endpoints (ndarray): M x 2 x 3 array of streamline start and end points.
        distance_threshold (float): Distance threshold for associating streamlines to vertices.

    Returns:
        list of lists: A list where each entry contains indices of vertices associated with that streamline.
    """
    n_streamlines = streamline_endpoints.shape[0]
    # Reshape endpoint coordinates to: (2 * N_streamlines, 3)
    endpoint_coords = streamline_endpoints.reshape(-1, 3)
    # Query all endpoint coordinates at once
    tree = KDTree(vertices)
    all_neighbors = tree.query_ball_point(endpoint_coords, distance_threshold)
    # Parse and combine results from each streamline's start and endpoint
    vertex_to_streamlines = [
        list(set(all_neighbors[2*i] + all_neighbors[2*i + 1]))  # Remove duplicates
        for i in range(n_streamlines)
    ]

    return vertex_to_streamlines

# Source Estimation Processing
def interpolate_surface_values_timeseries(vertices, stc_indices, stc_values_timeseries):
    """
    Interpolates missing scalar values at vertices over time using nearest neighbor interpolation.

    Parameters:
        vertices (ndarray): N x 3 array of surface vertex coordinates.
        stc_indices (ndarray): Indices of vertices with known values.
        stc_values_timeseries (ndarray): Known values of shape (len(stc_indices), timepoints).

    Returns:
        ndarray: Interpolated values for all vertices, shape (N, timepoints).
    """
    n_timepoints = stc_values_timeseries.shape[1]
    full_scalar_values = np.zeros((vertices.shape[0], n_timepoints))
    
    # Set known scalar values for all timepoints
    full_scalar_values[stc_indices, :] = stc_values_timeseries
    
    missing_indices = np.where(full_scalar_values[:, 0] == 0)[0]  # assumes missing values are 0 in all timepoints
    
    # Use KDTree to find the nearest known vertex for each missing vertex
    tree = KDTree(vertices[stc_indices])
    _, nearest_known_indices = tree.query(vertices[missing_indices])
    
    # Assign the scalar values of the nearest known vertex to each missing vertex for each timepoint
    full_scalar_values[missing_indices, :] = full_scalar_values[stc_indices][nearest_known_indices, :]
    
    return full_scalar_values

def smooth_values_sparse_timeseries(connectivity, scalar_values_timeseries, num_passes=10):
    """
    Smooths scalar values on a surface over time using neighbor averaging via a sparse connectivity matrix.

    Parameters:
        connectivity (csr_matrix): Sparse vertex adjacency matrix.
        scalar_values_timeseries (ndarray): Scalar values at each vertex and timepoint (N x T).
        num_passes (int): Number of laplacian smoothing passes to apply.

    Returns:
        ndarray: Smoothed scalar values (N x T).
    """
    smoothed_values = scalar_values_timeseries.copy()
    for _ in range(num_passes):
        neighbor_sums = connectivity.dot(smoothed_values)
        
        non_zero_neighbors = connectivity.dot((smoothed_values != 0).astype(float))
        
        # Avoid division by zero by only averaging where there are non-zero neighbors
        mask = non_zero_neighbors > 0
        smoothed_values[mask] = neighbor_sums[mask] / non_zero_neighbors[mask]
    
    return smoothed_values

def threshold_stc(source_estimate_path, surface_geometry, output_dir, stim_onset, time_range, label=''):
    """
    Thresholds source estimates using baseline z-score correction, interpolates missing values,
    applies temporal and spatial smoothing, and returns the final smoothed timeseries.

    Parameters:
        source_estimate_path (str): Path to .stc file.
        surface_geometry (dict): Output from `srf_to_wavefront`, includes vertices, faces, etc.
        output_dir (str): Output directory.
        stim_onset (int): Index of stimulus onset timepoint.
        time_range (ndarray): Time indices relative to stim_onset to include in final estimate.

    Returns:
        ndarray: Smoothed surface scalar values (n_timepoints x n_vertices).
    """
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

    np.save(os.path.join(output_dir, 'source_estimates', 'smoothed', f'{label}smoothed.npy'), smoothed_estimate)
    return(smoothed_estimate)

# Streamline Activation
def link_streamline_activation(scalars, vertex_associations, streamlines, output_dir, vis_thresh=True, label='', save=True):
    """
    Propagates scalar activation values from surface vertices to associated streamlines.

    Parameters:
        scalars (ndarray): Vertex activation values (N_vertices x timepoints).
        vertex_associations (list of lists): Streamline indices associated with each vertex.
        streamlines (list of ndarray): List of streamlines.
        output_dir (str): Output directory to save results.
        label (str): Optional prefix label for saved output.

    Returns:
        ndarray: Streamline activation timeseries (N_streamlines x timepoints).
    """
    # min-max normalize source estimation values from  0-1
    soft_max = np.percentile(scalars, 99.95)
    if vis_thresh:
        thresh = np.percentile(scalars[scalars > 0], 90)
    else:
        thresh = np.min(scalars[scalars > 0])

    normalized_scalars = (scalars - thresh) / (soft_max - thresh)
    normalized_scalars = np.where(normalized_scalars > 0, normalized_scalars, 0)
    
    n_streamlines, n_timepoints = len(streamlines), scalars.shape[1]
    active_streamlines = np.zeros((n_streamlines, n_timepoints))

    # Iterate through each streamline and assign the maximum activation value from associated vertices
    for str_idx, vtx_indices in enumerate(vertex_associations):
        if not vtx_indices:
            continue

        streamline_activation = np.max(normalized_scalars[vtx_indices, :], axis=0)
        active_streamlines[str_idx] = streamline_activation

    if save:
        # Save activation timeseries for Blender shading
        np.save(os.path.join(output_dir, 'source_estimates', 'normalized', f'{label}normalized.npy'), normalized_scalars)
        np.save(os.path.join(output_dir, 'tractography', f'{label}streamline_activation_timeseries.npy'), active_streamlines)

    return(active_streamlines)

# Connectome Analysis
def weighted_connectome_analysis(scalars, vertex_associations, streamlines, sift_weight_path, parcels_path, output_dir, label=''):
    """
    Creates a weighted structural connectome based on streamline activation and anatomical weights.

    Parameters:
        scalars (ndarray): Vertex-level scalar activations over time (n_vertices x n_timepoints).
        vertex_associations (list of lists): Mapping from each vertex to associated streamline indices.
        streamlines (list): List of streamlines (each streamline is an array of 3D points).
        sift_weight_path (str): Path to text file with SIFT streamline weights.
        parcels_path (str): Path to parcellation file (e.g., an atlas) for connectome construction.
        label (str): Optional prefix for output filenames.

    Returns:
        None
    """

    # Load SIFT weights if provided, otherwise default to equal weighting
    if sift_weight_path:
        with open(sift_weight_path, 'r') as weight_file:
            anatomical_weights = np.array(weight_file.readlines()[-1].split(' ')).astype(np.float64)
    else:
        anatomical_weights = np.ones(len(streamlines))

    # Compute streamline-level activation over time from surface scalars
    streamline_activations = link_streamline_activation(
        scalars, vertex_associations, streamlines,
        output_dir=os.path.join(output_dir, 'connectome'),
        save=False, vis_thresh=None, label=label
    )

    # Integrate activation across time for each streamline
    streamline_activations_integrated = np.sum(streamline_activations, axis=1)

    # Multiply by anatomical weights and apply nonlinear scaling
    streamline_activations_integrated_anat_weighted = streamline_activations_integrated**1.5 * anatomical_weights
    del streamline_activations  # Free memory

    # Select only streamlines with non-zero activation for analysis
    streamline_mask = [sl for (sl, val) in zip(streamlines, streamline_activations_integrated) if val]

    # Normalize and scale weights for tck2connectome input
    streamline_scalars_mask = min_max_norm(
        streamline_activations_integrated_anat_weighted[streamline_activations_integrated_anat_weighted > 0]
    ) * 10

    weights_output_path = os.path.join(output_dir, 'connectome', f'{label}activation_integration_weights.txt')
    np.savetxt(weights_output_path,
               streamline_scalars_mask, delimiter=' ', newline=' ')

    # Save filtered active streamlines to .tck
    tractogram = nib.streamlines.Tractogram(streamline_mask, affine_to_rasmm=np.eye(4))
    active_streamline_path = os.path.join(output_dir, 'connectome', f'{label}active_streamlines.tck')
    nib.streamlines.save(tractogram, active_streamline_path)

    # Generate connectome matrix using MRtrix tck2connectome
    parcel_output = os.path.join(output_dir, 'connectome', f'{label}parcels.csv')
    parcel_assignments_output = os.path.join(output_dir, 'connectome', f'{label}assignments_parcels.csv')

    env = os.environ.copy()
    env['MRTRIX_NOGUI'] = '1'
    subprocess.run([
        'tck2connectome', '-symmetric', '-zero_diagonal',
        active_streamline_path, parcels_path, parcel_output,
        '-scale_invnodevol',
        '-tck_weights_in', weights_output_path,
        '-out_assignments', parcel_assignments_output,
        '-force'
    ], env=env)

    # Load parcellation label LUT for node names and colors
    with open(f"{os.path.dirname(os.path.realpath(__file__))}/resources/fs_label_luts.json", "r") as f:
        fs_label_luts = json.load(f)
        labels = list(fs_label_luts.keys())
        node_colors = list(fs_label_luts.values())

    # Load the resulting connectome matrix
    connectome = np.loadtxt(f'{output_dir}/connectome/{label}parcels.csv', delimiter=',')

    # Determine which nodes are highly connected for styling
    max_conn_per_node = np.max(connectome, axis=1)
    label_colors = [
        '#555555' if max_conn < np.percentile(connectome, 99.5) else 'white'
        for max_conn in max_conn_per_node
    ]

    # Normalize connectome matrix for visualization
    connectome_scaled = connectome * (1 / np.percentile(connectome, 99.95))
    np.savetxt(os.path.join(output_dir, 'connectome', f'{label}connectome.txt'), connectome_scaled)

    # Compute circular layout and plot connectome
    node_angles = mne.viz.circular_layout(labels, labels)
    fig, ax = mne_connectivity.viz.plot_connectivity_circle(
        connectome_scaled, labels, colormap='magma',
        vmin=0, vmax=1, node_angles=node_angles,
        linewidth=3, node_colors=node_colors,
        colorbar=False, fontsize_names=10
    )

    # Recolor labels based on threshold
    for text, color in zip(ax.texts, label_colors):
        text.set_color(color)

    # Save final connectome visualization
    fig.savefig(f'{output_dir}/connectome/{label}connectome.svg', dpi=300, transparent=False)
    plt.close(fig)

def average_cortical_thickness(freesurfer_dir, subject):
    """
    Returns the average cortical thickness of a FreeSurfer subject.

    Parameters:
        freesurfer_dir (str): Path to the FreeSurfer subjects directory.
        subject (str): Subject ID.

    Returns:
        float: Average cortical thickness across both hemispheres.
    """
    surf_dir = os.path.join(freesurfer_dir, subject, 'surf')
    lh_thickness = nib.freesurfer.io.read_morph_data(os.path.join(surf_dir, 'lh.thickness'))
    rh_thickness = nib.freesurfer.io.read_morph_data(os.path.join(surf_dir, 'rh.thickness'))
    # Concatenate and compute the mean thickness
    all_thickness = np.concatenate([lh_thickness, rh_thickness])
    return(np.mean(all_thickness))
    
# Pipeline Control
def run_stream4d(freesurfer_dir, subject, tractography_path, source_estimate_path, output_dir, label="", connectome=True, sift_weight_path="", stim_onset=500, time_range=np.arange(-25, 175)):
    """
    Runs STREAM-4D integration pipeline: loads surface and tractography data, thresholds source estimates,
    links activation to streamlines, and saves outputs for visualization and connectomics.

    Parameters:
        freesurfer_dir (str): Path to FreeSurfer recon-all directory.
        subject (str): Subject identifier.
        tractography_path (str): Path to .tck tractography file.
        source_estimate_path (str): Path to .stc source estimate.
        output_dir (str): Directory to store pipeline outputs.
        label (str): Optional label prefix for output files.

    Returns:
        None
    """
    start_time = time.time()
    print('------------------------------')
    print('          STREAM-4D           ')
    print('------------------------------')

    print('[]: setting up directories')
    for subdir in ['wavefront', 'tractography', 'connectome', 'source_estimates/raw', 'source_estimates/smoothed', 'source_estimates/normalized', 'render']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    tract_link_path = os.path.join(output_dir, "tractography", os.path.basename(tractography_path))
    stc_link_path = os.path.join(output_dir, "source_estimates", "raw", os.path.basename(source_estimate_path))
    os.system(f'ln -s {tractography_path} {tract_link_path}')
    os.system(f'ln -s {source_estimate_path} {stc_link_path}')


    if label:
        label = label + '_'

    print('\nLoading FreeSurfer Surface Geometry\n')
    surface_geometry = srf_to_wavefront(freesurfer_dir, subject, os.path.join(output_dir, 'wavefront'))
    surface_geometry['connectivity'] = get_sparse_connectivity(surface_geometry['faces'], surface_geometry['n_vertices'])

    print('Sampling Tractography Data\n')
    streamlines = get_streamline_subset(tractography_path, os.path.join(output_dir, 'tractography'))

    print('Extracting Sample Endpoints\n')
    streamline_endpoints = get_streamline_endpoints(streamlines)     

    print('Thresholding Source Estimate\n')
    scalars = threshold_stc(source_estimate_path=source_estimate_path, surface_geometry=surface_geometry, output_dir=output_dir, stim_onset=stim_onset, time_range=time_range, label=label)

    print('Associating Streamlines to Vertices\n')
    dist_threshold = average_cortical_thickness(freesurfer_dir, subject)
    print(f'Computing associations with distance threshold: {round(dist_threshold, 3)}mm\n')

    print('Associating Streamlines to Vertices\n')
    vertex_associations = associate_vertices_to_streamlines(surface_geometry['vertices'], streamline_endpoints, dist_threshold)

    print('Linking Activation Timeseries\n')
    link_streamline_activation(scalars, vertex_associations, streamlines, output_dir, label=label)
    print(f'Associations Complete! Renderable output saved to {output_dir}')

    if connectome:
        print('Running Connectome Analysis\n')
        print('Loading Tractography Data')
        conn_streamlines = list(nib.streamlines.load(tractography_path).streamlines)
        print('Extracting Streamline Endpoints\n')
        conn_streamline_endpoints = get_streamline_endpoints(conn_streamlines)
        conn_vertex_associations = associate_vertices_to_streamlines(surface_geometry['vertices'], conn_streamline_endpoints, dist_threshold)
        print('Creating Connectome Parcellation\n')
        parcels = create_connectome_parcels(freesurfer_dir, subject, f'{output_dir}/connectome')
        print('Creating Structural Connectome with figures\n')
        weighted_connectome_analysis(scalars, conn_vertex_associations, conn_streamlines, sift_weight_path, parcels, output_dir, label)

    elapsed_time = time.time() - start_time
    print(f'\nSTREAM-4D completed in {elapsed_time / 60:.2f} minutes ({elapsed_time:.2f} seconds).')
 
def main():
    parser = argparse.ArgumentParser(description="Assign streamline activation intensity from source estimate")
    parser.add_argument("-t", "--tractography_path", type=str, help=".tck tractography file containing streamline data")
    parser.add_argument("-e", "--source_estimate_path", type=str, help=".stc source estimation path (MNE output, compatibility to improve with future releases)")
    parser.add_argument("-w", "--sift_weight_path", type=str, default="", help=".txt tcksift output to weight structural connectome")
    parser.add_argument("-f", "--freesurfer_dir", type=str, help="Path to the directory containing FreeSurfer's 'recon-all' output.")
    parser.add_argument("-s", "--subject", type=str, help="FreeSurfer Reconall subject")
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory path")
    parser.add_argument("-l", "--label", type=str, default="", help="Optional label to prefix output files")
    parser.add_argument("--stim_onset", type=int, default=500, help="Index of stimulus onset timepoint in source estimate (default: 500)")
    parser.add_argument("--time_range", type=int, nargs=2, default=[25,175], help="Time indices relative to stim_onset to include in final estimate (default: -25 to 175 ms)")
    parser.add_argument("--no-connectome", dest="connectome", action="store_false", help="Option to not run connectome analysis (performed by default)")    
    args = parser.parse_args()

    run_stream4d(
        freesurfer_dir=args.freesurfer_dir, 
        subject=args.subject, 
        tractography_path=args.tractography_path, 
        source_estimate_path=args.source_estimate_path, 
        output_dir=args.output_dir, 
        label=args.label,
        connectome=args.connectome, 
        sift_weight_path=args.sift_weight_path, 
        stim_onset=args.stim_onset,
        time_range=np.arange(args.time_range[0], args.time_range[1])
        )

if __name__ == "__main__":
    main()