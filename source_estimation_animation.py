import bpy
import bmesh
import numpy as np

scalar_values_timeseries_lh = np.load(bpy.path.abspath(f'//source_estimates/normalized/normalized_lh.npy'))
scalar_values_timeseries_rh = np.load(bpy.path.abspath(f'//source_estimates/normalized/normalized_rh.npy'))
n_frames = scalar_values_timeseries_lh.shape[-1]

def update_intensity(scene):
    current_frame = scene.frame_current

    # Ensure frame number is within bounds
    if current_frame < 0 or current_frame >= n_frames:
        return

    # Update intensity for the right hemisphere
    obj_rh = bpy.data.objects.get("rh.pial")
    if obj_rh:
        mesh_rh = obj_rh.data
        intensity_rh = mesh_rh.attributes["intensity"]
        intensity_rh.data.foreach_set(
            "value", (scalar_values_timeseries_rh[:, current_frame])
        )
        mesh_rh.update()

    # Update intensity for the left hemisphere
    obj_lh = bpy.data.objects.get("lh.pial")
    if obj_lh:
        mesh_lh = obj_lh.data
        intensity_lh = mesh_lh.attributes["intensity"]
        intensity_lh.data.foreach_set(
            "value", (scalar_values_timeseries_lh[:, current_frame])
        )
        mesh_lh.update()

# Clear existing handlers to avoid duplicates
bpy.app.handlers.frame_change_pre.clear()

# Add the update function to frame change handlers
bpy.app.handlers.frame_change_pre.append(update_intensity)