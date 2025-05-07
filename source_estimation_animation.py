import bpy
import bmesh
import numpy as np


def update_intensity(scene):
    current_frame = scene.frame_current

    # Ensure frame number is within bounds
    if current_frame < 0 or current_frame >= n_frames:
        return

    # Update vertex intensity
    obj = bpy.data.objects.get("pial")
    if obj:
        mesh = obj.data
        intensity_rh = mesh.attributes["intensity"]
        intensity_rh.data.foreach_set(
            "value", (scalar_values_timeseries[:, current_frame])
        )
        mesh.update()

scalar_values_timeseries = np.load(bpy.path.abspath(f'//source_estimates/normalized/normalized.npy'))
n_frames = scalar_values_timeseries.shape[-1]

# Clear existing handlers to avoid duplicates
bpy.app.handlers.frame_change_pre.clear()

# Add the update function to frame change handlers
bpy.app.handlers.frame_change_pre.append(update_intensity)