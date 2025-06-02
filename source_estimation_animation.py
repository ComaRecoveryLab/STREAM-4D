import bpy
import bmesh
import sys
import argparse
import numpy as np

def parse_arguments():
    # Remove Blender's arguments by splitting at "--"
    if "--" not in sys.argv:
        print("Error: Missing '--' to separate Blender args from script args.")
        sys.exit(1)

    script_args = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Parse arguments for Blender script.")
    parser.add_argument("--source_estimation", required=True, help=".npy file containing source estimation timeseries data")
    parser.add_argument("--render-frame", type=int, help="Render a single frame")
    parser.add_argument("--render-anim", action="store_true", help="Render an animation")
    
    args = parser.parse_args(script_args)
    return args

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

# Parse arguments
args = parse_arguments()
stc_path = args.source_estimation
print(f"Rendering visualization using data from: {stc_path}")

# Load source estimate
scalar_values_timeseries = np.load(bpy.path.abspath(stc_path))
n_frames = scalar_values_timeseries.shape[-1]

# Clear and register frame change handler
bpy.app.handlers.frame_change_pre.clear()
bpy.app.handlers.frame_change_pre.append(update_intensity)

# Perform rendering
if args.render_frame is not None:
    bpy.context.scene.frame_set(args.render_frame)
    bpy.ops.render.render(write_still=True)

elif args.render_anim:
    bpy.ops.render.render(animation=True)