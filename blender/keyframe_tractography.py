import bpy
import sys
import numpy as np
import argparse

def parse_arguments():
    # Remove Blender's own arguments by splitting at "--".
    if "--" not in sys.argv:
        print("Error: Missing '--' to separate Blender args from script args.")
        sys.exit(1)

    script_args = sys.argv[sys.argv.index("--") + 1:]

    # Set up argparse for the script arguments.
    parser = argparse.ArgumentParser(description="Parse arguments for Blender script.")
    parser.add_argument("--streamline_activation", required=True, help=".npy file containing streamline activation timeseries")

    args = parser.parse_args(script_args)
    return args

args = parse_arguments()
streamline_activation_timeseries_path = args.streamline_activation

# Load the activation time series
streamline_activation_timeseries = np.load(streamline_activation_timeseries_path)
# Retrieve the existing "Streamlines Collection"
streamline_collection = bpy.data.collections.get("Streamlines Collection")
if streamline_collection is None:
    raise ValueError("Streamlines Collection not found. Please ensure the streamlines are imported and in the Blender file.")

else:
    print(f"Streamline sample: {streamline_collection.name} found with {len(streamline_collection.objects)} streamlines.")

# Iterate over each streamline and set activation keyframes
for index, obj in enumerate(streamline_collection.objects):
    # print(f"\rKeyframing streamline {index + 1}/{len(streamline_collection.objects)}")
    activations = streamline_activation_timeseries[index, :]
    
    # Set keyframes for the 'activation' property
    for frame, is_active in enumerate(activations, start=1):
        if "activation" not in obj:
            obj["activation"] = 0.0
            obj.id_properties_ui("activation").update(min=0.0, max=1.0)

        obj["activation"] = is_active
        obj.keyframe_insert(data_path='["activation"]', frame=frame)

print(f"Keyframing complete")

# Save File
bpy.ops.wm.save_mainfile()