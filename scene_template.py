import bpy
import os
import argparse
import numpy as np
import json
import sys

def parse_arguments():
    # Remove Blender's own arguments by splitting at "--".
    if "--" not in sys.argv:
        print("Error: Missing '--' to separate Blender args from script args.")
        sys.exit(1)

    script_args = sys.argv[sys.argv.index("--") + 1:]

    # Set up argparse for the script arguments.
    parser = argparse.ArgumentParser(description="Parse arguments for Blender script.")
    parser.add_argument("--subject", required=True, help="Subject identifier")
    parser.add_argument("--output", required=True, help="STREAM 3D Subject Output Directory")

    args = parser.parse_args(script_args)
    return args

args = parse_arguments()
subject, output_dir = args.subject, args.output

bpy.ops.wm.save_as_mainfile(filepath=f"{output_dir}/{subject}.blend")

cwd = os.getcwd()
blender_template_path = os.path.join(cwd,"template.blend")

#WORLD SETUP
for node_group in ["white_matter", "streamline_geometry"]:
    bpy.ops.wm.append(filename=node_group, directory=os.path.join(blender_template_path, "NodeTree"))

for material in ["streamlines", "wireframe", "tms_map"]:
    bpy.ops.wm.append(filename=material, directory=os.path.join(blender_template_path, "Material"))

# Remove all default objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

context = bpy.context
scene = context.scene
for c in scene.collection.children:
    scene.collection.children.unlink(c)
    
# Create new collections
grey_matter = bpy.data.collections.new("grey_matter")
bpy.context.scene.collection.children.link(grey_matter)

white_matter = bpy.data.collections.new("white_matter")
bpy.context.scene.collection.children.link(white_matter)

wavefront_dir = f'{output_dir}/wavefront'
if not os.path.exists(wavefront_dir):
    raise FileNotFoundError(f"Directory not found: {wavefront_dir}")

# grey_matter_objects = {"lh.pial", "rh.pial", "cerebellum_cortex"}
# white_matter_objects = {"lh.white", "rh.white", "cerebellum_white_matter", "brain_stem"}

grey_matter_objects = {"lh.pial", "rh.pial"}
white_matter_objects = {"lh.white", "rh.white"}

# Import all .obj files and organize into collections
for obj_file in os.listdir(wavefront_dir):
    if obj_file.endswith('.obj'):
        obj_path = os.path.join(wavefront_dir, obj_file)
        bpy.ops.wm.obj_import(filepath=obj_path)
        obj_name = os.path.splitext(obj_file)[0]
        obj = bpy.context.selected_objects[0]
        
        # Reset X rotation to 0 degrees
        obj.rotation_euler[0] = 0
        
        if obj_name in grey_matter_objects:
            grey_matter.objects.link(obj)
            bpy.context.scene.collection.objects.unlink(obj)
            tms_material = bpy.data.materials["tms_map"]
            obj.data.materials.append(tms_material)
        elif obj_name in white_matter_objects:
            white_matter.objects.link(obj)
            bpy.context.scene.collection.objects.unlink(obj)
            # Add a geometry nodes modifier and set the node group to "white_matter"
            geo_modifier = obj.modifiers.new(name="GeometryNodes", type='NODES')
            geo_modifier.node_group = bpy.data.node_groups["white_matter"]

bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[0].default_value = (0, 0, 0, 1)

# Load the streamlines
with open(f'{output_dir}/tractography/streamline_subset.txt', 'r') as file:
    input_streamlines = json.load(file)

streamlines = [np.array(streamline) for streamline in input_streamlines]

# Get the pre-existing "streamline_geometry" geometry nodes setup
streamline_nodes = bpy.data.node_groups.get("streamline_geometry")

# Create a new collection for the streamlines
streamline_collection = bpy.data.collections.new("Streamlines Collection")
bpy.context.scene.collection.children.link(streamline_collection)

# Iterate over each streamline
for i, sl in enumerate(streamlines):
    # Create a new curve for each streamline
    curve = bpy.data.curves.new(name=f'streamline_{i}', type='CURVE')
    curve.dimensions = '3D'
    curve.resolution_u = 2

    spline = curve.splines.new('NURBS')
    spline.points.add(len(sl) - 1)
    
    # Assign points to the spline
    for j, point in enumerate(sl):
        spline.points[j].co = np.append(point, 1)

    # Create object and link it to the collection
    obj = bpy.data.objects.new(f'streamline_{i}', curve)
    streamline_collection.objects.link(obj)

    # Add the geometry nodes modifier and assign the 'streamline_geometry' node group
    modifier = obj.modifiers.new(name="Streamline Geometry", type='NODES')
    modifier.node_group = streamline_nodes

# Save File
bpy.ops.wm.save_mainfile()