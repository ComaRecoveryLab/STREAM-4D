import os
import argparse

global code_dir 
code_dir = '/autofs/space/nicc_003/users/xander/code/stream_4d/STREAM-4D'

def setup_blender(subject, output_dir):
    """Set up Blender for rendering"""
    os.system(f"blender --background --python {code_dir}/scene_template.py -- --subject {subject} --output {output_dir}")

def keyframe_tractography(output_dir, label=""):
    """Keyframe the streamlines"""
    os.system(f"blender --background {output_dir}/{subject}.blend --python {code_dir}/keyframe_tractography.py -- --streamline_activation {output_dir}/tractography/{label}streamline_activation_timeseries.npy")

def render_output(subject, render_output_dir):
    blender_file = f'{subject}/{subject}_stream4d.blend'
    render_output_dir = f'./render/frame_####.png'

    os.system(f'''\
    blender -b {blender_file} \
    --python-expr "import bpy; bpy.data.texts['source_estimation_import'].as_module().run()" \
    --engine BLENDER_EEVEE \
    --render-output {output_dir}/render/frame_####.png \
    --render-anim\
    ''')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for Blender script.")
    parser.add_argument("-s", "--subject", required=True, help="Subject identifier")
    parser.add_argument("-l", "--label", required=False, help="Subject identifier")
    parser.add_argument("-o", "--output", required=True, help="STREAM-4D Subject Output Directory")
    
    args = parser.parse_args()
    subject = args.subject
    output_dir = args.output
    label = args.label if args.label else ""
    if label:
        label = f"{label}_"

    setup_blender(subject, output_dir)
    keyframe_tractography(output_dir, label)

    # render_output(subject, output_dir)