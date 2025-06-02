import os
import argparse

global code_dir
# Set code directory 
code_dir = os.path.dirname(os.path.abspath(__file__))

def setup_blender(subject, output_dir):
    """Set up Blender for rendering"""
    print(f"blender --background --python {code_dir}/scene_template.py -- --subject {subject} --output {output_dir}")

def keyframe_tractography(output_dir, label=""):
    """Keyframe the streamlines"""
    print(f"blender --background {output_dir}/{subject}.blend --python {code_dir}/keyframe_tractography.py -- --streamline_activation {output_dir}/tractography/{label}streamline_activation_timeseries.npy")

def render_output(subject, output_dir, label=""):
    blender_file = f'{output_dir}/{subject}.blend'

    os.system(f'''\
    blender --background {blender_file} \
    --engine BLENDER_EEVEE \
    --render-output {output_dir}/render/frame_####.png \
    --python {code_dir}/source_estimation_animation.py -- --source_estimation {output_dir}/source_estimation/normalized/{label}normalized.npy \
    --render-anim
    ''')
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse arguments for Blender script.")
    parser.add_argument("-s", "--subject", required=True, help="Subject Identifier")
    parser.add_argument("-l", "--label", required=False, help="Session Label")
    parser.add_argument("-o", "--output_dir", required=True, help="STREAM-4D Subject Output Directory")
    
    args = parser.parse_args()
    subject = args.subject
    output_dir = args.output_dir
    label = args.label if args.label else ""
    if label:
        label = f"{label}_"

    setup_blender(subject, output_dir)
    keyframe_tractography(output_dir, label)
    # render_output(subject, output_dir)

