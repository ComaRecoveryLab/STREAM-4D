import os

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


print("\nSetting up Blender for rendering\n")
setup_blender(subject, output_dir)
print("Blender setup complete\n")