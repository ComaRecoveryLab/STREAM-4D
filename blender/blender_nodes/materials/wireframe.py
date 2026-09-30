import bpy
import mathutils
import os
import typing


wireframe = bpy.data.materials.new(name = "wireframe")
if bpy.app.version < (5, 0, 0):
    wireframe.use_nodes = True


wireframe.alpha_threshold = 0.5
wireframe.line_priority = 0
wireframe.max_vertex_displacement = 0.0
wireframe.metallic = 0.0
wireframe.paint_active_slot = 0
wireframe.paint_clone_slot = 0
wireframe.pass_index = 0
wireframe.refraction_depth = 0.0
wireframe.roughness = 0.4000000059604645
wireframe.show_transparent_back = True
wireframe.specular_intensity = 0.5
wireframe.use_backface_culling = False
wireframe.use_backface_culling_lightprobe_volume = False
wireframe.use_backface_culling_shadow = False
wireframe.use_preview_world = False
wireframe.use_raytrace_refraction = False
wireframe.use_screen_refraction = False
wireframe.use_sss_translucency = False
wireframe.use_thickness_from_shadow = False
wireframe.use_transparency_overlap = True
wireframe.use_transparent_shadow = False
wireframe.blend_method = 'HASHED'
wireframe.displacement_method = 'BUMP'
wireframe.preview_render_type = 'SPHERE'
wireframe.surface_render_method = 'DITHERED'
wireframe.thickness_mode = 'SPHERE'
wireframe.volume_intersection_method = 'FAST'
wireframe.specular_color = (1.0, 1.0, 1.0)
wireframe.diffuse_color = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
wireframe.line_color = (0.0, 0.0, 0.0, 0.0)

def shader_nodetree_node_group(node_tree_names: dict[typing.Callable, str]):
    """Initialize Shader Nodetree node group"""
    shader_nodetree = wireframe.node_tree

    # Start with a clean node tree
    for node in shader_nodetree.nodes:
        shader_nodetree.nodes.remove(node)
    shader_nodetree.color_tag = 'NONE'
    shader_nodetree.description = ""
    shader_nodetree.default_group_node_width = 140
    # Initialize shader_nodetree nodes

    # Node Material Output
    material_output = shader_nodetree.nodes.new("ShaderNodeOutputMaterial")
    material_output.name = "Material Output"
    material_output.is_active_output = True
    material_output.target = 'ALL'
    # Displacement
    material_output.inputs[2].default_value = (0.0, 0.0, 0.0)
    # Thickness
    material_output.inputs[3].default_value = 0.0

    # Node Emission
    emission = shader_nodetree.nodes.new("ShaderNodeEmission")
    emission.name = "Emission"
    # Color
    emission.inputs[0].default_value = (0.25, 0.25, 0.25, 1.0)
    # Strength
    emission.inputs[1].default_value = 1.0

    # Set locations
    shader_nodetree.nodes["Material Output"].location = (300.0, 300.0)
    shader_nodetree.nodes["Emission"].location = (15.167854309082031, 257.8080749511719)

    # Set dimensions
    shader_nodetree.nodes["Material Output"].width  = 140.0
    shader_nodetree.nodes["Material Output"].height = 100.0

    shader_nodetree.nodes["Emission"].width  = 140.0
    shader_nodetree.nodes["Emission"].height = 100.0


    # Initialize shader_nodetree links

    # emission.Emission -> material_output.Surface
    shader_nodetree.links.new(
        shader_nodetree.nodes["Emission"].outputs[0],
        shader_nodetree.nodes["Material Output"].inputs[0]
    )

    return shader_nodetree


if __name__ == "__main__":
    # Maps node tree creation functions to the node tree 
    # name, such that we don't recreate node trees unnecessarily
    node_tree_names : dict[typing.Callable, str] = {}

    shader_nodetree = shader_nodetree_node_group(node_tree_names)
    node_tree_names[shader_nodetree_node_group] = shader_nodetree.name

