import bpy
import mathutils
import os
import typing


def streamline_rgb_1_node_group(node_tree_names: dict[typing.Callable, str]):
    """Initialize streamline_RGB node group"""
    streamline_rgb_1 = bpy.data.node_groups.new(type = 'ShaderNodeTree', name = "streamline_RGB")

    streamline_rgb_1.color_tag = 'NONE'
    streamline_rgb_1.description = ""
    streamline_rgb_1.default_group_node_width = 140
    # streamline_rgb_1 interface

    # Socket Color
    color_socket = streamline_rgb_1.interface.new_socket(name="Color", in_out='OUTPUT', socket_type='NodeSocketColor')
    color_socket.default_value = (0.0, 0.0, 0.0, 0.0)
    color_socket.attribute_domain = 'POINT'
    color_socket.default_input = 'VALUE'
    color_socket.structure_type = 'AUTO'

    # Initialize streamline_rgb_1 nodes

    # Node Group Input
    group_input = streamline_rgb_1.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # Node Separate XYZ
    separate_xyz = streamline_rgb_1.nodes.new("ShaderNodeSeparateXYZ")
    separate_xyz.name = "Separate XYZ"

    # Node Attribute
    attribute = streamline_rgb_1.nodes.new("ShaderNodeAttribute")
    attribute.name = "Attribute"
    attribute.attribute_name = "curve_tangent"
    attribute.attribute_type = 'GEOMETRY'

    # Node Math.001
    math_001 = streamline_rgb_1.nodes.new("ShaderNodeMath")
    math_001.name = "Math.001"
    math_001.hide = True
    math_001.operation = 'ABSOLUTE'
    math_001.use_clamp = False

    # Node Math
    math = streamline_rgb_1.nodes.new("ShaderNodeMath")
    math.name = "Math"
    math.hide = True
    math.operation = 'ABSOLUTE'
    math.use_clamp = False

    # Node Combine Color
    combine_color = streamline_rgb_1.nodes.new("ShaderNodeCombineColor")
    combine_color.name = "Combine Color"
    combine_color.mode = 'RGB'

    # Node Math.002
    math_002 = streamline_rgb_1.nodes.new("ShaderNodeMath")
    math_002.name = "Math.002"
    math_002.hide = True
    math_002.operation = 'ABSOLUTE'
    math_002.use_clamp = False

    # Node Group Output
    group_output = streamline_rgb_1.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # Set locations
    streamline_rgb_1.nodes["Group Input"].location = (-707.8698120117188, 0.0)
    streamline_rgb_1.nodes["Separate XYZ"].location = (-321.701416015625, -39.890960693359375)
    streamline_rgb_1.nodes["Attribute"].location = (-509.07769775390625, -87.07675170898438)
    streamline_rgb_1.nodes["Math.001"].location = (-92.39863586425781, -74.28726196289062)
    streamline_rgb_1.nodes["Math"].location = (-92.92877197265625, -25.465396881103516)
    streamline_rgb_1.nodes["Combine Color"].location = (181.68682861328125, 20.682109832763672)
    streamline_rgb_1.nodes["Math.002"].location = (-90.05878448486328, -112.40291595458984)
    streamline_rgb_1.nodes["Group Output"].location = (424.00445556640625, 20.798175811767578)

    # Set dimensions
    streamline_rgb_1.nodes["Group Input"].width  = 140.0
    streamline_rgb_1.nodes["Group Input"].height = 100.0

    streamline_rgb_1.nodes["Separate XYZ"].width  = 140.0
    streamline_rgb_1.nodes["Separate XYZ"].height = 100.0

    streamline_rgb_1.nodes["Attribute"].width  = 140.0
    streamline_rgb_1.nodes["Attribute"].height = 100.0

    streamline_rgb_1.nodes["Math.001"].width  = 140.0
    streamline_rgb_1.nodes["Math.001"].height = 100.0

    streamline_rgb_1.nodes["Math"].width  = 140.0
    streamline_rgb_1.nodes["Math"].height = 100.0

    streamline_rgb_1.nodes["Combine Color"].width  = 140.0
    streamline_rgb_1.nodes["Combine Color"].height = 100.0

    streamline_rgb_1.nodes["Math.002"].width  = 140.0
    streamline_rgb_1.nodes["Math.002"].height = 100.0

    streamline_rgb_1.nodes["Group Output"].width  = 140.0
    streamline_rgb_1.nodes["Group Output"].height = 100.0


    # Initialize streamline_rgb_1 links

    # separate_xyz.X -> math.Value
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Separate XYZ"].outputs[0],
        streamline_rgb_1.nodes["Math"].inputs[0]
    )
    # math.Value -> combine_color.Red
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Math"].outputs[0],
        streamline_rgb_1.nodes["Combine Color"].inputs[0]
    )
    # math_001.Value -> combine_color.Green
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Math.001"].outputs[0],
        streamline_rgb_1.nodes["Combine Color"].inputs[1]
    )
    # separate_xyz.Y -> math_001.Value
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Separate XYZ"].outputs[1],
        streamline_rgb_1.nodes["Math.001"].inputs[0]
    )
    # combine_color.Color -> group_output.Color
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Combine Color"].outputs[0],
        streamline_rgb_1.nodes["Group Output"].inputs[0]
    )
    # attribute.Vector -> separate_xyz.Vector
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Attribute"].outputs[1],
        streamline_rgb_1.nodes["Separate XYZ"].inputs[0]
    )
    # separate_xyz.Z -> math_002.Value
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Separate XYZ"].outputs[2],
        streamline_rgb_1.nodes["Math.002"].inputs[0]
    )
    # math_002.Value -> combine_color.Blue
    streamline_rgb_1.links.new(
        streamline_rgb_1.nodes["Math.002"].outputs[0],
        streamline_rgb_1.nodes["Combine Color"].inputs[2]
    )

    return streamline_rgb_1


streamlines = bpy.data.materials.new(name = "streamlines")
if bpy.app.version < (5, 0, 0):
    streamlines.use_nodes = True


streamlines.alpha_threshold = 0.5
streamlines.line_priority = 0
streamlines.max_vertex_displacement = 0.0
streamlines.metallic = 0.0
streamlines.paint_active_slot = 0
streamlines.paint_clone_slot = 0
streamlines.pass_index = 0
streamlines.refraction_depth = 0.0
streamlines.roughness = 0.4000000059604645
streamlines.show_transparent_back = True
streamlines.specular_intensity = 0.5
streamlines.use_backface_culling = False
streamlines.use_backface_culling_lightprobe_volume = False
streamlines.use_backface_culling_shadow = False
streamlines.use_preview_world = False
streamlines.use_raytrace_refraction = False
streamlines.use_screen_refraction = False
streamlines.use_sss_translucency = False
streamlines.use_thickness_from_shadow = False
streamlines.use_transparency_overlap = True
streamlines.use_transparent_shadow = False
streamlines.blend_method = 'BLEND'
streamlines.displacement_method = 'BUMP'
streamlines.preview_render_type = 'SPHERE'
streamlines.surface_render_method = 'BLENDED'
streamlines.thickness_mode = 'SPHERE'
streamlines.volume_intersection_method = 'FAST'
streamlines.specular_color = (1.0, 1.0, 1.0)
streamlines.diffuse_color = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
streamlines.line_color = (0.0, 0.0, 0.0, 0.0)

def shader_nodetree_node_group(node_tree_names: dict[typing.Callable, str]):
    """Initialize Shader Nodetree node group"""
    shader_nodetree = streamlines.node_tree

    # Start with a clean node tree
    for node in shader_nodetree.nodes:
        shader_nodetree.nodes.remove(node)
    shader_nodetree.color_tag = 'NONE'
    shader_nodetree.description = ""
    shader_nodetree.default_group_node_width = 140
    # Initialize shader_nodetree nodes

    # Node Attribute
    attribute = shader_nodetree.nodes.new("ShaderNodeAttribute")
    attribute.name = "Attribute"
    attribute.attribute_name = "activation"
    attribute.attribute_type = 'OBJECT'

    # Node Math.008
    math_008 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_008.name = "Math.008"
    math_008.operation = 'GREATER_THAN'
    math_008.use_clamp = False
    # Value_001
    math_008.inputs[1].default_value = 0.0

    # Node Math.005
    math_005 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_005.name = "Math.005"
    math_005.operation = 'MULTIPLY'
    math_005.use_clamp = False
    # Value_001
    math_005.inputs[1].default_value = 0.10000000149011612

    # Node Material Output
    material_output = shader_nodetree.nodes.new("ShaderNodeOutputMaterial")
    material_output.name = "Material Output"
    material_output.is_active_output = True
    material_output.target = 'ALL'
    # Displacement
    material_output.inputs[2].default_value = (0.0, 0.0, 0.0)
    # Thickness
    material_output.inputs[3].default_value = 0.0

    # Node Invert Color.001
    invert_color_001 = shader_nodetree.nodes.new("ShaderNodeInvert")
    invert_color_001.name = "Invert Color.001"
    # Fac
    invert_color_001.inputs[0].default_value = 1.0

    # Node Principled BSDF
    principled_bsdf = shader_nodetree.nodes.new("ShaderNodeBsdfPrincipled")
    principled_bsdf.name = "Principled BSDF"
    principled_bsdf.distribution = 'GGX'
    principled_bsdf.subsurface_method = 'RANDOM_WALK_SKIN'
    # Metallic
    principled_bsdf.inputs[1].default_value = 0.0
    # Roughness
    principled_bsdf.inputs[2].default_value = 0.5
    # IOR
    principled_bsdf.inputs[3].default_value = 1.4500000476837158
    # Normal
    principled_bsdf.inputs[5].default_value = (0.0, 0.0, 0.0)
    # Diffuse Roughness
    principled_bsdf.inputs[7].default_value = 0.0
    # Subsurface Weight
    principled_bsdf.inputs[8].default_value = 0.0
    # Subsurface Radius
    principled_bsdf.inputs[9].default_value = (1.0, 0.20000000298023224, 0.10000000149011612)
    # Subsurface Scale
    principled_bsdf.inputs[10].default_value = 0.05000000074505806
    # Subsurface IOR
    principled_bsdf.inputs[11].default_value = 1.399999976158142
    # Subsurface Anisotropy
    principled_bsdf.inputs[12].default_value = 0.0
    # Specular IOR Level
    principled_bsdf.inputs[13].default_value = 0.5
    # Specular Tint
    principled_bsdf.inputs[14].default_value = (1.0, 1.0, 1.0, 1.0)
    # Anisotropic
    principled_bsdf.inputs[15].default_value = 0.0
    # Anisotropic Rotation
    principled_bsdf.inputs[16].default_value = 0.0
    # Tangent
    principled_bsdf.inputs[17].default_value = (0.0, 0.0, 0.0)
    # Transmission Weight
    principled_bsdf.inputs[18].default_value = 0.0
    # Coat Weight
    principled_bsdf.inputs[19].default_value = 0.0
    # Coat Roughness
    principled_bsdf.inputs[20].default_value = 0.029999999329447746
    # Coat IOR
    principled_bsdf.inputs[21].default_value = 1.5
    # Coat Tint
    principled_bsdf.inputs[22].default_value = (1.0, 1.0, 1.0, 1.0)
    # Coat Normal
    principled_bsdf.inputs[23].default_value = (0.0, 0.0, 0.0)
    # Sheen Weight
    principled_bsdf.inputs[24].default_value = 0.0
    # Sheen Roughness
    principled_bsdf.inputs[25].default_value = 0.5
    # Sheen Tint
    principled_bsdf.inputs[26].default_value = (1.0, 1.0, 1.0, 1.0)
    # Thin Film Thickness
    principled_bsdf.inputs[29].default_value = 0.0
    # Thin Film IOR
    principled_bsdf.inputs[30].default_value = 1.3300000429153442

    # Node Math.006
    math_006 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_006.name = "Math.006"
    math_006.operation = 'ADD'
    math_006.use_clamp = False
    # Value_001
    math_006.inputs[1].default_value = 0.019999999552965164

    # Node Math.004
    math_004 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_004.name = "Math.004"
    math_004.operation = 'ADD'
    math_004.use_clamp = False
    # Value_001
    math_004.inputs[1].default_value = 1.0

    # Node Mix
    mix = shader_nodetree.nodes.new("ShaderNodeMix")
    mix.name = "Mix"
    mix.blend_type = 'MIX'
    mix.clamp_factor = True
    mix.clamp_result = False
    mix.data_type = 'RGBA'
    mix.factor_mode = 'UNIFORM'
    # B_Color
    mix.inputs[7].default_value = (0.13300000131130219, 0.13300000131130219, 0.13300000131130219, 1.0)

    # Node Color Ramp.001
    color_ramp_001 = shader_nodetree.nodes.new("ShaderNodeValToRGB")
    color_ramp_001.label = "hot_colormap"
    color_ramp_001.name = "Color Ramp.001"
    color_ramp_001.hide = True
    color_ramp_001.color_ramp.color_mode = 'RGB'
    color_ramp_001.color_ramp.hue_interpolation = 'NEAR'
    color_ramp_001.color_ramp.interpolation = 'LINEAR'

    # Initialize color ramp elements
    color_ramp_001.color_ramp.elements.remove(color_ramp_001.color_ramp.elements[0])
    color_ramp_001_cre_0 = color_ramp_001.color_ramp.elements[0]
    color_ramp_001_cre_0.position = 0.0
    color_ramp_001_cre_0.alpha = 1.0
    color_ramp_001_cre_0.color = (1.0, 0.0, 0.012271025218069553, 1.0)

    color_ramp_001_cre_1 = color_ramp_001.color_ramp.elements.new(0.4102271795272827)
    color_ramp_001_cre_1.alpha = 1.0
    color_ramp_001_cre_1.color = (1.0, 0.25525325536727905, 0.0, 1.0)

    color_ramp_001_cre_2 = color_ramp_001.color_ramp.elements.new(0.852272629737854)
    color_ramp_001_cre_2.alpha = 1.0
    color_ramp_001_cre_2.color = (1.0, 0.9939420819282532, 0.0, 1.0)

    color_ramp_001_cre_3 = color_ramp_001.color_ramp.elements.new(1.0)
    color_ramp_001_cre_3.alpha = 1.0
    color_ramp_001_cre_3.color = (1.0, 0.7820460200309753, 0.7216414213180542, 1.0)


    # Node Group
    group = shader_nodetree.nodes.new("ShaderNodeGroup")
    group.name = "Group"
    group.node_tree = bpy.data.node_groups[node_tree_names[streamline_rgb_1_node_group]]

    # Node Math.001
    math_001 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_001.name = "Math.001"
    math_001.operation = 'MULTIPLY'
    math_001.use_clamp = False
    # Value_001
    math_001.inputs[1].default_value = 4.0

    # Node Color Ramp
    color_ramp = shader_nodetree.nodes.new("ShaderNodeValToRGB")
    color_ramp.label = "cool_colormap"
    color_ramp.name = "Color Ramp"
    color_ramp.color_ramp.color_mode = 'RGB'
    color_ramp.color_ramp.hue_interpolation = 'NEAR'
    color_ramp.color_ramp.interpolation = 'LINEAR'

    # Initialize color ramp elements
    color_ramp.color_ramp.elements.remove(color_ramp.color_ramp.elements[0])
    color_ramp_cre_0 = color_ramp.color_ramp.elements[0]
    color_ramp_cre_0.position = 0.0
    color_ramp_cre_0.alpha = 1.0
    color_ramp_cre_0.color = (0.12999999523162842, 0.12999999523162842, 0.12999999523162842, 1.0)

    color_ramp_cre_1 = color_ramp.color_ramp.elements.new(0.20000000298023224)
    color_ramp_cre_1.alpha = 1.0
    color_ramp_cre_1.color = (0.25421392917633057, 0.0, 1.0, 1.0)

    color_ramp_cre_2 = color_ramp.color_ramp.elements.new(0.4000000059604645)
    color_ramp_cre_2.alpha = 1.0
    color_ramp_cre_2.color = (0.0, 0.03477068617939949, 1.0, 1.0)

    color_ramp_cre_3 = color_ramp.color_ramp.elements.new(0.800000011920929)
    color_ramp_cre_3.alpha = 1.0
    color_ramp_cre_3.color = (0.0, 0.7332571744918823, 1.0, 1.0)

    color_ramp_cre_4 = color_ramp.color_ramp.elements.new(1.0)
    color_ramp_cre_4.alpha = 1.0
    color_ramp_cre_4.color = (0.899669349193573, 0.7461378574371338, 1.0, 1.0)


    # Set locations
    shader_nodetree.nodes["Attribute"].location = (-364.28985595703125, 145.5752410888672)
    shader_nodetree.nodes["Math.008"].location = (69.92945861816406, 100.12646484375)
    shader_nodetree.nodes["Math.005"].location = (293.0906677246094, -29.048782348632812)
    shader_nodetree.nodes["Material Output"].location = (1142.5408935546875, 362.6705017089844)
    shader_nodetree.nodes["Invert Color.001"].location = (293.4691162109375, 93.2122802734375)
    shader_nodetree.nodes["Principled BSDF"].location = (841.5931396484375, 338.6326599121094)
    shader_nodetree.nodes["Math.006"].location = (466.4854736328125, -32.111785888671875)
    shader_nodetree.nodes["Math.004"].location = (470.2176818847656, -208.53533935546875)
    shader_nodetree.nodes["Mix"].location = (560.8639526367188, 341.391845703125)
    shader_nodetree.nodes["Color Ramp.001"].location = (153.50033569335938, 215.13327026367188)
    shader_nodetree.nodes["Group"].location = (-43.77235412597656, 451.0094299316406)
    shader_nodetree.nodes["Math.001"].location = (289.1485900878906, -207.838623046875)
    shader_nodetree.nodes["Color Ramp"].location = (163.42022705078125, 448.226318359375)

    # Set dimensions
    shader_nodetree.nodes["Attribute"].width  = 140.0
    shader_nodetree.nodes["Attribute"].height = 100.0

    shader_nodetree.nodes["Math.008"].width  = 140.0
    shader_nodetree.nodes["Math.008"].height = 100.0

    shader_nodetree.nodes["Math.005"].width  = 140.0
    shader_nodetree.nodes["Math.005"].height = 100.0

    shader_nodetree.nodes["Material Output"].width  = 140.0
    shader_nodetree.nodes["Material Output"].height = 100.0

    shader_nodetree.nodes["Invert Color.001"].width  = 140.0
    shader_nodetree.nodes["Invert Color.001"].height = 100.0

    shader_nodetree.nodes["Principled BSDF"].width  = 240.0
    shader_nodetree.nodes["Principled BSDF"].height = 100.0

    shader_nodetree.nodes["Math.006"].width  = 140.0
    shader_nodetree.nodes["Math.006"].height = 100.0

    shader_nodetree.nodes["Math.004"].width  = 140.0
    shader_nodetree.nodes["Math.004"].height = 100.0

    shader_nodetree.nodes["Mix"].width  = 140.0
    shader_nodetree.nodes["Mix"].height = 100.0

    shader_nodetree.nodes["Color Ramp.001"].width  = 240.0
    shader_nodetree.nodes["Color Ramp.001"].height = 100.0

    shader_nodetree.nodes["Group"].width  = 173.386474609375
    shader_nodetree.nodes["Group"].height = 100.0

    shader_nodetree.nodes["Math.001"].width  = 140.0
    shader_nodetree.nodes["Math.001"].height = 100.0

    shader_nodetree.nodes["Color Ramp"].width  = 240.0
    shader_nodetree.nodes["Color Ramp"].height = 100.0


    # Initialize shader_nodetree links

    # principled_bsdf.BSDF -> material_output.Surface
    shader_nodetree.links.new(
        shader_nodetree.nodes["Principled BSDF"].outputs[0],
        shader_nodetree.nodes["Material Output"].inputs[0]
    )
    # mix.Result -> principled_bsdf.Base Color
    shader_nodetree.links.new(
        shader_nodetree.nodes["Mix"].outputs[2],
        shader_nodetree.nodes["Principled BSDF"].inputs[0]
    )
    # math_008.Value -> invert_color_001.Color
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.008"].outputs[0],
        shader_nodetree.nodes["Invert Color.001"].inputs[1]
    )
    # math_005.Value -> math_006.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.005"].outputs[0],
        shader_nodetree.nodes["Math.006"].inputs[0]
    )
    # attribute.Factor -> math_008.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute"].outputs[2],
        shader_nodetree.nodes["Math.008"].inputs[0]
    )
    # math_001.Value -> math_004.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.001"].outputs[0],
        shader_nodetree.nodes["Math.004"].inputs[0]
    )
    # math_008.Value -> math_005.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.008"].outputs[0],
        shader_nodetree.nodes["Math.005"].inputs[0]
    )
    # invert_color_001.Color -> mix.Factor
    shader_nodetree.links.new(
        shader_nodetree.nodes["Invert Color.001"].outputs[0],
        shader_nodetree.nodes["Mix"].inputs[0]
    )
    # color_ramp.Color -> mix.A
    shader_nodetree.links.new(
        shader_nodetree.nodes["Color Ramp"].outputs[0],
        shader_nodetree.nodes["Mix"].inputs[6]
    )
    # math_006.Value -> principled_bsdf.Alpha
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.006"].outputs[0],
        shader_nodetree.nodes["Principled BSDF"].inputs[4]
    )
    # math_004.Value -> principled_bsdf.Emission Strength
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.004"].outputs[0],
        shader_nodetree.nodes["Principled BSDF"].inputs[28]
    )
    # mix.Result -> principled_bsdf.Emission Color
    shader_nodetree.links.new(
        shader_nodetree.nodes["Mix"].outputs[2],
        shader_nodetree.nodes["Principled BSDF"].inputs[27]
    )
    # attribute.Factor -> color_ramp.Factor
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute"].outputs[2],
        shader_nodetree.nodes["Color Ramp"].inputs[0]
    )
    # attribute.Factor -> color_ramp_001.Factor
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute"].outputs[2],
        shader_nodetree.nodes["Color Ramp.001"].inputs[0]
    )
    # attribute.Factor -> math_001.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute"].outputs[2],
        shader_nodetree.nodes["Math.001"].inputs[0]
    )

    return shader_nodetree


if __name__ == "__main__":
    # Maps node tree creation functions to the node tree 
    # name, such that we don't recreate node trees unnecessarily
    node_tree_names : dict[typing.Callable, str] = {}

    streamline_rgb = streamline_rgb_1_node_group(node_tree_names)
    node_tree_names[streamline_rgb_1_node_group] = streamline_rgb.name

    shader_nodetree = shader_nodetree_node_group(node_tree_names)
    node_tree_names[shader_nodetree_node_group] = shader_nodetree.name

