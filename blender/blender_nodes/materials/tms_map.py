import bpy
import mathutils
import os
import typing


tms_map = bpy.data.materials.new(name = "tms_map")
if bpy.app.version < (5, 0, 0):
    tms_map.use_nodes = True


tms_map.alpha_threshold = 0.5
tms_map.line_priority = 0
tms_map.max_vertex_displacement = 0.0
tms_map.metallic = 0.0
tms_map.paint_active_slot = 0
tms_map.paint_clone_slot = 0
tms_map.pass_index = 0
tms_map.refraction_depth = 0.0
tms_map.roughness = 0.4000000059604645
tms_map.show_transparent_back = True
tms_map.specular_intensity = 0.5
tms_map.use_backface_culling = False
tms_map.use_backface_culling_lightprobe_volume = False
tms_map.use_backface_culling_shadow = False
tms_map.use_preview_world = False
tms_map.use_raytrace_refraction = False
tms_map.use_screen_refraction = False
tms_map.use_sss_translucency = False
tms_map.use_thickness_from_shadow = False
tms_map.use_transparency_overlap = True
tms_map.use_transparent_shadow = False
tms_map.blend_method = 'HASHED'
tms_map.displacement_method = 'BUMP'
tms_map.preview_render_type = 'SPHERE'
tms_map.surface_render_method = 'DITHERED'
tms_map.thickness_mode = 'SPHERE'
tms_map.volume_intersection_method = 'FAST'
tms_map.specular_color = (1.0, 1.0, 1.0)
tms_map.diffuse_color = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
tms_map.line_color = (0.0, 0.0, 0.0, 0.0)

def shader_nodetree_node_group(node_tree_names: dict[typing.Callable, str]):
    """Initialize Shader Nodetree node group"""
    shader_nodetree = tms_map.node_tree

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
    attribute.attribute_name = "intensity"
    attribute.attribute_type = 'GEOMETRY'

    # Node Math.001
    math_001 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_001.name = "Math.001"
    math_001.operation = 'ADD'
    math_001.use_clamp = False
    # Value_001
    math_001.inputs[1].default_value = 1.0

    # Node Math.002
    math_002 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_002.name = "Math.002"
    math_002.operation = 'MULTIPLY'
    math_002.use_clamp = False
    # Value_001
    math_002.inputs[1].default_value = 2.0

    # Node Math.003
    math_003 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_003.name = "Math.003"
    math_003.operation = 'MULTIPLY'
    math_003.use_clamp = False

    # Node Math
    math = shader_nodetree.nodes.new("ShaderNodeMath")
    math.name = "Math"
    math.operation = 'GREATER_THAN'
    math.use_clamp = False
    # Value_001
    math.inputs[1].default_value = 0.0

    # Node Principled BSDF
    principled_bsdf = shader_nodetree.nodes.new("ShaderNodeBsdfPrincipled")
    principled_bsdf.name = "Principled BSDF"
    principled_bsdf.distribution = 'GGX'
    principled_bsdf.subsurface_method = 'RANDOM_WALK_SKIN'
    # Base Color
    principled_bsdf.inputs[0].default_value = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
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

    # Node Mix Shader.001
    mix_shader_001 = shader_nodetree.nodes.new("ShaderNodeMixShader")
    mix_shader_001.name = "Mix Shader.001"

    # Node Material Output
    material_output = shader_nodetree.nodes.new("ShaderNodeOutputMaterial")
    material_output.name = "Material Output"
    material_output.is_active_output = True
    material_output.target = 'ALL'
    # Displacement
    material_output.inputs[2].default_value = (0.0, 0.0, 0.0)
    # Thickness
    material_output.inputs[3].default_value = 0.0

    # Node tms_map
    tms_map_1 = shader_nodetree.nodes.new("ShaderNodeBsdfPrincipled")
    tms_map_1.name = "tms_map"
    tms_map_1.distribution = 'GGX'
    tms_map_1.subsurface_method = 'RANDOM_WALK_SKIN'
    # Base Color
    tms_map_1.inputs[0].default_value = (0.800000011920929, 0.800000011920929, 0.800000011920929, 1.0)
    # Metallic
    tms_map_1.inputs[1].default_value = 0.0
    # Roughness
    tms_map_1.inputs[2].default_value = 0.5
    # IOR
    tms_map_1.inputs[3].default_value = 1.4500000476837158
    # Alpha
    tms_map_1.inputs[4].default_value = 0.10000000149011612
    # Normal
    tms_map_1.inputs[5].default_value = (0.0, 0.0, 0.0)
    # Diffuse Roughness
    tms_map_1.inputs[7].default_value = 0.0
    # Subsurface Weight
    tms_map_1.inputs[8].default_value = 0.0
    # Subsurface Radius
    tms_map_1.inputs[9].default_value = (1.0, 0.20000000298023224, 0.10000000149011612)
    # Subsurface Scale
    tms_map_1.inputs[10].default_value = 0.05000000074505806
    # Subsurface IOR
    tms_map_1.inputs[11].default_value = 1.399999976158142
    # Subsurface Anisotropy
    tms_map_1.inputs[12].default_value = 0.0
    # Specular IOR Level
    tms_map_1.inputs[13].default_value = 0.5
    # Specular Tint
    tms_map_1.inputs[14].default_value = (1.0, 1.0, 1.0, 1.0)
    # Anisotropic
    tms_map_1.inputs[15].default_value = 0.0
    # Anisotropic Rotation
    tms_map_1.inputs[16].default_value = 0.0
    # Tangent
    tms_map_1.inputs[17].default_value = (0.0, 0.0, 0.0)
    # Transmission Weight
    tms_map_1.inputs[18].default_value = 0.0
    # Coat Weight
    tms_map_1.inputs[19].default_value = 0.0
    # Coat Roughness
    tms_map_1.inputs[20].default_value = 0.029999999329447746
    # Coat IOR
    tms_map_1.inputs[21].default_value = 1.5
    # Coat Tint
    tms_map_1.inputs[22].default_value = (1.0, 1.0, 1.0, 1.0)
    # Coat Normal
    tms_map_1.inputs[23].default_value = (0.0, 0.0, 0.0)
    # Sheen Weight
    tms_map_1.inputs[24].default_value = 0.0
    # Sheen Roughness
    tms_map_1.inputs[25].default_value = 0.5
    # Sheen Tint
    tms_map_1.inputs[26].default_value = (1.0, 1.0, 1.0, 1.0)
    # Emission Color
    tms_map_1.inputs[27].default_value = (0.004814247600734234, 0.004814247600734234, 0.004814247600734234, 1.0)
    # Emission Strength
    tms_map_1.inputs[28].default_value = 0.0
    # Thin Film Thickness
    tms_map_1.inputs[29].default_value = 0.0
    # Thin Film IOR
    tms_map_1.inputs[30].default_value = 1.3300000429153442

    # Node Math.004
    math_004 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_004.name = "Math.004"
    math_004.operation = 'SUBTRACT'
    math_004.use_clamp = False

    # Node Math.005
    math_005 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_005.name = "Math.005"
    math_005.operation = 'DIVIDE'
    math_005.use_clamp = False

    # Node Math.006
    math_006 = shader_nodetree.nodes.new("ShaderNodeMath")
    math_006.name = "Math.006"
    math_006.operation = 'SUBTRACT'
    math_006.use_clamp = False
    # Value
    math_006.inputs[0].default_value = 1.0

    # Node Value
    value = shader_nodetree.nodes.new("ShaderNodeValue")
    value.name = "Value"

    value.outputs[0].default_value = 0.0
    # Node Color Ramp.002
    color_ramp_002 = shader_nodetree.nodes.new("ShaderNodeValToRGB")
    color_ramp_002.name = "Color Ramp.002"
    color_ramp_002.color_ramp.color_mode = 'RGB'
    color_ramp_002.color_ramp.hue_interpolation = 'NEAR'
    color_ramp_002.color_ramp.interpolation = 'LINEAR'

    # Initialize color ramp elements
    color_ramp_002.color_ramp.elements.remove(color_ramp_002.color_ramp.elements[0])
    color_ramp_002_cre_0 = color_ramp_002.color_ramp.elements[0]
    color_ramp_002_cre_0.position = 0.0
    color_ramp_002_cre_0.alpha = 0.10000000149011612
    color_ramp_002_cre_0.color = (0.0, 0.0, 0.0, 0.10000000149011612)

    color_ramp_002_cre_1 = color_ramp_002.color_ramp.elements.new(0.20000000298023224)
    color_ramp_002_cre_1.alpha = 0.30000001192092896
    color_ramp_002_cre_1.color = (0.22277499735355377, 0.0, 0.0022460720501840115, 0.30000001192092896)

    color_ramp_002_cre_2 = color_ramp_002.color_ramp.elements.new(0.4000000059604645)
    color_ramp_002_cre_2.alpha = 0.5
    color_ramp_002_cre_2.color = (1.0, 0.0, 0.007906528189778328, 0.5)

    color_ramp_002_cre_3 = color_ramp_002.color_ramp.elements.new(0.6000000238418579)
    color_ramp_002_cre_3.alpha = 0.75
    color_ramp_002_cre_3.color = (1.0, 0.2656368911266327, 0.0, 0.75)

    color_ramp_002_cre_4 = color_ramp_002.color_ramp.elements.new(0.800000011920929)
    color_ramp_002_cre_4.alpha = 0.800000011920929
    color_ramp_002_cre_4.color = (1.0, 0.6985067129135132, 0.0, 0.800000011920929)

    color_ramp_002_cre_5 = color_ramp_002.color_ramp.elements.new(1.0)
    color_ramp_002_cre_5.alpha = 0.8999999761581421
    color_ramp_002_cre_5.color = (1.0, 1.0, 1.0, 0.8999999761581421)


    # Set locations
    shader_nodetree.nodes["Attribute"].location = (-1429.58154296875, 428.6315002441406)
    shader_nodetree.nodes["Math.001"].location = (-10.648505210876465, 200.165771484375)
    shader_nodetree.nodes["Math.002"].location = (-186.53375244140625, 199.69541931152344)
    shader_nodetree.nodes["Math.003"].location = (-350.9078063964844, 198.68142700195312)
    shader_nodetree.nodes["Math"].location = (-613.6688842773438, 35.63408660888672)
    shader_nodetree.nodes["Principled BSDF"].location = (250.03680419921875, 550.6865844726562)
    shader_nodetree.nodes["Mix Shader.001"].location = (673.8995361328125, 57.601356506347656)
    shader_nodetree.nodes["Material Output"].location = (897.8855590820312, 83.10791778564453)
    shader_nodetree.nodes["tms_map"].location = (132.451904296875, -101.18753051757812)
    shader_nodetree.nodes["Math.004"].location = (-1140.529296875, 454.2554626464844)
    shader_nodetree.nodes["Math.005"].location = (-839.5078125, 340.0110168457031)
    shader_nodetree.nodes["Math.006"].location = (-1104.8441162109375, 243.43069458007812)
    shader_nodetree.nodes["Value"].location = (-1397.3685302734375, 152.94642639160156)
    shader_nodetree.nodes["Color Ramp.002"].location = (-278.6380310058594, 490.549072265625)

    # Set dimensions
    shader_nodetree.nodes["Attribute"].width  = 140.0
    shader_nodetree.nodes["Attribute"].height = 100.0

    shader_nodetree.nodes["Math.001"].width  = 140.0
    shader_nodetree.nodes["Math.001"].height = 100.0

    shader_nodetree.nodes["Math.002"].width  = 140.0
    shader_nodetree.nodes["Math.002"].height = 100.0

    shader_nodetree.nodes["Math.003"].width  = 140.0
    shader_nodetree.nodes["Math.003"].height = 100.0

    shader_nodetree.nodes["Math"].width  = 140.0
    shader_nodetree.nodes["Math"].height = 100.0

    shader_nodetree.nodes["Principled BSDF"].width  = 240.0
    shader_nodetree.nodes["Principled BSDF"].height = 100.0

    shader_nodetree.nodes["Mix Shader.001"].width  = 140.0
    shader_nodetree.nodes["Mix Shader.001"].height = 100.0

    shader_nodetree.nodes["Material Output"].width  = 140.0
    shader_nodetree.nodes["Material Output"].height = 100.0

    shader_nodetree.nodes["tms_map"].width  = 240.0
    shader_nodetree.nodes["tms_map"].height = 100.0

    shader_nodetree.nodes["Math.004"].width  = 140.0
    shader_nodetree.nodes["Math.004"].height = 100.0

    shader_nodetree.nodes["Math.005"].width  = 140.0
    shader_nodetree.nodes["Math.005"].height = 100.0

    shader_nodetree.nodes["Math.006"].width  = 140.0
    shader_nodetree.nodes["Math.006"].height = 100.0

    shader_nodetree.nodes["Value"].width  = 140.0
    shader_nodetree.nodes["Value"].height = 100.0

    shader_nodetree.nodes["Color Ramp.002"].width  = 240.0
    shader_nodetree.nodes["Color Ramp.002"].height = 100.0


    # Initialize shader_nodetree links

    # math.Value -> math_003.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math"].outputs[0],
        shader_nodetree.nodes["Math.003"].inputs[1]
    )
    # math_002.Value -> math_001.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.002"].outputs[0],
        shader_nodetree.nodes["Math.001"].inputs[0]
    )
    # math_003.Value -> math_002.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.003"].outputs[0],
        shader_nodetree.nodes["Math.002"].inputs[0]
    )
    # math_001.Value -> principled_bsdf.Emission Strength
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.001"].outputs[0],
        shader_nodetree.nodes["Principled BSDF"].inputs[28]
    )
    # math.Value -> mix_shader_001.Factor
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math"].outputs[0],
        shader_nodetree.nodes["Mix Shader.001"].inputs[0]
    )
    # mix_shader_001.Shader -> material_output.Surface
    shader_nodetree.links.new(
        shader_nodetree.nodes["Mix Shader.001"].outputs[0],
        shader_nodetree.nodes["Material Output"].inputs[0]
    )
    # tms_map_1.BSDF -> mix_shader_001.Shader
    shader_nodetree.links.new(
        shader_nodetree.nodes["tms_map"].outputs[0],
        shader_nodetree.nodes["Mix Shader.001"].inputs[1]
    )
    # principled_bsdf.BSDF -> mix_shader_001.Shader
    shader_nodetree.links.new(
        shader_nodetree.nodes["Principled BSDF"].outputs[0],
        shader_nodetree.nodes["Mix Shader.001"].inputs[2]
    )
    # math_004.Value -> math_005.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.004"].outputs[0],
        shader_nodetree.nodes["Math.005"].inputs[0]
    )
    # value.Value -> math_004.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Value"].outputs[0],
        shader_nodetree.nodes["Math.004"].inputs[1]
    )
    # attribute.Factor -> math_004.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Attribute"].outputs[2],
        shader_nodetree.nodes["Math.004"].inputs[0]
    )
    # value.Value -> math_006.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Value"].outputs[0],
        shader_nodetree.nodes["Math.006"].inputs[1]
    )
    # math_006.Value -> math_005.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.006"].outputs[0],
        shader_nodetree.nodes["Math.005"].inputs[1]
    )
    # math_005.Value -> math.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.005"].outputs[0],
        shader_nodetree.nodes["Math"].inputs[0]
    )
    # math_005.Value -> math_003.Value
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.005"].outputs[0],
        shader_nodetree.nodes["Math.003"].inputs[0]
    )
    # math_005.Value -> color_ramp_002.Factor
    shader_nodetree.links.new(
        shader_nodetree.nodes["Math.005"].outputs[0],
        shader_nodetree.nodes["Color Ramp.002"].inputs[0]
    )
    # color_ramp_002.Color -> principled_bsdf.Emission Color
    shader_nodetree.links.new(
        shader_nodetree.nodes["Color Ramp.002"].outputs[0],
        shader_nodetree.nodes["Principled BSDF"].inputs[27]
    )
    # color_ramp_002.Alpha -> principled_bsdf.Alpha
    shader_nodetree.links.new(
        shader_nodetree.nodes["Color Ramp.002"].outputs[1],
        shader_nodetree.nodes["Principled BSDF"].inputs[4]
    )

    return shader_nodetree


if __name__ == "__main__":
    # Maps node tree creation functions to the node tree 
    # name, such that we don't recreate node trees unnecessarily
    node_tree_names : dict[typing.Callable, str] = {}

    shader_nodetree = shader_nodetree_node_group(node_tree_names)
    node_tree_names[shader_nodetree_node_group] = shader_nodetree.name

