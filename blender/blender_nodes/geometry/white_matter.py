import bpy
import mathutils
import os
import typing


def white_matter_1_node_group(node_tree_names: dict[typing.Callable, str]):
    """Initialize white_matter node group"""
    white_matter_1 = bpy.data.node_groups.new(type='GeometryNodeTree', name="white_matter")

    white_matter_1.color_tag = 'NONE'
    white_matter_1.description = ""
    white_matter_1.default_group_node_width = 140
    white_matter_1.is_modifier = True
    white_matter_1.show_modifier_manage_panel = True

    # white_matter_1 interface

    # Socket Geometry
    geometry_socket = white_matter_1.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    geometry_socket.attribute_domain = 'POINT'
    geometry_socket.default_input = 'VALUE'
    geometry_socket.structure_type = 'AUTO'

    # Socket Geometry
    geometry_socket_1 = white_matter_1.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    geometry_socket_1.attribute_domain = 'POINT'
    geometry_socket_1.default_input = 'VALUE'
    geometry_socket_1.structure_type = 'AUTO'

    # Initialize white_matter_1 nodes

    # Node Group Input
    group_input = white_matter_1.nodes.new("NodeGroupInput")
    group_input.name = "Group Input"

    # Node Mesh to Curve
    mesh_to_curve = white_matter_1.nodes.new("GeometryNodeMeshToCurve")
    mesh_to_curve.name = "Mesh to Curve"
    mesh_to_curve.mode = 'EDGES'
    # Selection
    mesh_to_curve.inputs[1].default_value = True

    # Node Curve to Mesh
    curve_to_mesh = white_matter_1.nodes.new("GeometryNodeCurveToMesh")
    curve_to_mesh.name = "Curve to Mesh"
    # Fill Caps
    curve_to_mesh.inputs[3].default_value = False

    # Node Curve Circle
    curve_circle = white_matter_1.nodes.new("GeometryNodeCurvePrimitiveCircle")
    curve_circle.name = "Curve Circle"
    curve_circle.mode = 'RADIUS'
    # Resolution
    curve_circle.inputs[0].default_value = 6
    # Radius
    curve_circle.inputs[4].default_value = 0.0024999999441206455

    # Node Group Output
    group_output = white_matter_1.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # Node Set Material
    set_material = white_matter_1.nodes.new("GeometryNodeSetMaterial")
    set_material.name = "Set Material"
    # Selection
    set_material.inputs[1].default_value = True
    if "wireframe" in bpy.data.materials:
        set_material.inputs[2].default_value = bpy.data.materials["wireframe"]

    # Node Named Attribute
    named_attribute = white_matter_1.nodes.new("GeometryNodeInputNamedAttribute")
    named_attribute.name = "Named Attribute"
    named_attribute.data_type = 'FLOAT'
    # Name
    named_attribute.inputs[0].default_value = "radius"

    # Node Switch
    switch = white_matter_1.nodes.new("GeometryNodeSwitch")
    switch.name = "Switch"
    switch.input_type = 'FLOAT'
    # False
    switch.inputs[1].default_value = 1.0

    # Set locations
    white_matter_1.nodes["Group Input"].location = (-340.0, 0.0)
    white_matter_1.nodes["Mesh to Curve"].location = (-120.00000762939453, 25.905776977539062)
    white_matter_1.nodes["Curve to Mesh"].location = (177.47213745117188, 18.835580825805664)
    white_matter_1.nodes["Curve Circle"].location = (-104.0994873046875, -129.63827514648438)
    white_matter_1.nodes["Group Output"].location = (623.2998657226562, 12.12031364440918)
    white_matter_1.nodes["Set Material"].location = (403.5368347167969, 10.755356788635254)
    white_matter_1.nodes["Named Attribute"].location = (152.47213745117188, 18.835580825805664)
    white_matter_1.nodes["Switch"].location = (152.47213745117188, 18.835580825805664)

    # Set dimensions
    white_matter_1.nodes["Group Input"].width  = 140.0
    white_matter_1.nodes["Group Input"].height = 100.0

    white_matter_1.nodes["Mesh to Curve"].width  = 140.0
    white_matter_1.nodes["Mesh to Curve"].height = 100.0

    white_matter_1.nodes["Curve to Mesh"].width  = 140.0
    white_matter_1.nodes["Curve to Mesh"].height = 100.0

    white_matter_1.nodes["Curve Circle"].width  = 140.0
    white_matter_1.nodes["Curve Circle"].height = 100.0

    white_matter_1.nodes["Group Output"].width  = 140.0
    white_matter_1.nodes["Group Output"].height = 100.0

    white_matter_1.nodes["Set Material"].width  = 140.0
    white_matter_1.nodes["Set Material"].height = 100.0

    white_matter_1.nodes["Named Attribute"].width  = 140.0
    white_matter_1.nodes["Named Attribute"].height = 100.0

    white_matter_1.nodes["Switch"].width  = 140.0
    white_matter_1.nodes["Switch"].height = 100.0


    # Initialize white_matter_1 links

    # set_material.Geometry -> group_output.Geometry
    white_matter_1.links.new(
        white_matter_1.nodes["Set Material"].outputs[0],
        white_matter_1.nodes["Group Output"].inputs[0]
    )
    # group_input.Geometry -> mesh_to_curve.Mesh
    white_matter_1.links.new(
        white_matter_1.nodes["Group Input"].outputs[0],
        white_matter_1.nodes["Mesh to Curve"].inputs[0]
    )
    # mesh_to_curve.Curve -> curve_to_mesh.Curve
    white_matter_1.links.new(
        white_matter_1.nodes["Mesh to Curve"].outputs[0],
        white_matter_1.nodes["Curve to Mesh"].inputs[0]
    )
    # curve_circle.Curve -> curve_to_mesh.Profile Curve
    white_matter_1.links.new(
        white_matter_1.nodes["Curve Circle"].outputs[0],
        white_matter_1.nodes["Curve to Mesh"].inputs[1]
    )
    # curve_to_mesh.Mesh -> set_material.Geometry
    white_matter_1.links.new(
        white_matter_1.nodes["Curve to Mesh"].outputs[0],
        white_matter_1.nodes["Set Material"].inputs[0]
    )
    # named_attribute.Exists -> switch.Switch
    white_matter_1.links.new(
        white_matter_1.nodes["Named Attribute"].outputs[1],
        white_matter_1.nodes["Switch"].inputs[0]
    )
    # named_attribute.Attribute -> switch.True
    white_matter_1.links.new(
        white_matter_1.nodes["Named Attribute"].outputs[0],
        white_matter_1.nodes["Switch"].inputs[2]
    )
    # switch.Output -> curve_to_mesh.Scale
    white_matter_1.links.new(
        white_matter_1.nodes["Switch"].outputs[0],
        white_matter_1.nodes["Curve to Mesh"].inputs[2]
    )

    return white_matter_1


if __name__ == "__main__":
    # Maps node tree creation functions to the node tree 
    # name, such that we don't recreate node trees unnecessarily
    node_tree_names : dict[typing.Callable, str] = {}

    white_matter = white_matter_1_node_group(node_tree_names)
    node_tree_names[white_matter_1_node_group] = white_matter.name

