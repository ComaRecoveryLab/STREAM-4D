import bpy
import mathutils
import os
import typing


def streamline_geometry_1_node_group(node_tree_names: dict[typing.Callable, str]):
    """Initialize streamline_geometry node group"""
    streamline_geometry_1 = bpy.data.node_groups.new(type='GeometryNodeTree', name="streamline_geometry")

    streamline_geometry_1.color_tag = 'NONE'
    streamline_geometry_1.description = ""
    streamline_geometry_1.default_group_node_width = 140
    streamline_geometry_1.is_modifier = True
    streamline_geometry_1.show_modifier_manage_panel = True

    # streamline_geometry_1 interface

    # Socket Geometry
    geometry_socket = streamline_geometry_1.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    geometry_socket.attribute_domain = 'POINT'
    geometry_socket.default_input = 'VALUE'
    geometry_socket.structure_type = 'AUTO'

    # Socket Geometry
    geometry_socket_1 = streamline_geometry_1.interface.new_socket(name="Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry')
    geometry_socket_1.attribute_domain = 'POINT'
    geometry_socket_1.default_input = 'VALUE'
    geometry_socket_1.structure_type = 'AUTO'

    # Socket Geometry
    geometry_socket_2 = streamline_geometry_1.interface.new_socket(name="Geometry", in_out='INPUT', socket_type='NodeSocketGeometry')
    geometry_socket_2.attribute_domain = 'POINT'
    geometry_socket_2.default_input = 'VALUE'
    geometry_socket_2.structure_type = 'AUTO'

    # Initialize streamline_geometry_1 nodes

    # Node Resample Curve
    resample_curve = streamline_geometry_1.nodes.new("GeometryNodeResampleCurve")
    resample_curve.name = "Resample Curve"
    resample_curve.keep_last_segment = False
    # Selection
    resample_curve.inputs[1].default_value = True
    # Mode
    resample_curve.inputs[2].default_value = 'Count'
    # Count
    resample_curve.inputs[3].default_value = 60
    # Length
    resample_curve.inputs[4].default_value = 0.10000000149011612

    # Node Group Input.001
    group_input_001 = streamline_geometry_1.nodes.new("NodeGroupInput")
    group_input_001.name = "Group Input.001"

    # Node Blur Attribute
    blur_attribute = streamline_geometry_1.nodes.new("GeometryNodeBlurAttribute")
    blur_attribute.name = "Blur Attribute"
    blur_attribute.data_type = 'FLOAT_VECTOR'
    # Iterations
    blur_attribute.inputs[1].default_value = 2
    # Weight
    blur_attribute.inputs[2].default_value = 1.0

    # Node Position
    position = streamline_geometry_1.nodes.new("GeometryNodeInputPosition")
    position.name = "Position"

    # Node Set Position
    set_position = streamline_geometry_1.nodes.new("GeometryNodeSetPosition")
    set_position.name = "Set Position"
    # Selection
    set_position.inputs[1].default_value = True
    # Offset
    set_position.inputs[3].default_value = (0.0, 0.0, 0.0)

    # Node Curve Tangent
    curve_tangent = streamline_geometry_1.nodes.new("GeometryNodeInputTangent")
    curve_tangent.name = "Curve Tangent"

    # Node Curve Circle
    curve_circle = streamline_geometry_1.nodes.new("GeometryNodeCurvePrimitiveCircle")
    curve_circle.name = "Curve Circle"
    curve_circle.mode = 'RADIUS'
    # Resolution
    curve_circle.inputs[0].default_value = 2
    # Radius
    curve_circle.inputs[4].default_value = 0.07000000029802322

    # Node Store Named Attribute.002
    store_named_attribute_002 = streamline_geometry_1.nodes.new("GeometryNodeStoreNamedAttribute")
    store_named_attribute_002.name = "Store Named Attribute.002"
    store_named_attribute_002.data_type = 'FLOAT_VECTOR'
    store_named_attribute_002.domain = 'POINT'
    # Selection
    store_named_attribute_002.inputs[1].default_value = True
    # Name
    store_named_attribute_002.inputs[2].default_value = "curve_tangent"

    # Node Curve to Mesh
    curve_to_mesh = streamline_geometry_1.nodes.new("GeometryNodeCurveToMesh")
    curve_to_mesh.name = "Curve to Mesh"
    # Fill Caps
    curve_to_mesh.inputs[3].default_value = False

    # Node Set Material
    set_material = streamline_geometry_1.nodes.new("GeometryNodeSetMaterial")
    set_material.name = "Set Material"
    # Selection
    set_material.inputs[1].default_value = True
    if "streamlines" in bpy.data.materials:
        set_material.inputs[2].default_value = bpy.data.materials["streamlines"]

    # Node Group Output
    group_output = streamline_geometry_1.nodes.new("NodeGroupOutput")
    group_output.name = "Group Output"
    group_output.is_active_output = True

    # Node Named Attribute
    named_attribute = streamline_geometry_1.nodes.new("GeometryNodeInputNamedAttribute")
    named_attribute.name = "Named Attribute"
    named_attribute.data_type = 'FLOAT'
    # Name
    named_attribute.inputs[0].default_value = "radius"

    # Node Switch
    switch = streamline_geometry_1.nodes.new("GeometryNodeSwitch")
    switch.name = "Switch"
    switch.input_type = 'FLOAT'
    # False
    switch.inputs[1].default_value = 1.0

    # Set locations
    streamline_geometry_1.nodes["Resample Curve"].location = (-658.8970947265625, 100.7811050415039)
    streamline_geometry_1.nodes["Group Input.001"].location = (-875.0371704101562, 46.29372787475586)
    streamline_geometry_1.nodes["Blur Attribute"].location = (-657.7915649414062, -67.34431457519531)
    streamline_geometry_1.nodes["Position"].location = (-864.8965454101562, -124.46916961669922)
    streamline_geometry_1.nodes["Set Position"].location = (-458.1473693847656, 103.08470153808594)
    streamline_geometry_1.nodes["Curve Tangent"].location = (-289.2955322265625, -124.9947738647461)
    streamline_geometry_1.nodes["Curve Circle"].location = (-121.8592300415039, -187.1483612060547)
    streamline_geometry_1.nodes["Store Named Attribute.002"].location = (-122.89041900634766, 20.983989715576172)
    streamline_geometry_1.nodes["Curve to Mesh"].location = (112.38925170898438, 43.85099411010742)
    streamline_geometry_1.nodes["Set Material"].location = (313.8537902832031, 46.569602966308594)
    streamline_geometry_1.nodes["Group Output"].location = (505.39202880859375, 56.625091552734375)
    streamline_geometry_1.nodes["Named Attribute"].location = (87.38925170898438, 43.85099411010742)
    streamline_geometry_1.nodes["Switch"].location = (87.38925170898438, 43.85099411010742)

    # Set dimensions
    streamline_geometry_1.nodes["Resample Curve"].width  = 140.0
    streamline_geometry_1.nodes["Resample Curve"].height = 100.0

    streamline_geometry_1.nodes["Group Input.001"].width  = 140.0
    streamline_geometry_1.nodes["Group Input.001"].height = 100.0

    streamline_geometry_1.nodes["Blur Attribute"].width  = 140.0
    streamline_geometry_1.nodes["Blur Attribute"].height = 100.0

    streamline_geometry_1.nodes["Position"].width  = 140.0
    streamline_geometry_1.nodes["Position"].height = 100.0

    streamline_geometry_1.nodes["Set Position"].width  = 140.0
    streamline_geometry_1.nodes["Set Position"].height = 100.0

    streamline_geometry_1.nodes["Curve Tangent"].width  = 140.0
    streamline_geometry_1.nodes["Curve Tangent"].height = 100.0

    streamline_geometry_1.nodes["Curve Circle"].width  = 140.0
    streamline_geometry_1.nodes["Curve Circle"].height = 100.0

    streamline_geometry_1.nodes["Store Named Attribute.002"].width  = 140.0
    streamline_geometry_1.nodes["Store Named Attribute.002"].height = 100.0

    streamline_geometry_1.nodes["Curve to Mesh"].width  = 140.0
    streamline_geometry_1.nodes["Curve to Mesh"].height = 100.0

    streamline_geometry_1.nodes["Set Material"].width  = 140.0
    streamline_geometry_1.nodes["Set Material"].height = 100.0

    streamline_geometry_1.nodes["Group Output"].width  = 140.0
    streamline_geometry_1.nodes["Group Output"].height = 100.0

    streamline_geometry_1.nodes["Named Attribute"].width  = 140.0
    streamline_geometry_1.nodes["Named Attribute"].height = 100.0

    streamline_geometry_1.nodes["Switch"].width  = 140.0
    streamline_geometry_1.nodes["Switch"].height = 100.0


    # Initialize streamline_geometry_1 links

    # curve_to_mesh.Mesh -> set_material.Geometry
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Curve to Mesh"].outputs[0],
        streamline_geometry_1.nodes["Set Material"].inputs[0]
    )
    # store_named_attribute_002.Geometry -> curve_to_mesh.Curve
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Store Named Attribute.002"].outputs[0],
        streamline_geometry_1.nodes["Curve to Mesh"].inputs[0]
    )
    # curve_circle.Curve -> curve_to_mesh.Profile Curve
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Curve Circle"].outputs[0],
        streamline_geometry_1.nodes["Curve to Mesh"].inputs[1]
    )
    # set_material.Geometry -> group_output.Geometry
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Set Material"].outputs[0],
        streamline_geometry_1.nodes["Group Output"].inputs[0]
    )
    # group_input_001.Geometry -> resample_curve.Curve
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Group Input.001"].outputs[0],
        streamline_geometry_1.nodes["Resample Curve"].inputs[0]
    )
    # resample_curve.Curve -> set_position.Geometry
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Resample Curve"].outputs[0],
        streamline_geometry_1.nodes["Set Position"].inputs[0]
    )
    # position.Position -> blur_attribute.Value
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Position"].outputs[0],
        streamline_geometry_1.nodes["Blur Attribute"].inputs[0]
    )
    # blur_attribute.Value -> set_position.Position
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Blur Attribute"].outputs[0],
        streamline_geometry_1.nodes["Set Position"].inputs[2]
    )
    # curve_tangent.Tangent -> store_named_attribute_002.Value
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Curve Tangent"].outputs[0],
        streamline_geometry_1.nodes["Store Named Attribute.002"].inputs[3]
    )
    # set_position.Geometry -> store_named_attribute_002.Geometry
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Set Position"].outputs[0],
        streamline_geometry_1.nodes["Store Named Attribute.002"].inputs[0]
    )
    # named_attribute.Exists -> switch.Switch
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Named Attribute"].outputs[1],
        streamline_geometry_1.nodes["Switch"].inputs[0]
    )
    # named_attribute.Attribute -> switch.True
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Named Attribute"].outputs[0],
        streamline_geometry_1.nodes["Switch"].inputs[2]
    )
    # switch.Output -> curve_to_mesh.Scale
    streamline_geometry_1.links.new(
        streamline_geometry_1.nodes["Switch"].outputs[0],
        streamline_geometry_1.nodes["Curve to Mesh"].inputs[2]
    )

    return streamline_geometry_1


if __name__ == "__main__":
    # Maps node tree creation functions to the node tree 
    # name, such that we don't recreate node trees unnecessarily
    node_tree_names : dict[typing.Callable, str] = {}

    streamline_geometry = streamline_geometry_1_node_group(node_tree_names)
    node_tree_names[streamline_geometry_1_node_group] = streamline_geometry.name

