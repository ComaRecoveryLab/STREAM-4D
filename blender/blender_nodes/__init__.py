def setup_scene_nodes():
    """
    Creates all materials and geometry node groups needed for the STREAM-4D
    scene programmatically.

    Call order matters:
      1. wireframe and tms_map materials (no dependencies)
      2. streamline_RGB node group, then streamlines material (depends on streamline_RGB)
      3. streamline_geometry and white_matter node groups (depend on their respective materials)
    """
    from blender_nodes.materials import wireframe as wireframe_mod
    from blender_nodes.materials import tms_map as tms_map_mod
    from blender_nodes.materials import streamlines as streamlines_mod
    from blender_nodes.geometry import streamline_geometry as sg_mod
    from blender_nodes.geometry import white_matter as wm_mod

    node_tree_names = {}

    wireframe_mod.shader_nodetree_node_group(node_tree_names)
    tms_map_mod.shader_nodetree_node_group(node_tree_names)

    streamline_rgb = streamlines_mod.streamline_rgb_1_node_group(node_tree_names)
    node_tree_names[streamlines_mod.streamline_rgb_1_node_group] = streamline_rgb.name
    streamlines_mod.shader_nodetree_node_group(node_tree_names)

    sg_mod.streamline_geometry_1_node_group(node_tree_names)
    wm_mod.white_matter_1_node_group(node_tree_names)
