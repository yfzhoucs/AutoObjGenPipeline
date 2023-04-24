import bpy
import bmesh
import os
import sys

bpy.context.preferences.view.show_splash = False




def add_texture_to_the_plane(plane, img_path):

    # Load the image
    img = bpy.data.images.load(img_path)

    # Create a new material
    mat = bpy.data.materials.new("Texture")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear existing nodes
    nodes.clear()

    # Create texture node
    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = img
    # tex_node.location = (-300, 0)

    # Create Principled BSDF node
    principled_node = nodes.new(type="ShaderNodeBsdfPrincipled")
    # principled_node.location = (0, 0)

    # Create Material Output node
    output_node = nodes.new(type="ShaderNodeOutputMaterial")
    # output_node.location = (300, 0)

    # Link the nodes
    links.new(tex_node.outputs[0], principled_node.inputs['Base Color'])
    links.new(principled_node.outputs['BSDF'], output_node.inputs['Surface'])

    # Assign the material to the plane object
    plane.data.materials.append(mat)

# # Deselect all
# bpy.ops.object.select_all(action='DESELECT')


def export_obj_from_depth_and_rgb(depth_file, rgb_file, stl_folder, obj_folder, W, H):
    object_name = depth_file.split(r'/')[-1]
    object_name = object_name.split(r'.')[0]

    # Select all
    bpy.ops.object.select_all(action='SELECT')

    # Delete selected
    bpy.ops.object.delete(use_global=True, confirm=False)

    # Create a plane
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=W, y_subdivisions=H)

    # Select current plane
    plane = bpy.context.active_object

    # Add texture to the plane
    add_texture_to_the_plane(plane, rgb_file)
    print(f'{object_name} added rgb texture')

    # Define depth image as texture
    heightTex = bpy.data.textures.new('Texture name', type = 'IMAGE')
    heightTex.image = bpy.data.images.load(depth_file)

    # Create a displacement modifier on the plane object
    plane = bpy.context.active_object
    dispMod = plane.modifiers.new("Displace", type='DISPLACE')
    dispMod.texture = heightTex
    dispMod.strength = 0.8
    dispMod.mid_level = 0.
    bpy.ops.object.modifier_apply(modifier='Displace')
    print(f'{object_name} added displace')

    # Create a second plane
    bpy.ops.mesh.primitive_plane_add()
    plane2 = bpy.data.objects['Plane']
    plane2.location.z = 0.10

    # print(list(bpy.data.objects))

    # Select the previous plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Grid"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Grid"]

    # Create boolean modifier
    boolMod = plane.modifiers.new('boolean', type="BOOLEAN")
    boolMod.object = bpy.data.objects['Plane']

    # Apply the 2 modifiers
    bpy.ops.object.modifier_apply(modifier='boolean')
    print(f'{object_name} cutted half')


    # Create mirror modifier
    plane2.location.z = 0.
    mirrorMod = plane.modifiers.new('mirror', type='MIRROR')
    mirrorMod.mirror_object = bpy.data.objects['Plane']
    mirrorMod.use_axis = [False, False, True]
    bpy.ops.object.modifier_apply(modifier="mirror")

    # Delete the previous plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Plane"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Plane"]
    bpy.ops.object.delete(use_global=True, confirm=False)
    print(f'{object_name} mirrored half')



    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Grid"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Grid"]
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.fill_holes(1000)


    context = bpy.context
    ob = context.edit_object
    me = ob.data
    bm = bmesh.from_edit_mesh(me)
    for f in bm.faces:
        f.select = f.calc_area() > 0.5
        
    bmesh.update_edit_mesh(me)
    bpy.ops.mesh.bridge_edge_loops()
    bpy.ops.object.editmode_toggle()

    # # Bridge edge loops to connect the 2 halves
    # ob = bpy.data.objects["Grid"]
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects["Grid"].select_set(True)
    # bpy.context.view_layer.objects.active = bpy.data.objects["Grid"]
    # if ob and ob.type == 'MESH':
    #     me = ob.data
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.select_all(action='DESELECT')
    #     bpy.ops.object.editmode_toggle()
    #     for v in me.vertices:
    #         if v.co[2] > -0.11 and v.co[2] < 0.11:
    #             v.select = True
    #     bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.bridge_edge_loops()
    # # Enter object mode
    # bpy.ops.object.mode_set(mode="OBJECT")
    # print(f'{object_name} connected 2 halves')

    # ob = bpy.data.objects["Grid"]
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects["Grid"].select_set(True)
    # bpy.context.view_layer.objects.active = bpy.data.objects["Grid"]
    # me = ob.data
    # bpy.ops.object.mode_set(mode='EDIT')
    # bpy.ops.mesh.select_all(action='DESELECT')
    # bm = bmesh.from_edit_mesh(me)

    

    # ngons = [f for f in bm.faces if len(f.verts) > 4]
    # if ngons:
    #     for v in ngons.pop(0).verts:
    #         v.select = True
    # # bpy.ops.object.editmode_toggle()
    # bmesh.update_edit_mesh(me)


    # bpy.ops.mesh.bridge_edge_loops()

        
    # Laplacian Smoothing
    bpy.context.view_layer.objects.active = bpy.data.objects['Grid']
    bpy.data.objects['Grid'].select_set(True)
    bpy.ops.object.modifier_add(type='LAPLACIANSMOOTH')
    bpy.context.object.modifiers['LaplacianSmooth'].lambda_factor = 10
    bpy.ops.object.modifier_apply(modifier="LaplacianSmooth")
    print(f'{object_name} added Laplacian Smoothing')


    # Decimate
    bpy.context.view_layer.objects.active = bpy.data.objects['Grid']
    bpy.data.objects['Grid'].select_set(True)
    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers['Decimate'].decimate_type = 'COLLAPSE'
    bpy.context.object.modifiers['Decimate'].ratio = 0.1
    bpy.ops.object.modifier_apply(modifier="Decimate")
    print(f'{object_name} added Decimate')

    # Select the previous plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Grid"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Grid"]

    # Select the object
    plane_name = "Grid"  # Replace with the name of your 3D object in Blender
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[plane_name].select_set(True)


    # Export the object as an STL file
    output_path = os.path.join(stl_folder, object_name + '.stl')
    print(f'{object_name} saving stl at {output_path}')
    bpy.ops.export_mesh.stl(filepath=output_path, use_selection=True)
    print(f'{object_name} saved stl at {output_path}')

    # Set the output path for the OBJ and MTL files
    output_obj_path = os.path.join(obj_folder, object_name + ".obj")
    output_mtl_path = os.path.join(obj_folder, object_name + ".mtl")
    print(f'{object_name} saving obj at {output_obj_path}')

    # Export the object as an OBJ file
    # bpy.ops.export_scene.obj(filepath=output_obj_path, use_selection=True, use_materials=True, use_triangles=True, path_mode='AUTO')
    bpy.ops.export_scene.obj(filepath=output_obj_path, use_selection=True, path_mode='COPY')
    print(f'{object_name} saved obj at {output_obj_path}')

    # # # Save textures as PNG files
    # # for mat in bpy.data.materials:
    # #     if mat.use_nodes:
    # #         print(1, mat)
    # #         for node in mat.node_tree.nodes:
    # #             print(2, node)
    # #             print(3, node.type)
    # #             if node.type == 'TEX_IMAGE':
    # #                 img = node.image
    # #                 print(4, img)
    # #                 img_path = os.path.join(output_folder, img.name)
    # #                 print(5, img_path)
    # #                 img.filepath_raw = img_path
    # #                 img.file_format = 'PNG'
    # #                 img.save()






    # # Exit
    # bpy.ops.wm.quit_blender()


export_obj_from_depth_and_rgb(sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], int(sys.argv[8]), int(sys.argv[9]))


# # Exit
# bpy.ops.wm.quit_blender()