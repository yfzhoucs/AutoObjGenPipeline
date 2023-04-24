import bpy
import os

bpy.context.preferences.view.show_splash = False
print(bpy.data.objects)




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


def export_obj_from_depth_and_rgb(depth_file, rgb_file, stl_folder, obj_folder):
    object_name = depth_file.split(r'/')[-1]
    object_name = object_name.split(r'.')[0]

    # Select all
    bpy.ops.object.select_all(action='SELECT')

    # Delete selected
    bpy.ops.object.delete(use_global=True, confirm=False)

    # Create a plane
    bpy.ops.mesh.primitive_plane_add()

    # Enter edit mode
    bpy.ops.object.mode_set(mode='EDIT')

    # Subdivide the plane
    bpy.ops.mesh.subdivide(number_cuts=100)

    # Enter object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Select current plane
    plane = bpy.context.active_object

    # Add texture to the plane
    add_texture_to_the_plane(plane, rgb_file)

    # Define depth image as texture
    heightTex = bpy.data.textures.new('Texture name', type = 'IMAGE')
    heightTex.image = bpy.data.images.load(depth_file)

    # Create a displacement modifier on the plane object
    plane = bpy.context.active_object
    dispMod = plane.modifiers.new("Displace", type='DISPLACE')
    dispMod.texture = heightTex
    dispMod.strength = 1
    dispMod.mid_level = 0.
    bpy.ops.object.modifier_apply(modifier='Displace')

    # Create a second plane
    bpy.ops.mesh.primitive_plane_add()
    plane2 = bpy.data.objects['Plane.001']
    plane2.location.z = 0.07

    # print(list(bpy.data.objects))

    # Select the previous plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Plane"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Plane"]

    # Create boolean modifier
    boolMod = plane.modifiers.new('boolean', type="BOOLEAN")
    boolMod.object = bpy.data.objects['Plane.001']

    # Apply the 2 modifiers
    bpy.ops.object.modifier_apply(modifier='boolean')


    # Create mirror modifier
    plane2.location.z = 0.
    mirrorMod = plane.modifiers.new('mirror', type='MIRROR')
    mirrorMod.mirror_object = bpy.data.objects['Plane.001']
    mirrorMod.use_axis = [False, False, True]
    bpy.ops.object.modifier_apply(modifier="mirror")

    # Delete the previous plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Plane.001"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Plane.001"]
    bpy.ops.object.delete(use_global=True, confirm=False)

    # Bridge edge loops to connect the 2 halves
    ob = bpy.data.objects["Plane"]
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Plane"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Plane"]
    if ob and ob.type == 'MESH':
        me = ob.data
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.select_all(action='DESELECT')
        bpy.ops.object.editmode_toggle()
        for v in me.vertices:
            if v.co[2] > -0.11 and v.co[2] < 0.11:
                v.select = True
        bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.bridge_edge_loops()
    # Enter object mode
    bpy.ops.object.mode_set(mode="OBJECT")





    # # Add a new plane for cutting off the bottom
    # bpy.ops.mesh.primitive_plane_add()
    # plane2 = bpy.data.objects['Plane.001']
    # plane2.rotation_euler = (-1.57, 0, 0)
    # plane2.location.y = -0.9

    # # Select the previous plane
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects["Plane"].select_set(True)
    # bpy.context.view_layer.objects.active = bpy.data.objects["Plane"]

    # # Create boolean modifier
    # boolMod = plane.modifiers.new('boolean2', type="BOOLEAN")
    # boolMod.object = bpy.data.objects['Plane.001']

    # # Apply the 2 modifiers
    # bpy.ops.object.modifier_apply(modifier='boolean2')

    # # Delete the previous plane
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects["Plane.001"].select_set(True)
    # bpy.context.view_layer.objects.active = bpy.data.objects["Plane.001"]
    # bpy.ops.object.delete(use_global=True, confirm=False)





    # # Fill the cut face
    # ob = bpy.data.objects["Plane"]
    # bpy.ops.object.select_all(action='DESELECT')
    # bpy.data.objects["Plane"].select_set(True)
    # bpy.context.view_layer.objects.active = bpy.data.objects["Plane"]
    # if ob and ob.type == 'MESH':
    #     me = ob.data
    #     bpy.ops.object.mode_set(mode='EDIT')
    #     bpy.ops.mesh.select_all(action='DESELECT')
    #     bpy.ops.object.editmode_toggle()
    #     for v in me.vertices:
    #         if v.co[1] < -0.899:
    #             v.select = True
    #         else:
    #             v.select = False
    #     bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.fill()
    # # Enter object mode
    # bpy.ops.object.mode_set(mode="OBJECT")

        
    # Laplacian Smoothing
    bpy.context.view_layer.objects.active = bpy.data.objects['Plane']
    bpy.data.objects['Plane'].select_set(True)
    bpy.ops.object.modifier_add(type='LAPLACIANSMOOTH')
    bpy.context.object.modifiers['LaplacianSmooth'].lambda_factor = 10
    bpy.ops.object.modifier_apply(modifier="LaplacianSmooth")

    # Select the previous plane
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects["Plane"].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects["Plane"]

    # Select the object
    plane_name = "Plane"  # Replace with the name of your 3D object in Blender
    bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[plane_name].select_set(True)


    # Export the object as an STL file
    output_path = os.path.join(stl_folder, object_name + '.stl')
    bpy.ops.export_mesh.stl(filepath=output_path, use_selection=True)

    # Set the output path for the OBJ and MTL files
    output_obj_path = os.path.join(obj_folder, object_name + ".obj")
    output_mtl_path = os.path.join(obj_folder, object_name + ".mtl")

    # Export the object as an OBJ file
    # bpy.ops.export_scene.obj(filepath=output_obj_path, use_selection=True, use_materials=True, use_triangles=True, path_mode='AUTO')
    bpy.ops.export_scene.obj(filepath=output_obj_path, use_selection=True)

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


folders = open('folders.txt', 'r')
rgb_folder = folders.readline().strip()
depth_folder = folders.readline().strip()
stl_folder = folders.readline().strip()
obj_folder = folders.readline().strip()
folders.close()

rgb_files = os.listdir(rgb_folder)
for filename in rgb_files:
    depth_file = os.path.join(depth_folder, filename)
    rgb_file = os.path.join(rgb_folder, filename)
    export_obj_from_depth_and_rgb(depth_file, rgb_file, stl_folder, obj_folder)


# # Exit
# bpy.ops.wm.quit_blender()