import numpy as np
from progressbar import progressbar

from tiny_image import TinyImage

import our_gl as gl
from geom import Vector_3D, cross_product
from model import Model_Storage, get_model_face_ids, get_vertices, NormalMapType
from tiny_shaders import Flat_Shader, Gouraud_Shader, Gouraud_Shader_Segregated, Diffuse_Gouraud_Shader, \
                         Normalmap_Shader, Specularmap_Shader

if __name__ == "__main__":
    
    # Model property selection
    model_prop_set = 0
    if model_prop_set == 0:
        obj_filename = "obj/autumn/autumn.obj"
        diffuse_filename = "obj/autumn/TEX_autumn_body_color.png"
        normal_map_filename = "obj/autumn/TEX_autumn_body_normals_wrld_space.tga"
        normal_map_type = NormalMapType.GLOBAL
        specular_map_filename = "obj/autumn/TEX_autumn_body_spec.tga"
        output_filename = "renders/out.png"

    else:
        obj_filename = "obj/head/head.obj"
        diffuse_filename = "obj/head/head_diffuse.tga"
        normal_map_filename = "obj/head/head_nm.tga"
        normal_map_type = NormalMapType.GLOBAL
        specular_map_filename = "obj/head/head_spec.tga"
        output_filename = "renders/out.png"
    
    # Image property selection
    img_prop_set = 0
    if img_prop_set == 0:
        (w, h) = (2000, 2000)
    else:
        (w, h) = (800, 800)

    image = TinyImage(w, h)

    # View property selection
    view_prop_set = 1
    if view_prop_set == 0:
        eye = Vector_3D(0, 0, 1) # Lookat camera 'eye' position
        center = Vector_3D(0, 0, 0) # Lookat 'center'. 'eye' looks at center
        up = Vector_3D(0, 1, 0) # Camera 'up' direction
        scale = .8 # Viewport scaling
    elif view_prop_set == 1:
        eye = Vector_3D(1, 0, 1)
        center = Vector_3D(0, 0, 0)
        up = Vector_3D(0, 1, 0)
        scale = .8
    else:
        eye = Vector_3D(1, 0, 0) # Lookat camera 'eye' position
        center = Vector_3D(0, 0, 0) # Lookat 'center'. 'eye' looks at center
        up = Vector_3D(0, 1, 0) # Camera 'up' direction
        scale = .8 # Viewport scaling

    # Light property
    light_dir = Vector_3D(1, 0, 1).norm()

    
    print("Reading modeldata ...")
    mdl = Model_Storage(object_name = "autumn", obj_filename = obj_filename, 
                        diffuse_map_filename = diffuse_filename, 
                        normal_map_filename=normal_map_filename, normal_map_type = normal_map_type,
                        specular_map_filename=specular_map_filename)

    # Define tranformation matrices

    # Generate model transformation matrix which transforms vertices according to the model bounding box
    # min[-1, -1, -1] to max[1, 1, 1] object space
    M_model = gl.model_transform(mdl.bbox[0], mdl.bbox[1])

    # Generate cam transformation
    M_lookat = gl.lookat(eye, center, up)

    # Generate perspective transformation
    M_perspective = gl.perspective(4.0)

    # Generate transformation to final viewport
    M_viewport = gl.viewport(+scale*w/8, +scale*h/8, scale*w, scale*h, 255)

    # Combine matrices
    M_modelview = M_lookat * M_model
    M_pe = M_perspective * M_modelview
    M_sc = M_viewport * M_pe

    M_pe_IT = M_pe.tr().inv()

    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)]

    shader_prop_set = 4
    if shader_prop_set == 0:
        shader = Gouraud_Shader(mdl, light_dir, M_sc)
    elif shader_prop_set == 1:
        shader = Gouraud_Shader_Segregated(mdl, light_dir, M_sc, 4)
    elif shader_prop_set == 2:
        shader = Diffuse_Gouraud_Shader(mdl, light_dir, M_sc)
    elif shader_prop_set == 3:
        shader = Normalmap_Shader(mdl, light_dir, M_pe, M_sc, M_pe_IT)
    elif shader_prop_set == 4:
        shader = Specularmap_Shader(mdl, light_dir, M_pe, M_sc, M_pe_IT)
    else:
        shader = Flat_Shader(mdl, light_dir, M_sc)

    # Iterate model faces
    print("Drawing triangles ...")

    screen_coords = [None] * 3

    for face_idx in progressbar(range(mdl.get_face_count())):
        for face_vert_idx in range(3):
            # Get transformed vertex and prepare internal shader data
            screen_coords[face_vert_idx] = shader.vertex(face_idx, face_vert_idx)        
        
        # Rasterize triangle
        image = gl.draw_triangle(screen_coords, shader, zbuffer, image)

    image.save_to_disk(output_filename)