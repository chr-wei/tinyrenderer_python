import numpy as np
from progressbar import progressbar

from tiny_image import TinyImage

import our_gl as gl
from geom import Vector_3D, transform_vertex, cross_product
from model import Model_Storage, get_model_face_ids, get_vertices

if __name__ == "__main__":
    
    # Model property selection
    model_prop_set = 1
    if model_prop_set == 0:
        obj_filename = "obj/autumn.obj"
        texture_filename = "obj/TEX_autumn_body_color.png"
        output_filename = "out.png"

    elif model_prop_set == 1:
        obj_filename = "obj/autumn.obj"
        texture_filename = None
        output_filename = "out.png"

    else:
        obj_filename = "obj/head.obj"
        texture_filename = "obj/african_head_diffuse.tga"
        output_filename = "out.png"
    
    # Image property selection
    img_prop_set = 0
    if img_prop_set == 0:
        (w, h) = (2000, 2000)
    else:
        (w, h) = (800, 800)

    image = TinyImage(w, h)

    # View property selection
    view_prop_set = 0
    if view_prop_set == 0:
        eye = Vector_3D(0, 0, 1) # Lookat camera 'eye' position
        center = Vector_3D(0, 0, 0) # Lookat 'center'. 'eye' looks at center
        up = Vector_3D(0, 1, 0) # Camera 'up' direction
        scale = .8 # Viewport scaling
    else:
        eye = Vector_3D(0, 0, 1) # Lookat camera 'eye' position
        center = Vector_3D(0, 0, 0) # Lookat 'center'. 'eye' looks at center
        up = Vector_3D(0, 1, 0) # Camera 'up' direction
        scale = .8 # Viewport scaling

    # Light property
    light_dir = Vector_3D(0, -1, -1)

    
    print("Reading modeldata ...")
    mdl = Model_Storage("autumn", obj_filename, texture_filename)
    
    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)]

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
    M = M_viewport * M_perspective * M_modelview
    
    light_dir = (M_modelview * light_dir.expand_4D_vect()).project_3D()

    # Iterate model faces
    print("Drawing triangles ...")

    for face in progressbar(mdl.face_id_data):
        vert_ids = face.VertexIds

        v0 = mdl.vertices[vert_ids.id_one - 1]
        v1 = mdl.vertices[vert_ids.id_two - 1]
        v2 = mdl.vertices[vert_ids.id_three - 1]

        v0 = transform_vertex(v0, M)
        v1 = transform_vertex(v1, M)
        v2 = transform_vertex(v2, M)

        if not mdl.diffuse_map is None:
            texture_pt_ids = face.TexturePointIds
            p0 = mdl.diffuse_points[texture_pt_ids.id_one - 1]
            p1 = mdl.diffuse_points[texture_pt_ids.id_two - 1]
            p2 = mdl.diffuse_points[texture_pt_ids.id_three -1]
        else:
            p0 = None
            p1 = None
            p2 = None
        
        # Calculating color shading
        n = cross_product(v0 - v1, v2 - v0)
        if n.norm() is None:
            continue
        cos_phi = n.norm() * light_dir.norm()
        cos_phi = 0 if cos_phi < 0 else cos_phi
        
        image = gl.draw_triangle(v0, v1, v2, zbuffer,
                                 p0, p1, p2, mdl.diffuse_map, cos_phi, 
                                 image)

    image.save_to_disk(output_filename)