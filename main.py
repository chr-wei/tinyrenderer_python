import numpy as np
from progressbar import progressbar

from tiny_image import TinyImage

import our_gl as gl
from geom import Vector_3D, Matrix_3D, matmul, cross_product, transform_vertex
from model import get_model_face_ids, get_vertices, get_model_texture_points

if __name__ == "__main__":
    
    draw = 0
    if draw == 0:
        obj_filename = "obj/autumn.obj"
        texture_filename = "obj/TEX_autumn_body_color.png"
        output_filename = "out.png"

    elif draw == 1:
        obj_filename = "obj/autumn.obj"
        texture_filename = None
        output_filename = "out.png"

    else:
        obj_filename = "obj/head.obj"
        texture_filename = "obj/african_head_diffuse.tga"
        output_filename = "out.png"

    image = TinyImage(2000, 2000)

    if texture_filename is None:
        texture_image = None
    else:
        texture_image = TinyImage()
        texture_image.load_image(texture_filename)
      
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bbox = get_vertices(obj_filename)

    print("Reading texture coordinates ...")
    texture_points = get_model_texture_points(obj_filename)

    print("Drawing triangles ...")
    
    w, h = image.get_width(), image.get_height()
    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)]

    # Generate model transformation matrix which transforms vertices according to the model bounding box
    # min[-1, -1, -1] to max[1, 1, 1] object space
    M_model = gl.model(bbox[0], bbox[1])


    M_lookat = gl.lookat(Vector_3D(0, 0, 1), 
                      Vector_3D(0, 0, 0), 
                      Vector_3D(0, 1, 0))

    M_modelview = M_lookat * M_model

    M_perspective = gl.perspective(4.0)
    
    scale = .8
    M_viewport = gl.viewport(+scale*w/8, +scale*h/8, scale*w, scale*h, 255)

    M = M_viewport * M_perspective * M_modelview
    
    light_dir = Vector_3D(0, -1, -1)
    light_dir = (M_modelview * light_dir.expand_4D_vect()).project_3D()

    for face in progressbar(face_id_data, ):
        vert_ids = face.VertexIds

        v0 = vertices[vert_ids.id_one - 1]
        v1 = vertices[vert_ids.id_two - 1]
        v2 = vertices[vert_ids.id_three - 1]

        v0 = transform_vertex(v0, M)
        v1 = transform_vertex(v1, M)
        v2 = transform_vertex(v2, M)

        if not texture_image is None:
            texture_pt_ids = face.TexturePointIds
            p0 = texture_points[texture_pt_ids.id_one]
            p1 = texture_points[texture_pt_ids.id_two]
            p2 = texture_points[texture_pt_ids.id_three]
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
                              p0, p1, p2, texture_image, cos_phi, 
                              image)

    image.save_to_disk(output_filename)