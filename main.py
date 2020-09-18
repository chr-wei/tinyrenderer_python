from tiny_image import TinyImage
import our_gl as gl
import numpy as np

from geom import Vector_3D, Matrix_3D, matmul

from model import get_model_face_ids, get_vertices, get_model_texture_points

if __name__ == "__main__":

    obj_filename = "obj/head.obj"
    texture_filename = None
    output_filename = "out.png"

    image = TinyImage(500, 500)

    if texture_filename is None:
        texture_image = None
    else:
        texture_image = TinyImage()
        texture_image.load_image(texture_filename)
      
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices = get_vertices(obj_filename)

    print("Reading texture coordinates ...")
    texture_points = get_model_texture_points(obj_filename)
    gl.draw_textured_mesh(face_id_data, vertices, texture_points, texture_image, image)

    image.save_to_disk(output_filename)