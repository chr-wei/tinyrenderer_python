from tiny_image import TinyImage
import our_gl as gl
import numpy as np

from geom import Vector_3D, Matrix_3D, matmul

from model import get_model_face_ids, get_vertices, get_model_texture_points

def excercise_textured_mesh(obj_filename, texture_filename, output_filename):
    """Draw a filled mesh with random facet colors"""
    image = TinyImage(2000, 2000)

    if texture_filename is None:
        texture_image = None
    else:
        texture_image = TinyImage()
        texture_image.load_image(texture_filename)
      
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bounding_box = get_vertices(obj_filename)

    print("Reading texture coordinates ...")
    texture_points = get_model_texture_points(obj_filename)
    gl.draw_textured_mesh(face_id_data, vertices, bounding_box, texture_points, texture_image, image)

    image.save_to_disk(output_filename)

if __name__ == "__main__":
    a = Vector_3D(1, 2, 3)
    b = Vector_3D(2,3,4)
    # c= a*b
    # print(c)

    d = Matrix_3D([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])
    # e = Matrix_3D([[1, 5, 3],
    #                [4, 5, 6],
    #                [7, 5, 9]])
    # print(d*e)
    # print(d*a)
    # print(d.tr())

    # f = np.array([[1, 2, 3],
    #                [4, 5, 6],
    #                [7, 8, 9]])
    # g = np.array([[1, 5, 3],
    #                [4, 5, 6],
    #                [7, 5, 9]])
    # h = np.array([1,2,3])

    # print(np.matmul(f,g))
    # print(np.matmul(f,h))
    e = Matrix_3D([[1, 2, 0],
                   [2, 4, 1],
                   [2, 1, 0]])
    print( e.inv() * e)
    
    # excercise_textured_mesh("obj/head.obj", None, "out.png")##9.1 and 9.2