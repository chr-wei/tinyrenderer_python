from tiny_image import TinyImage
import our_gl as gl

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
    excercise_textured_mesh("obj/autumn.obj", None, "autumn.png")##9.1 and 9.2