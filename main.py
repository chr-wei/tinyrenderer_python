import sys

from tiny_image import TinyImage
from our_gl import triangle, line
from model import get_model_face_ids, get_vertices

if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Please pass .obj filepath and output img filepath.")
        sys.exit(1)
    else:
        obj_filename = sys.argv[1]
        output_filename = sys.argv[2]

    image = TinyImage(3200, 1800)
    
    #Excercises
    #image.set(50,30, "red") ##1
    #image = line(0, 0, 100, 20, image, "white")##2
    #image = our_gl.line(0, 0, 20, 100, image, "white")##3
    #image = triangle((3,5), (20,100), (110,50), image, "white")##4
    
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bounding_box = get_vertices(obj_filename)

    x_shift = (bounding_box.x_max + bounding_box.x_min) / 2
    y_shift = (bounding_box.y_max + bounding_box.y_min) / 2

    x_scale = image.width / (bounding_box.x_max - bounding_box.x_min)
    y_scale = image.height / (bounding_box.y_max - bounding_box.y_min)

    scale = min(x_scale, y_scale) * .8

    print("Drawing " + str(len(face_id_data)) + " triangles ...")
    for id, face in face_id_data.items():
        vert_ids = face.Vertex_Ids
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]

        x0 = int((v0.x-x_shift)*scale + image.width /2)
        y0 = int((v0.y-y_shift)*scale + image.height / 2)

        x1 = int((v1.x-x_shift)*scale + image.width /2)
        y1 = int((v1.y-y_shift)*scale + image.height / 2)

        x2 = int((v2.x-x_shift)*scale + image.width /2)
        y2 = int((v2.y-y_shift)*scale + image.height / 2)
        
        image = triangle((x0, y0), (x1, y1), (x2, y2), image, "white")
        
    image.save_to_disk(output_filename)