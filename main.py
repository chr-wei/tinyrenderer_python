import sys

from tiny_image import TinyImage
from our_gl import Point, draw_triangle, draw_line, draw_meshtriangles, draw_filled_triangle, draw_filled_meshtriangles

from model import get_model_face_ids, get_vertices 


#Excercises
#
#
def excercise_point():
    image = TinyImage(100, 100)
    image.set(50,30, "red") ##1.1
    image.save_to_disk("out.png")

def excercise_lines():
    image = TinyImage(200, 200)

    image = draw_line((0,0), (100,20), image, "white")##2.1
    image = draw_line((40,100), (100, 29), image, "white")##2.2
    # image = draw_line((0,0), (image.width-1,image.height-1), image, "white")##2.3
    image.save_to_disk("out.png")

def excercise_triangles():
    image = TinyImage(200, 200)
    image = draw_triangle((3,5), (20,100), (110,50), image, "white")##3.1
    image.save_to_disk("out.png")

def excercise_mesh(obj_filename, output_filename):
    image = TinyImage(2000, 2000)
        
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bounding_box = get_vertices(obj_filename)

    draw_meshtriangles(face_id_data, vertices, bounding_box, image, "white")##6

    image.save_to_disk(output_filename)
        
def excercise_filled_triangles():
    image = TinyImage(300, 300)
    p1 = Point(100, 5)
    p2 = Point(100, 150)
    p3 = Point(200, 50)

    image = draw_filled_triangle(p1, p2, p3, image, "white")##5.1
    image = draw_triangle(p1, p2, p3, image, "red")##5.1

    # p4 = Point(20, 240)##5.2
    # p5 = Point(180, 199)##5.2
    # image = draw_filled_triangle(p2, p4, p5, image, "white")##5.2

def excercise_filled_mesh(obj_filename, output_filename):
    image = TinyImage(2000, 2000)
        
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bounding_box = get_vertices(obj_filename)

    draw_filled_meshtriangles(face_id_data, vertices, bounding_box, image)

    image.save_to_disk(output_filename)


if __name__ == "__main__":

    # excercise_point()##1
    # excercise_lines()##2
    # excercise_triangles()##3
    # excercise_mesh("obj/autumn.obj", "autumn.png")##4
    # excercise_filled_triangles()##5
    excercise_filled_mesh("obj/spring_autumn.obj", "spring_autumn.png")##6