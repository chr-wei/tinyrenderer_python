import sys

from tiny_image import TinyImage
from our_gl import (Point, Vertex, cross_product, 
                    draw_triangle, draw_line, draw_meshtriangles, draw_ref_triangle, 
                    draw_filled_triangle, draw_filled_meshtriangles, draw_rasterized_triangle, draw_flat_shaded_meshtriangles)

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
    image.save_to_disk("out.png")

def excercise_filled_mesh(obj_filename, output_filename):
    image = TinyImage(2000, 2000)
        
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bounding_box = get_vertices(obj_filename)

    draw_filled_meshtriangles(face_id_data, vertices, bounding_box, image)

    image.save_to_disk(output_filename)

def excercise_rasterized_triangle():
    # import numpy##7.1
    # Test cross product
    # v1 = Vertex(1,2,3)##7.1
    # v2 = Vertex(4,5,6)##7.1
    # print(cross_product(v1,v2))##7.1
    # print(numpy.cross(v1,v2))##7.1

    image = TinyImage(300, 300)
    p0 = Point(50, 5)
    p1 = Point(200, 10)
    p2 = Point(100,  250)


    image = draw_ref_triangle(p0, p1, p2, image, "red")#7.2
    image = draw_rasterized_triangle(p0, p1, p2, image, "white")#7.2
    image.save_to_disk("out.png")

def excercise_flat_shaded_mesh(obj_filename, output_filename):
    image = TinyImage(2000, 2000)
        
    print("Reading facedata ...")
    face_id_data = get_model_face_ids(obj_filename)

    print("Reading vertices ...")
    vertices, bounding_box = get_vertices(obj_filename)

    draw_flat_shaded_meshtriangles(face_id_data, vertices, bounding_box, image)

    image.save_to_disk(output_filename)

if __name__ == "__main__":

    # excercise_point()##1
    # excercise_lines()##2
    # excercise_triangles()##3
    # excercise_mesh("obj/autumn.obj", "autumn.png")##4
    # excercise_filled_triangles()##5
    # excercise_filled_mesh("obj/spring_autumn.obj", "spring_autumn.png")##6
    # excercise_rasterized_triangle()##7
    excercise_filled_mesh("obj/autumn.obj", "autumn.png")##7
    # excercise_flat_shaded_mesh("obj/autumn.obj", "autumn.png")##8