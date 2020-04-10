from collections import namedtuple
from operator import attrgetter
from tiny_image import TinyImage
import random

# Tuple definitions

Point = namedtuple ("Point", "x y")
Vertex = namedtuple("Vertex", "x y z")
VertexIds = namedtuple("VertexIds", "id_one id_two id_three")
TangentIds = namedtuple("TangentIds", "id_one id_two id_three")
NormalIds = namedtuple("NormalIds", "id_one id_two id_three")
FacedataIds = namedtuple("FacedataIds", "VertexIds TangentIds NormalIds")

BoundingBox = namedtuple("BoundingBox", "x_min y_min z_min x_max y_max z_max")



def draw_line(p0, p1, image, color):
    """Draw a line onto an image."""

    (x0, y0) = p0
    (x1, y1) = p1

    if abs(x1-x0) < abs(y1-y0):
        # Swap to prevent whitespace when y distance is higher than x distance (steep line)
        steep_line = True
        (y1, y0, x1, x0) = (x1, x0, y1, y0)
    else:
        steep_line = False


    if x0 == x1:
        # Due to switching steep lines this only occurs if y0 == y1 and x0 == x1
        # Only draw a dot in this case
        image.set(x0, y0, color)
        return image

    elif x0 > x1: 
        (y1, y0, x1, x0) = (y0, y1, x0, x1)

    for x in range(x0, x1+1):
        #ToDo: Optimize speed using non float operations
        y = y0 + (y1-y0) / (x1-x0) * (x-x0) + .5

        if steep_line:
            image.set(y, x, color)
        else:
            image.set(x, y, color)

    return image



def draw_triangle(p0, p1, p2, image, color):
    image = draw_line(p0, p1, image, color)
    image = draw_line(p1, p2, image, color)
    image = draw_line(p2, p0, image, color)
    return image

    

def draw_meshtriangles(face_id_data, vertices, bounding_box, image, color):

    x_shift = (bounding_box.x_max + bounding_box.x_min) / 2
    y_shift = (bounding_box.y_max + bounding_box.y_min) / 2

    x_scale = image.width / (bounding_box.x_max - bounding_box.x_min)
    y_scale = image.height / (bounding_box.y_max - bounding_box.y_min)

    scale = min(x_scale, y_scale) * .8

    print("Drawing " + str(len(face_id_data)) + " triangles ...")
    
    for face in face_id_data.values():
        vert_ids = face.VertexIds
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]

        x0 = int((v0.x-x_shift)*scale + image.get_width() /2)
        y0 = int((v0.y-y_shift)*scale + image.get_height() / 2)

        x1 = int((v1.x-x_shift)*scale + image.get_width() /2)
        y1 = int((v1.y-y_shift)*scale + image.height / 2)

        x2 = int((v2.x-x_shift)*scale + image.get_width() /2)
        y2 = int((v2.y-y_shift)*scale + image.get_height() / 2)

        image = draw_triangle(Point(x0, y0), Point(x1, y1), Point(x2, y2), image, color)
    return image



def draw_filled_meshtriangles(face_id_data, vertices, bounding_box, image):

    x_shift = (bounding_box.x_max + bounding_box.x_min) / 2
    y_shift = (bounding_box.y_max + bounding_box.y_min) / 2

    x_scale = image.width / (bounding_box.x_max - bounding_box.x_min)
    y_scale = image.height / (bounding_box.y_max - bounding_box.y_min)

    scale = min(x_scale, y_scale) * .8

    print("Drawing " + str(len(face_id_data)) + " triangles ...")
    
    for face in face_id_data.values():
        vert_ids = face.VertexIds
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]

        x0 = int((v0.x-x_shift)*scale + image.get_width() /2)
        y0 = int((v0.y-y_shift)*scale + image.get_height() / 2)

        x1 = int((v1.x-x_shift)*scale + image.get_width() /2)
        y1 = int((v1.y-y_shift)*scale + image.height / 2)

        x2 = int((v2.x-x_shift)*scale + image.get_width() /2)
        y2 = int((v2.y-y_shift)*scale + image.get_height() / 2)

        colors = ["cyan", "gray", "lightblue", "orange", "purple"]
        image = draw_filled_triangle(Point(x0, y0), Point(x1, y1), Point(x2, y2), image, 
            random.choice(colors))
    return image



def draw_filled_triangle(p0: Point, p1: Point, p2: Point, image: TinyImage, color):
    # image = triangle(p0,p1,p2, image, color)
    sorted_points = [p0, p1, p2]
    sorted_points.sort(key=attrgetter('x'))

    edge_main = (sorted_points[0], sorted_points[2])
    edge_left = (sorted_points[0], sorted_points[1])
    edge_right = (sorted_points[1], sorted_points[2])

    x0_main = edge_main[0].x
    y0_main = edge_main[0].y
    x1_main = edge_main[1].x
    y1_main = edge_main[1].y

    for x_sweep in range(x0_main, x1_main + 1):
        if x_sweep < edge_left[1].x:
            x0_minor = edge_left[0].x
            y0_minor = edge_left[0].y
            x1_minor = edge_left[1].x
            y1_minor = edge_left[1].y
        else:
            x0_minor = edge_right[0].x
            y0_minor = edge_right[0].y 
            x1_minor = edge_right[1].x
            y1_minor = edge_right[1].y
        
        if x1_main == x0_main:
            y_main = y1_main
        else:
            y_main = int(y0_main + (y1_main-y0_main) / (x1_main-x0_main) * (x_sweep-x0_main) + .5)

        if x1_minor == x0_minor:
            y_minor = y0_minor
        else:
            y_minor = int(y0_minor + (y1_minor-y0_minor) / (x1_minor-x0_minor) * (x_sweep-x0_minor) + .5)

        if (y_minor > y_main):
            # Make y_main greater than y_minor
            (y_main, y_minor) = (y_minor, y_main)
        
        for y in range(y_minor, y_main + 1):
            image.set(x_sweep, y, color)

    return image