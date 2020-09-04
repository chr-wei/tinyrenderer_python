from collections import namedtuple
from operator import attrgetter
from tiny_image import TinyImage
from model import get_texture_color

import random
import numpy as np
import PIL

# Tuple definitions
Point = namedtuple ("Point", "x y")
Vertex = namedtuple("Vertex", "x y z")

BoundingBox = namedtuple("BoundingBox", "x_min y_min z_min x_max y_max z_max")

light_dir = Vertex(-1, -1, -1)

def draw_line(p0, p1, image, color):
    """Draw p0 line onto an image."""

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
        # Only draw p0 dot in this case
        image.set(x0, y0, color)
        return image

    elif x0 > x1: 
        (y1, y0, x1, x0) = (y0, y1, x0, x1)

    for x in range(x0, x1+1):
        #ToDo: Optimize speed using non float operations
        y = int(y0 + (y1-y0) / (x1-x0) * (x-x0) + .5)

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

    
def draw_ref_triangle(p0, p1, p2, image, color):
    image._draw.line((p0.x, p0.y, p1.x, p1.y), fill = color)
    image._draw.line((p0.x, p0.y, p2.x, p2.y), fill = color)
    image._draw.line((p1.x, p1.y, p2.x, p2.y), fill = color)
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
        # image = draw_filled_triangle(Point(x0, y0), Point(x1, y1), Point(x2, y2), image, 
        #    random.choice(colors))##5
        image = draw_rasterized_triangle(Point(x0, y0), Point(x1, y1), Point(x2, y2), image, 
            random.choice(colors))##7
    return image

def draw_filled_triangle(p0: Point, p1: Point, p2: Point, image: TinyImage, color):
    image = draw_triangle(p0,p1,p2, image, color)
    
    sorted_points = [p0, p1, p2]
    sorted_points.sort(key=attrgetter('x'))

    edge_main = (sorted_points[0], sorted_points[2])
    edge_left = (sorted_points[0], sorted_points[1])
    edge_right = (sorted_points[1], sorted_points[2])

    x0_main = edge_main[0].x
    y0_main = edge_main[0].y
    x1_main = edge_main[1].x
    y1_main = edge_main[1].y

    for x_sweep in range(x0_main + 1, x1_main):
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
        
        for y in range(y_minor+1, y_main):
            image.set(x_sweep, y, color)

    return image

def draw_rasterized_triangle(p0: Point, p1: Point, p2: Point, image: TinyImage, color):
    points = [p0, p1, p2]

    points.sort(key=attrgetter('x'))
    x_min = points[0].x
    x_max = points[2].x

    points.sort(key=attrgetter('y'))
    y_min = points[0].y
    y_max = points[2].y

    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            (one_uv, u, v) = barycentric(p0, p1, p2, Point(x,y))
            if one_uv >= 0 and u >= 0 and v >= 0:
                image.set(x, y, color)
    return image

def draw_zbuffered_triangle(v0: Vertex, v1: Vertex, v2: Vertex, zbuffer: list, image: TinyImage, color):
    points = [v0, v1, v2]

    points.sort(key=attrgetter('x'))
    x_min = points[0].x
    x_max = points[2].x

    points.sort(key=attrgetter('y'))
    y_min = points[0].y
    y_max = points[2].y

    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            (one_uv, u, v) = barycentric(Point(v0.x, v0.y), Point(v1.x, v1.y), Point(v2.x, v2.y), Point(x,y))
            if one_uv >= 0 and u >= 0 and v >= 0:
                z = one_uv*v0.z + u * v1.z + v * v2.z
                if z > zbuffer[x][y]:
                    zbuffer[x][y] = z
                    image.set(x, y, color)
    return image

def draw_textured_triangle(v0: Vertex, v1: Vertex, v2: Vertex, zbuffer: list, 
                           p0: Point, p1: Point, p2: Point, texture_image : TinyImage, shading_factor: float, 
                           image: TinyImage):
    points = [v0, v1, v2]

    points.sort(key=attrgetter('x'))
    x_min = points[0].x
    x_max = points[2].x

    points.sort(key=attrgetter('y'))
    y_min = points[0].y
    y_max = points[2].y

    for x in range(x_min, x_max+1):
        for y in range(y_min, y_max+1):
            (one_uv, u, v) = barycentric(Point(v0.x, v0.y), Point(v1.x, v1.y), Point(v2.x, v2.y), Point(x,y))
            if one_uv >= 0 and u >= 0 and v >= 0:
                z = one_uv*v0.z + u * v1.z + v * v2.z
                p_texture = Point(*(np.multiply(one_uv, p0) + np.multiply(u, p1) + np.multiply(v, p2)))
            
                if z > zbuffer[x][y]:
                    zbuffer[x][y] = z

                    # Get texture color
                    color = get_texture_color(texture_image, p_texture.x, p_texture.y)
                    color = tuple(int(elem) for elem in np.multiply(color, shading_factor))
                    image.set(x, y, color)
    return image

def barycentric(p0:Point, p1:Point, p2:Point, P:Point):
    (u, v, r) = cross_product(Vertex(p1.x-p0.x, p2.x-p0.x, p0.x-P.x), Vertex(p1.y-p0.y, p2.y-p0.y, p0.y-P.y))
    
    if r == 0:
        # Triangle is degenerated
        return (-1,-1,-1)
    else:
        # Component r should be 1: Normalize components 
        return (1-(u+v)/r, u/r, v/r)

def cross_product(v0:Vertex, v1:Vertex):
    c0 = v0.y*v1.z - v0.z*v1.y
    c1 = v0.z*v1.x - v0.x*v1.z
    c2 = v0.x*v1.y - v0.y*v1.x
    return Vertex(c0, c1, c2)

def normalize(v0:Vertex):
    length = (v0.x**2 + v0.y**2 + v0.z**2)**(1/2)
    return Vertex(*np.multiply(v0, 1/length))

def scalar(v0:Vertex, v1:Vertex):
    return v0.x*v1.x + v0.y*v1.y + v0.z*v1.z

def draw_flat_shaded_meshtriangles(face_id_data, vertices, bounding_box, image):

    x_shift = (bounding_box.x_max + bounding_box.x_min) / 2
    y_shift = (bounding_box.y_max + bounding_box.y_min) / 2

    x_scale = image.width / (bounding_box.x_max - bounding_box.x_min)
    y_scale = image.height / (bounding_box.y_max - bounding_box.y_min)

    scale = min(x_scale, y_scale) * .8

    print("Drawing " + str(len(face_id_data)) + " triangles ...")
    
    w, h = image.get_width(), image.get_height()
    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)] #8.2 

    for face in face_id_data.values():
        vert_ids = face.VertexIds
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]
        
        # Calculating color shading
        n = cross_product(Vertex(*np.subtract(v0, v1)) , Vertex(*np.subtract(v2, v0)))
        cos_phi = scalar(normalize(n), normalize(light_dir))

        if cos_phi < 0:
            continue
        
        color = int(255 * cos_phi)
        color = (color, color, color)

        x0 = int((v0.x-x_shift)*scale + w /2)
        y0 = int((v0.y-y_shift)*scale + h / 2)

        x1 = int((v1.x-x_shift)*scale + w /2)
        y1 = int((v1.y-y_shift)*scale + h / 2)

        x2 = int((v2.x-x_shift)*scale + w / 2)
        y2 = int((v2.y-y_shift)*scale + h / 2)

        # image = draw_rasterized_triangle(Point(x0, y0), Point(x1, y1), Point(x2, y2), image, color)##8.1
        image = draw_zbuffered_triangle(Vertex(x0, y0, v0.z),  Vertex(x1, y1, v1.z), Vertex(x2, y2, v2.z), zbuffer, image, color)##8.2
    return image

def draw_textured_mesh(face_id_data : dict, vertices : dict, bounding_box : BoundingBox, 
                       texture_points : dict, texture_image : TinyImage, 
                       image : TinyImage):

    x_shift = (bounding_box.x_max + bounding_box.x_min) / 2
    y_shift = (bounding_box.y_max + bounding_box.y_min) / 2

    x_scale = image.width / (bounding_box.x_max - bounding_box.x_min)
    y_scale = image.height / (bounding_box.y_max - bounding_box.y_min)

    scale = min(x_scale, y_scale) * .8

    print("Drawing " + str(len(face_id_data)) + " triangles ...")
    
    w, h = image.get_width(), image.get_height()
    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)] #8.2 

    for face in face_id_data.values():
        vert_ids = face.VertexIds
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]

        texture_pt_ids = face.TexturePointIds
        p0 = texture_points[texture_pt_ids.id_one]
        p1 = texture_points[texture_pt_ids.id_two]
        p2 = texture_points[texture_pt_ids.id_three]
        
        # Calculating color shading
        n = cross_product(Vertex(*np.subtract(v0, v1)) , Vertex(*np.subtract(v2, v0)))
        cos_phi = scalar(normalize(n), normalize(light_dir))

        if cos_phi < 0:
            continue
        
        color = int(255 * cos_phi)
        color = (color, color, color)

        x0 = int((v0.x-x_shift)*scale + w /2)
        y0 = int((v0.y-y_shift)*scale + h / 2)

        x1 = int((v1.x-x_shift)*scale + w /2)
        y1 = int((v1.y-y_shift)*scale + h / 2)

        x2 = int((v2.x-x_shift)*scale + w / 2)
        y2 = int((v2.y-y_shift)*scale + h / 2)

        image = draw_textured_triangle(Vertex(x0, y0, v0.z),  Vertex(x1, y1, v1.z), Vertex(x2, y2, v2.z), zbuffer,
                                       Point(p0.x, p0.y), Point(p1.x, p1.y), Point(p2.x, p2.y), texture_image, cos_phi, 
                                       image)
    return image