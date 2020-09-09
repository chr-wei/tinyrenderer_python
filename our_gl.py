from operator import attrgetter

import numpy as np
from numpy import array
from nptyping import NDArray
from typing import Any

from tiny_image import TinyImage
from model import get_texture_color

light_dir = array([0, 0, -1, 0])

c = 4
M_perspective : NDArray((4,3), Any) = array([
                                            [1, 0, 0,    0],
                                            [0, 1, 0,    0],
                                            [0, 0, 1,    0],
                                            [0, 0, -1/c, 1],
                                            ])

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

def draw_triangle_lines(p0, p1, p2, image, color):
    image = draw_line(p0, p1, image, color)
    image = draw_line(p1, p2, image, color)
    image = draw_line(p2, p0, image, color)
    return image

def draw_triangle(v0: NDArray(Any, Any), v1: NDArray(Any, Any), v2: NDArray(Any, Any), zbuffer: list, 
                           p0: NDArray(Any, Any), p1: NDArray(Any, Any), p2: NDArray(Any, Any), texture_image : TinyImage, shading_factor: float, 
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
            
                if z > zbuffer[x][y]:
                    zbuffer[x][y] = z

                    if None in [p0, p1, p2, texture_image]:
                        color = (255, 255, 255)
                    else:
                        # Get texture color
                        p_texture = Point(*(np.multiply(one_uv, p0) + np.multiply(u, p1) + np.multiply(v, p2)))
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

        v0 = transform_vertex(v0)
        v1 = transform_vertex(v1)
        v2 = transform_vertex(v2)

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
        n = cross_product(Vertex(*np.subtract(v0, v1)) , Vertex(*np.subtract(v2, v0)))
        cos_phi = scalar(normalize(n), normalize(light_dir))

        if cos_phi < 0:
            continue

        # Calculate screen coords
        x0 = int((v0.x-x_shift)*scale + w /2)
        y0 = int((v0.y-y_shift)*scale + h / 2)

        x1 = int((v1.x-x_shift)*scale + w /2)
        y1 = int((v1.y-y_shift)*scale + h / 2)

        x2 = int((v2.x-x_shift)*scale + w / 2)
        y2 = int((v2.y-y_shift)*scale + h / 2)

        image = draw_triangle(Vertex(x0, y0, v0.z),  Vertex(x1, y1, v1.z), Vertex(x2, y2, v2.z), zbuffer,
                                       p0, p1, p2, texture_image, cos_phi, 
                                       image)
    return image

def transform_vertex(v : Vertex):
    v = array([v.x, v.y, v.z, 1])
    v = M_perspective.dot(v)
    v = v / v[3]
    return Vertex(v[0], v[1], v[2])