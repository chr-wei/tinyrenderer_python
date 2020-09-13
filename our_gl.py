from operator import attrgetter

import numpy as np
from numpy import array

from tiny_image import TinyImage
from model import get_texture_color

light_dir = array([0, 0, -1])


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

def draw_triangle(v0: array, v1: array, v2: array, zbuffer: list, 
                           p0: array, p1: array, p2: array, texture_image : TinyImage, shading_factor: float, 
                           image: TinyImage):
                           
    points = np.column_stack((v0, v1, v2))
    max_vals = points.max(axis = 1)
    min_vals = points.min(axis = 1)

    x_min = min_vals[0]
    x_max = max_vals[0]
    y_min = min_vals[1]
    y_max = max_vals[1]

    for x in range(int(x_min), int(x_max)+1):
        for y in range(int(y_min), int(y_max)+1):
            (one_uv, u, v) = barycentric(v0[0:2], v1[0:2], v2[0:2], [x,y])
            if one_uv >= 0 and u >= 0 and v >= 0:
                z = one_uv*v0[2] + u * v1[2] + v * v2[2]
            
                if z > zbuffer[x][y]:
                    zbuffer[x][y] = z

                    if None in [p0, p1, p2, texture_image]:
                        color = (255, 255, 255)
                    else:
                        # Get texture color
                        p_texture = one_uv * p0 + u * p1 + v * p2
                        color = get_texture_color(texture_image, p_texture[0], p_texture[1])
                    
                    color = tuple(int(elem) for elem in np.multiply(color, shading_factor))
                    image.set(x, y, color)
    return image

def barycentric(p0: array, p1: array, p2: array, P: array):
    
    (u, v, r) = np.cross([p1[0]-p0[0], p2[0]-p0[0], p0[0] - P[0]], 
                              [p1[1]-p0[1], p2[1]-p0[1], p0[1]-P[1]])
    
    if r == 0:
        # Triangle is degenerated
        return (-1,-1,-1)
    else:
        # Component r should be 1: Normalize components 
        return (1-(u+v)/r, u/r, v/r)

def draw_textured_mesh(face_id_data : list, vertices : list, bounding_box : array, 
                       texture_points : list, texture_image : TinyImage, 
                       image : TinyImage):

    # x_shift = (bounding_box[0, 1] + bounding_box[0, 0]) / 2
    # y_shift = (bounding_box[1, 1] + bounding_box[1, 0]) / 2

    # x_scale = image.width / (bounding_box[0, 1] - bounding_box[0, 0])
    # y_scale = image.height / (bounding_box[1, 1] - bounding_box[1, 0])

    # scale = min(x_scale, y_scale) * .8

    print("Drawing " + str(len(face_id_data)) + " triangles ...")
    
    w, h = image.get_width(), image.get_height()
    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)]


    M_modelview = lookat(array([0, 0, 1]), 
                         array([0, 0, 0]), 
                         array([0, 1, 0]))

    M_perspective = perspective(4)
    M_viewport = viewport(0, 0, w, h, 255)

    M = np.matmul(np.matmul(M_viewport, M_perspective), M_modelview)

    for idx, face in enumerate(face_id_data):
        print(idx)
        vert_ids = face.VertexIds
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]

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
        n = np.cross(v0-v1, v2-v0)
        n = n/np.linalg.norm(n)
        cos_phi = np.dot(n, light_dir/np.linalg.norm(light_dir))

        if cos_phi < 0:
            continue

        # Calculate screen coords
        x0 = int(v0[0])
        y0 = int(v0[1])

        x1 = int(v1[0])
        y1 = int(v1[1])

        x2 = int(v2[0])
        y2 = int(v2[1])

        image = draw_triangle(v0, v1, v2, zbuffer,
                                       p0, p1, p2, texture_image, cos_phi, 
                                       image)
    return image

def transform_vertex(v : array, M: array):

    v = array([v[0], v[1], v[2], 1])
    v = M.dot(v)
    v = v / v[3]
    return v[0:3]

def lookat(eye: array, center: array, up: array):
    z = eye - center
    z = z/np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    y = y / np.linalg.norm(x)

    M_inv = array([[x[0], x[1], x[2], 0], 
                  [y[0], y[1], y[2], 0],
                  [z[0], z[1], z[2], 0],
                  [0   , 0   , 0   , 1]])

    M_tr = array([[1, 0, 0, -center[0]], 
                  [0, 1, 0, -center[1]],
                  [0, 0, 1, -center[2]],
                  [0, 0, 0, 1         ]])

    M_modelview = M_inv * M_tr

    return M_modelview

def perspective(c: float):
    M_perspective = array([[1, 0, 0,    0],
                           [0, 1, 0,    0],
                           [0, 0, 1,    0],
                           [0, 0, -1/c, 1]])
    
    return M_perspective

def viewport(o_x, o_y, w, h, d):
    M_viewport = array([[w/2, 0,   0  , o_x + w/2],
                       [0  , h/2, 0  , o_y + h/2],
                       [0  , 0,   d/2, d/2      ],
                       [0  , 0,   0  , 1        ]])
    return M_viewport

def normal_transformation(M_transform):
    return M_transform.transpose().inverse()