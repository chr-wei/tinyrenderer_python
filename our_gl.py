from operator import attrgetter
from tiny_image import TinyImage
from model import get_texture_color

from geom import Matrix_4D, Vector_3D, Point_2D, cross_product

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

def draw_triangle(v0: Vector_3D, v1: Vector_3D, v2: Vector_3D, zbuffer: list, 
                           p0: Point_2D, p1: Point_2D, p2: Point_2D, texture_image : TinyImage, shading_factor: float, 
                           image: TinyImage):
    points = [v0, v1, v2]

    points.sort(key=attrgetter('x'))
    x_min = int(min(max(points[0].x, 0), image.width - 1))
    x_max = int(min(max(points[2].x, 0), image.width - 1))

    points.sort(key=attrgetter('y'))
    y_min = int(min(max(points[0].y, 0), image.height - 1))
    y_max = int(min(max(points[2].y, 0), image.height - 1))

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            (one_uv, u, v) = barycentric(Point_2D(v0.x, v0.y), Point_2D(v1.x, v1.y), Point_2D(v2.x, v2.y), Point_2D(x,y))
            if one_uv >= 0 and u >= 0 and v >= 0:
                z = one_uv*v0.z + u * v1.z + v * v2.z
            
                if z > zbuffer[x][y]:
                    zbuffer[x][y] = z

                    if None in [p0, p1, p2, texture_image]:
                        color = Vector_3D(255, 255, 255)
                    else:
                        # Get texture color
                        p_texture = one_uv * p0 + u * p1 + v * p2
                        color = Vector_3D(*get_texture_color(texture_image, p_texture.x, p_texture.y))
                    
                    image.set(x, y, (color * shading_factor) // 1)
    return image

def barycentric(p0:Point_2D, p1:Point_2D, p2:Point_2D, P:Point_2D):
    (u, v, r) = cross_product(Vector_3D(p1.x-p0.x, p2.x-p0.x, p0.x-P.x), Vector_3D(p1.y-p0.y, p2.y-p0.y, p0.y-P.y))
    
    if r == 0:
        # Triangle is degenerated
        return (-1,-1,-1)
    else:
        # Component r should be 1: Normalize components 
        return (1-(u+v)/r, u/r, v/r)

def model(bounds_min: Vector_3D, bounds_max: Vector_3D):
    bounds_delta = bounds_max - bounds_min
    bounds_scale = 2.0 / max(bounds_delta.x, bounds_delta.y, bounds_delta.z)
    bounds_offset = (bounds_max + bounds_min) * bounds_scale / 2

    M_model = Matrix_4D([[bounds_scale, 0,            0,            -bounds_offset.x], 
                         [0,            bounds_scale, 0,            -bounds_offset.y],
                         [0,            0,            bounds_scale, -bounds_offset.z],
                         [0,            0,            0,            1               ]])
    return M_model

def lookat(eye: Vector_3D, center: Vector_3D, up: Vector_3D):
    z = (eye - center).norm()
    x = cross_product(up, z).norm()
    y = cross_product(z, x).norm()

    M_inv = Matrix_4D([[x.x, x.y, x.z, 0], 
                       [y.x, y.y, y.z, 0],
                       [z.x, z.y, z.z, 0],
                       [0  , 0  , 0  , 1]])

    M_tr = Matrix_4D([[1, 0, 0, -center.x], 
                      [0, 1, 0, -center.y],
                      [0, 0, 1, -center.z],
                      [0, 0, 0, 1        ]])

    M_lookat = M_inv * M_tr

    return M_lookat

def perspective(c: float):
    M_perspective = Matrix_4D([[1, 0, 0,    0],
                               [0, 1, 0,    0],
                               [0, 0, 1,    0],
                               [0, 0, -1/c, 1]])
    
    return M_perspective

def viewport(o_x, o_y, w, h, d):
    M_viewport = Matrix_4D([[w/2, 0,   0  , o_x + w/2],
                            [0  , h/2, 0  , o_y + h/2],
                            [0  , 0,   d/2, d/2      ],
                            [0  , 0,   0  , 1        ]])
    return M_viewport

def normal_transformation(M_transform: Matrix_4D):
    return M_transform.tr().inv()