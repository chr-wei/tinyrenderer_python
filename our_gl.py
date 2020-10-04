"""our_gl module mimics OpenGL functionality. Specifiy shader implementations
   have to be made externally based on provided abstract class Shader.
"""
from operator import attrgetter
from abc import ABC, abstractmethod

from tiny_image import TinyImage

from geom import Matrix4D, Vector3D, Barycentric, Point2D, cross_product

class Shader(ABC):
    """Abstract class for tiny shaders."""
    @abstractmethod
    def vertex(self, face_idx: int, vert_idx: int):
        """Vertex shader modifies vertices of mesh. Returns four-dimensional vector."""

    @abstractmethod
    def fragment(self, bary: Barycentric): # Returns bool and color
        """Fragment shader modifies single image pixels. Returns color."""

def draw_line(p_0, p_1, image, color):
    """Draw p_0 line onto an image."""

    (x_0, y_0) = p_0
    (x_1, y_1) = p_1

    if abs(x_1-x_0) < abs(y_1-y_0):
        # Swap to prevent whitespace when y distance is higher than x distance (steep line)
        steep_line = True
        (y_1, y_0, x_1, x_0) = (x_1, x_0, y_1, y_0)
    else:
        steep_line = False

    if x_0 == x_1:
        # Due to switching steep lines this only occurs if y_0 == y_1 and x_0 == x_1
        # Only draw p_0 dot in this case
        image.set(x_0, y_0, color)
        return image

    elif x_0 > x_1:
        (y_1, y_0, x_1, x_0) = (y_0, y_1, x_0, x_1)

    for x_sweep in range(x_0, x_1+1):
        y_sweep = int(y_0 + (y_1 - y_0) / (x_1 - x_0) * (x_sweep - x_0) + .5)

        if steep_line:
            image.set(y_sweep, x_sweep, color)
        else:
            image.set(x_sweep, y_sweep, color)

    return image

def draw_triangle_edges(p_0, p_1, p_2, image, color):
    """Draws triangle lines for three given points onto image."""
    image = draw_line(p_0, p_1, image, color)
    image = draw_line(p_1, p_2, image, color)
    image = draw_line(p_2, p_0, image, color)
    return image

def draw_triangle(screen_coords: list, shader: Shader, zbuffer: list, image: TinyImage):
    """Base method of rasterizer which calls fragment shader."""
    temp_coords = screen_coords.copy()
    temp_coords.sort(key=attrgetter('x'))
    x_min = int(min(max(temp_coords[0].x, 0), image.width - 1))
    x_max = int(min(max(temp_coords[2].x, 0), image.width - 1))

    temp_coords.sort(key=attrgetter('y'))
    y_min = int(min(max(temp_coords[0].y, 0), image.height - 1))
    y_max = int(min(max(temp_coords[2].y, 0), image.height - 1))

    for image_x in range(x_min, x_max):
        for image_y in range(y_min, y_max):
            bary = barycentric(Point2D(screen_coords[0].x, screen_coords[0].y),
                               Point2D(screen_coords[1].x, screen_coords[1].y),
                               Point2D(screen_coords[2].x, screen_coords[2].y),
                               Point2D(image_x, image_y))

            (one_uv_bary, u_bary, v_bary) = bary

            if one_uv_bary >= 0 and u_bary >= 0 and v_bary >= 0:
                z_comp = one_uv_bary * screen_coords[0].z + u_bary \
                         * screen_coords[1].z + v_bary * screen_coords[2].z

                if z_comp > zbuffer[image_x][image_y]:
                    zbuffer[image_x][image_y] = z_comp

                    discard, color = shader.fragment(bary)
                    if not discard:
                        image.set(image_x, image_y, color)
    return image

def barycentric(p_0: Point2D, p_1: Point2D, p_2: Point2D, p_tri: Point2D):
    """Returns barycentric coordinates for three given triangle points and fourth point
       located relative to the triangle points.
    """

    (u_bary, v_bary, r_bary) = cross_product(Vector3D(p_1.x-p_0.x, p_2.x-p_0.x, p_0.x-p_tri.x), \
        Vector3D(p_1.y-p_0.y, p_2.y-p_0.y, p_0.y-p_tri.y))

    if r_bary == 0:
        # Triangle is degenerated
        return (-1,-1,-1)

    # Component r_bary should be 1: Normalize components
    return (1 - (u_bary + v_bary) / r_bary, u_bary / r_bary, v_bary / r_bary)

def model_transform(bounds_min: Vector3D, bounds_max: Vector3D):
    """Returns transformation matrix of model. .obj data is scaled and offset to
       appear to be x, y, z element of [-1, 1].
    """

    bounds_delta = bounds_max - bounds_min
    bounds_scale = 2.0 / max(bounds_delta.x, bounds_delta.y, bounds_delta.z)
    bounds_offset = (bounds_max + bounds_min) * bounds_scale / 2

    M_model = Matrix4D([[bounds_scale, 0,            0,            -bounds_offset.x], # pylint: disable=invalid-name
                        [0,            bounds_scale, 0,            -bounds_offset.y],
                        [0,            0,            bounds_scale, -bounds_offset.z],
                        [0,            0,            0,            1               ]])
    return M_model

def lookat(eye: Vector3D, center: Vector3D, up_dir: Vector3D):
    """Returns transformatoin matrix for view (OpenGl gl_ulookat)."""

    z_v = (eye - center).normalize()
    x_v = cross_product(up_dir, z_v).normalize()
    y_v = cross_product(z_v, x_v).normalize()

    M_inv = Matrix4D([[x_v.x, x_v.y, x_v.z, 0], # pylint: disable=invalid-name
                      [y_v.x, y_v.y, y_v.z, 0],
                      [z_v.x, z_v.y, z_v.z, 0],
                      [0    , 0    , 0    , 1]])

    M_tr = Matrix4D([[1, 0, 0, -center.x], # pylint: disable=invalid-name
                     [0, 1, 0, -center.y],
                     [0, 0, 1, -center.z],
                     [0, 0, 0, 1        ]])

    M_lookat = M_inv * M_tr # pylint: disable=invalid-name

    return M_lookat

def perspective(z_dist: float):
    """Returns transformation matrix for perspective transformation."""
    M_perspective = Matrix4D([[1, 0, 0        , 0], # pylint: disable=invalid-name
                              [0, 1, 0        , 0],
                              [0, 0, 1        , 0],
                              [0, 0, -1/z_dist, 1]])

    return M_perspective

def viewport(o_x, o_y, img_width, img_height, z_spread):
    """Returns viewport transformation. Transforms vertices to screen coordinates."""

    M_viewport = Matrix4D([[img_width/2, 0           , 0         , o_x + img_width/2 ], # pylint: disable=invalid-name
                           [0          , img_height/2, 0         , o_y + img_height/2],
                           [0          , 0           , z_spread/2, z_spread/2        ],
                           [0          , 0           , 0         , 1                 ]])
    return M_viewport

def normal_transformation(M_transform: Matrix4D): # pylint: disable=invalid-name
    """Transformation matrix for normal vectors. Retunrs inversed transpose
       of transformatoin matrix."""

    return M_transform.tr().inv()
