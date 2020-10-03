"""This modules contains all tiny shaders. Shaders all have the
   same structure maintained by an abstract base class in our_gl module."""

import math
import our_gl as gl
from geom import Matrix_4D, Matrix_3D, Matrix_uv, \
                 Vector_4D_Type, Vector_3D, Barycentric, Point_2D, Point_UV, \
                 transform_3D4D3D, transform_vertex_to_screen, \
                 cross_product, matmul, transpose, \
                 comp_min, unpack_nested_iterable_to_list

from model import Model_Storage, NormalMapType

class FlatShader(gl.Shader):
    """Shader which creates a paperfold effect."""
    mdl: Model_Storage
    varying_vertex = [Vector_3D(0,0,0)] * 3 # Written by vertex shader, read by fragment shader
    light_dir: Vector_3D
    M: Matrix_4D
    n: Vector_3D

    def __init__(self, mdl, light_dir, M):
        self.mdl = mdl
        self.light_dir = light_dir
        self.M = M

    def vertex(self, face_idx: int, vert_idx: int):
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        self.varying_vertex[vert_idx] = vertex

        # Get normal vector of triangle; should only be done at last vertex
        if vert_idx == 2:
            self.n = cross_product(self.varying_vertex[2] - self.varying_vertex[0],
                                   self.varying_vertex[0] - self.varying_vertex[1]).norm()
        else:
            self.n = None

        return transform_vertex_to_screen(vertex, self.M) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        if self.n is None:
            return(True, None) # discard pixel
        else:
            cos_phi = self.n.tr() * self.light_dir
            cos_phi = 0 if cos_phi < 0 else cos_phi

            color = (Vector_3D(255, 255, 255) * cos_phi) // 1
            return (False, color) # Do not discard pixel and return color

class GouraudShader(gl.Shader):
    """Shader which interpolates normals of triangle vertices."""
    mdl: Model_Storage
    varying_intensity = [None] * 3 # Written by vertex shader, read by fragment shader
    light_dir: Vector_3D
    M: Matrix_4D

    def __init__(self, mdl, light_dir, M):
        self.mdl = mdl
        self.light_dir = light_dir
        self.M = M

    def vertex(self, face_idx: int, vert_idx: int):
        n = self.mdl.get_normal(face_idx, vert_idx)

        # Get diffuse lighting intensity
        self.varying_intensity[vert_idx] = max(0, n.tr() * self.light_dir)

         # Read the vertex data and return
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.M)

    def fragment(self, barycentric: tuple):
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2]

        # Interpolate intensity for the current pixel
        color = (Vector_3D(255, 255, 255) * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class GouraudShaderSegregated(gl.Shader):
    """Gouraud shader with distinct, segregated grey tones."""

    mdl: Model_Storage
    varying_intensity = [None] * 3 # Written by vertex shader, read by fragment shader
    light_dir: Vector_3D
    M: Matrix_4D
    segregate_count = 1

    def __init__(self, mdl, light_dir, M, segregate_count):
        self.mdl = mdl
        self.light_dir = light_dir
        self.M = M
        self.segregate_count = segregate_count

    def vertex(self, face_idx: int, vert_idx: int):
        n = self.mdl.get_normal(face_idx, vert_idx)

        # Get diffuse lighting intensity
        self.varying_intensity[vert_idx] = max(0, n.tr() * self.light_dir)
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.M)

    def fragment(self, barycentric: tuple):
        # Interpolate intensity for current pixel
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2]

        # Segregates intensity values to n = 'segregate_count' distinct values
        intensity = round(intensity * self.segregate_count) / self.segregate_count

        color = (Vector_3D(255, 255, 255) * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class DiffuseGouraudShader(gl.Shader):
    """Shader which combines Gouraud shading and diffuse texture color."""
    mdl: Model_Storage

    # Written by vertex shader, read by fragment shader: varying_data
    varying_intensity = [None] * 3
    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = [Point_2D(0,0)] * 3
    varying_uv_shape = (3,2)

    light_dir: Vector_3D
    M: Matrix_4D

    def __init__(self, mdl, light_dir, M):
        self.mdl = mdl
        self.light_dir = light_dir
        self.M = M

    def vertex(self, face_idx: int, vert_idx: int):
        n = self.mdl.get_normal(face_idx, vert_idx)

         # Get diffuse lighting intensity
        self.varying_intensity[vert_idx] = max(0, n.tr() * self.light_dir)

        # Read the vertex
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Get uv map point for diffuse color interpolation
        self.varying_uv[vert_idx] = self.mdl.get_uv_map_point(face_idx, vert_idx)

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.M)

    def fragment(self, barycentric: tuple):
        # Interpolate intensity for the current pixel
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2]

        # For interpolation with barycentric coordinates we need a 2 rows x 3 columns matrix
        transposed_uv, tr_uv_shape = transpose(self.varying_uv, self.varying_uv_shape)
        p_uv, _ = matmul(transposed_uv, tr_uv_shape, barycentric, (3,1))

        p_uv = Point_2D(*p_uv)

        color = self.mdl.get_diffuse_color(p_uv.x, p_uv.y)
        color = (color * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class GlobalNormalmapShader(gl.Shader):
    """Shader reading a global space normal map to increase detail."""
    mdl: Model_Storage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = [Point_2D(0,0)] * 3

    uniform_light_dir: Vector_3D
    uniform_M_pe: Matrix_4D
    uniform_M_sc: Matrix_4D
    uniform_M_pe_IT: Matrix_4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.GLOBAL:
            raise ValueError

        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe
        self.uniform_M_sc = M_sc
        self.uniform_M_pe_IT = M_pe_IT

    def vertex(self, face_idx: int, vert_idx: int):
        # Read the vertex
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Get uv map point for diffuse color interpolation
        self.varying_uv[vert_idx] = self.mdl.get_uv_map_point(face_idx, vert_idx)

         # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M_sc)

    def fragment(self, barycentric: tuple):
        # For interpolation with barycentric coordinates we need a 2 rows x 3 columns matrix
        transposed_uv, tr_uv_shape = transpose(self.varying_uv, (3,2))
        p_uv, _ = matmul(transposed_uv, tr_uv_shape, barycentric, (3,1))

        p_uv = Point_2D(*p_uv)

        n = self.mdl.get_normal_from_map(p_uv.x, p_uv.y)
        n = transform_3D4D3D(n, Vector_4D_Type.DIRECTION, self.uniform_M_pe_IT).norm()
        l = transform_3D4D3D(self.uniform_light_dir, Vector_4D_Type.DIRECTION,
                             self.uniform_M_pe).norm()

        # Get diffuse lighting intensity
        intensity = max(0, n.tr() * l)

        color = self.mdl.get_diffuse_color(p_uv.x, p_uv.y)
        color = (color * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class SpecularmapShader(gl.Shader):
    """Shader combining global normal map shading and specular lighting."""
    mdl: Model_Storage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = [Point_2D(0,0)] * 3

    uniform_light_dir: Vector_3D
    uniform_M_pe: Matrix_4D
    uniform_M_sc: Matrix_4D
    uniform_M_pe_IT: Matrix_4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.GLOBAL:
            raise ValueError

        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe
        self.uniform_M_sc = M_sc
        self.uniform_M_pe_IT = M_pe_IT

    def vertex(self, face_idx: int, vert_idx: int):
        # Read the vertex
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Get uv map point for diffuse color interpolation
        self.varying_uv[vert_idx] = self.mdl.get_uv_map_point(face_idx, vert_idx)

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M_sc)

    def fragment(self, barycentric: tuple):
        # For interpolation with barycentric coordinates we need a 2 rows x 3 columns matrix
        transposed_uv, tr_uv_shape = transpose(self.varying_uv, (3,2))
        p_uv, _ = matmul(transposed_uv, tr_uv_shape, barycentric, (3,1))

        n = self.mdl.get_normal_from_map(*p_uv)
        n = transform_3D4D3D(n, Vector_4D_Type.DIRECTION, self.uniform_M_pe_IT).norm()
        l = transform_3D4D3D(self.uniform_light_dir, Vector_4D_Type.DIRECTION, self.uniform_M_pe).norm()
        n_l = n.tr() * l

        # Get diffuse lighting intensity
        diffuse_intensity = max(0, n_l)

        # Reflected light direction (already transformed as n and l got transformed)
        reflect = (2 * (n_l) * n - l).norm()
        cos_r_z = max(0, reflect.z) # equals: reflect.tr() * Vector_3D(0, 0, 1) == reflect.z
        specular_intensity = math.pow(cos_r_z, self.mdl.get_specular_power_from_map(*p_uv))

        color = self.mdl.get_diffuse_color(*p_uv)

        # Combine base, diffuse and specular intensity
        color = 10 * Vector_3D(1, 1, 1) + (diffuse_intensity + 0.5 * specular_intensity) * color
        color = comp_min(Vector_3D(255, 255, 255), color) // 1

        # Do not discard pixel and return color
        return (False, color)

class TangentNormalmapShader(gl.Shader):
    """Shader equal to GlobalNormalmapShader but with tangent space normal map."""
    mdl: Model_Storage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = Matrix_uv([0]*6)
    varying_vert = [Vector_3D(0,0,0)] * 3
    varying_nvert = [Vector_3D(0,0,0)] * 3

    uniform_light_dir: Vector_3D
    uniform_M_pe: Matrix_4D
    uniform_M_sc: Matrix_4D
    uniform_M_pe_IT: Matrix_4D

    def __init__(self, mdl, light_dir, M_pe, M_pe_IT, M_viewport):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.TANGENT:
            raise ValueError("Only use tangent space normalmaps with this shader")

        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe
        self.uniform_M_pe_IT = M_pe_IT
        self.uniform_M_viewport = M_viewport

    def vertex(self, face_idx: int, vert_idx: int):
        # Store triangle vertex (after being transformed to perspective)
        vertex = self.mdl.get_vertex(face_idx, vert_idx)
        vertex = transform_3D4D3D(vertex, Vector_4D_Type.POINT, self.uniform_M_pe)
        self.varying_vert[vert_idx] = vertex

        nvert = self.mdl.get_normal(face_idx, vert_idx) # Read normal of the vertex
        nvert = transform_3D4D3D(nvert, Vector_4D_Type.DIRECTION, self.uniform_M_pe_IT).norm()

        # Store triangle vertex normal (after being transformed to perspective)
        self.varying_nvert[vert_idx] = nvert

        # Get uv map point for diffuse color interpolation and store it
        self.varying_uv = \
            self.varying_uv.set_col(vert_idx, self.mdl.get_uv_map_point(face_idx, vert_idx))

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M_viewport)

    def fragment(self, barycentric: tuple):
        bary = Barycentric(barycentric)
        p = self.varying_uv * bary
        p_uv = Point_UV(*p)

        transposed_nvert, tr_nvert_shape = \
            transpose(unpack_nested_iterable_to_list(self.varying_nvert), (3,3))

        n_bary, _ = matmul(transposed_nvert, tr_nvert_shape, barycentric, (3,1))
        n_bary = Vector_3D(*n_bary).norm()

        A_inv = Matrix_3D([self.varying_vert[2] - self.varying_vert[0],
                           self.varying_vert[1] - self.varying_vert[0],
                           n_bary                                     ]).inv()

        b_u = Vector_3D(self.varying_uv.u2 - self.varying_uv.u0, self.varying_uv.u1 - self.varying_uv.u0, 0)
        b_v = Vector_3D(self.varying_uv.v2 - self.varying_uv.v0, self.varying_uv.v1 - self.varying_uv.v0, 0)

        i = (A_inv * b_u).norm()
        j = (A_inv * b_v).norm()

        B = Matrix_3D([i     ,
                       j     ,
                       n_bary]).tr()

        n = (B * self.mdl.get_normal_from_map(*p_uv)).norm() # Load normal of tangent space and multiply
        
        l = transform_3D4D3D(self.uniform_light_dir, Vector_4D_Type.DIRECTION, self.uniform_M_pe).norm()
        n_l = n.tr() * l
        diffuse_intensity = max(0, n_l) # Get diffuse lighting intensity

        color = Vector_3D(255,255,255)#self.mdl.get_diffuse_color(*p_uv)
        color = diffuse_intensity * color // 1
        
        return (False, color) # Do not discard pixel and return color
