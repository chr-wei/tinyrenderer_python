"""This modules contains all tiny shaders. Shaders all have the
   same structure maintained by an abstract base class in our_gl module."""

import math
import our_gl as gl
from geom import Matrix4D, Matrix3D, MatrixUV, \
                 Vector4DType, Vector3D, Barycentric, PointUV, \
                 transform_3D4D3D, transform_vertex_to_screen, \
                 cross_product, comp_min

from model import ModelStorage, NormalMapType

class FlatShader(gl.Shader):
    """Shader which creates a paperfold effect."""
    mdl: ModelStorage

    # Vertices are stored row-wise
    varying_vert = Matrix3D(9*[0])

    varying_n_tri: Vector3D
    uniform_l_global: Vector3D
    uniform_M: Matrix4D

    def __init__(self, mdl, light_dir, M):
        self.mdl = mdl
        self.uniform_l_global = light_dir
        self.uniform_M = M # pylint: disable=invalid-name

    def vertex(self, face_idx: int, vert_idx: int):
        vert = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        self.varying_vert = self.varying_vert.set_row(vert_idx, vert)

        # Get normal vector of triangle; should only be done at last vertex
        if vert_idx == 2:
            v_0 = Vector3D(self.varying_vert.get_row(0))
            v_1 = Vector3D(self.varying_vert.get_row(1))
            v_2 = Vector3D(self.varying_vert.get_row(2))
            self.varying_n_tri = cross_product(v_2 - v_0, v_0 - v_1).normalize()

        return transform_vertex_to_screen(vert, self.uniform_M) # Transform it to screen coordinates

    def fragment(self, bary: Barycentric):
        cos_phi = max(0, self.varying_n_tri.tr() * self.uniform_l_global)

        color = (Vector3D(255, 255, 255) * cos_phi) // 1
        return (False, color) # Do not discard pixel and return color

class GouraudShader(gl.Shader):
    """Shader which interpolates normals of triangle vertices."""
    mdl: ModelStorage

    # Written by vertex shader, read by fragment shader
    varying_intensity = Vector3D(0, 0, 0, shape = (1,3))
    uniform_l_global: Vector3D
    uniform_M: Matrix4D

    def __init__(self, mdl, light_dir, M):
        self.mdl = mdl
        self.uniform_l_global = light_dir
        self.uniform_M = M # pylint: disable=invalid-name

    def vertex(self, face_idx: int, vert_idx: int):
        n_tri = self.mdl.get_normal(face_idx, vert_idx)

        # Get diffuse lighting intensity
        cos_phi = max(0, n_tri.tr() * self.uniform_l_global)
        self.varying_intensity = self.varying_intensity.set_col(vert_idx, cos_phi)

         # Read the vertex data and return
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M)

    def fragment(self, bary: Barycentric):
        intensity = self.varying_intensity * bary

        # Interpolate intensity for the current pixel
        color = (Vector3D(255, 255, 255) * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class GouraudShaderSegregated(gl.Shader):
    """Gouraud shader with distinct, segregated grey tones."""
    mdl: ModelStorage
    varying_intensity = Vector3D(0, 0, 0, shape = (1,3))
    uniform_l_global: Vector3D
    uniform_M: Matrix4D
    segregate_count = 1

    def __init__(self, mdl, light_dir, M, segregate_count):
        self.mdl = mdl
        self.uniform_l_global = light_dir
        self.uniform_M = M # pylint: disable=invalid-name
        self.segregate_count = segregate_count

    def vertex(self, face_idx: int, vert_idx: int):
        n_tri = self.mdl.get_normal(face_idx, vert_idx)

        # Get diffuse lighting intensity
        cos_phi = max(0, n_tri.tr() * self.uniform_l_global)
        self.varying_intensity = self.varying_intensity.set_col(vert_idx, cos_phi)

        vert = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vert, self.uniform_M)

    def fragment(self, bary: Barycentric):
        # Interpolate intensity for current pixel
        intensity = self.varying_intensity * bary

        # Segregates intensity values to n = 'segregate_count' distinct values
        intensity = round(intensity * self.segregate_count) / self.segregate_count

        color = (Vector3D(255, 255, 255) * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class DiffuseGouraudShader(gl.Shader):
    """Shader which combines Gouraud shading and diffuse texture color."""
    mdl: ModelStorage

    # Written by vertex shader, read by fragment shader: varying_data
    varying_intensity = Vector3D(0, 0, 0, shape = (1,3))
    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = MatrixUV(6*[0])

    uniform_l_global: Vector3D
    uniform_M: Matrix4D

    def __init__(self, mdl, light_dir, M):
        self.mdl = mdl
        self.uniform_l_global = light_dir
        self.uniform_M = M # pylint: disable=invalid-name

    def vertex(self, face_idx: int, vert_idx: int):
        n_tri = self.mdl.get_normal(face_idx, vert_idx)

         # Get diffuse lighting intensity
        cos_phi = max(0, n_tri.tr() * self.uniform_l_global)
        self.varying_intensity = self.varying_intensity.set_col(vert_idx, cos_phi)

        # Read the vertex
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Get uv map point for diffuse color interpolation and store it
        self.varying_uv = \
            self.varying_uv.set_col(vert_idx, self.mdl.get_uv_map_point(face_idx, vert_idx))

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M)

    def fragment(self, bary: Barycentric):
        # Interpolate intensity for the current pixel
        intensity = self.varying_intensity * bary

        # For interpolation with bary coordinates we need a 2 rows x 3 columns matrix
        p_uv = PointUV(self.varying_uv * bary)

        color = self.mdl.get_diffuse_color(p_uv)
        color = (color * intensity) // 1

        # Do not discard pixel and return color
        return (False, color)

class GlobalNormalmapShader(gl.Shader):
    """Shader reading a global space normal map to increase detail."""
    mdl: ModelStorage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = MatrixUV(6*[0])

    uniform_light_dir: Vector3D
    uniform_M_pe: Matrix4D
    uniform_M_sc: Matrix4D
    uniform_M_pe_IT: Matrix4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.GLOBAL:
            raise ValueError("Only use global space normalmaps with this shader")

        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe # pylint: disable=invalid-name
        self.uniform_M_sc = M_sc # pylint: disable=invalid-name
        self.uniform_M_pe_IT = M_pe_IT # pylint: disable=invalid-name

    def vertex(self, face_idx: int, vert_idx: int):
        # Read the vertex
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Get uv map point for diffuse color interpolation and store it
        self.varying_uv = \
            self.varying_uv.set_col(vert_idx, self.mdl.get_uv_map_point(face_idx, vert_idx))

         # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M_sc)

    def fragment(self, bary: Barycentric):
        # For interpolation with bary coordinates we need a 2 rows x 3 columns matrix
        p_uv = PointUV(self.varying_uv * bary)

        n_global = self.mdl.get_normal_from_map(p_uv)
        n_local = transform_3D4D3D(n_global, Vector4DType.DIRECTION, \
            self.uniform_M_pe_IT).normalize()
        l_local = transform_3D4D3D(self.uniform_light_dir, Vector4DType.DIRECTION,
                             self.uniform_M_pe).normalize()

        # Get diffuse lighting intensity
        cos_phi = max(0, n_local.tr() * l_local)

        color = self.mdl.get_diffuse_color(p_uv)
        color = (color * cos_phi) // 1

        # Do not discard pixel and return color
        return (False, color)

class SpecularmapShader(gl.Shader):
    """Shader combining global normal map shading and specular lighting."""
    mdl: ModelStorage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = MatrixUV(6*[0])

    uniform_light_dir: Vector3D
    uniform_M_pe: Matrix4D
    uniform_M_sc: Matrix4D
    uniform_M_pe_IT: Matrix4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.GLOBAL:
            raise ValueError("Only use global space normalmaps with this shader")

        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe # pylint: disable=invalid-name
        self.uniform_M_sc = M_sc # pylint: disable=invalid-name
        self.uniform_M_pe_IT = M_pe_IT # pylint: disable=invalid-name

    def vertex(self, face_idx: int, vert_idx: int):
        # Read the vertex
        vertex = self.mdl.get_vertex(face_idx, vert_idx)

        # Get uv map point for diffuse color interpolation and store it
        self.varying_uv = \
            self.varying_uv.set_col(vert_idx, self.mdl.get_uv_map_point(face_idx, vert_idx))

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vertex, self.uniform_M_sc)

    def fragment(self, bary: Barycentric):
        # For interpolation with bary coordinates we need a 2 rows x 3 columns matrix
        p_uv = PointUV(self.varying_uv * bary)

        n_global = self.mdl.get_normal_from_map(p_uv)
        n_local = transform_3D4D3D(n_global, Vector4DType.DIRECTION, \
            self.uniform_M_pe_IT).normalize()

        l_local = transform_3D4D3D(self.uniform_light_dir, Vector4DType.DIRECTION, \
            self.uniform_M_pe).normalize()
        cos_phi = n_local.tr() * l_local

        # Get diffuse lighting intensity
        diffuse_intensity = max(0, cos_phi)

        # Reflected light direction (already transformed as n and l got transformed)
        reflect = (2 * (cos_phi) * n_local - l_local).normalize()
        cos_r_z = max(0, reflect.z) # equals: reflect.tr() * Vector3D(0, 0, 1) == reflect.z
        specular_intensity = math.pow(cos_r_z, self.mdl.get_specular_power_from_map(p_uv))

        color = self.mdl.get_diffuse_color(p_uv)

        # Combine base, diffuse and specular intensity
        color = 10 * Vector3D(1, 1, 1) + (diffuse_intensity + 0.5 * specular_intensity) * color
        color = comp_min(Vector3D(255, 255, 255), color) // 1

        # Do not discard pixel and return color
        return (False, color)

class TangentNormalmapShader(gl.Shader):
    """Shader equal to GlobalNormalmapShader but with tangent space normal map."""
    mdl: ModelStorage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = MatrixUV(6*[0])

    # Contains precalculated info of varying_vert to save ops in fragment shader
    varying_vert = Matrix3D(9*[0])
    varying_A = Matrix3D(9*[0])

    varying_normal = Matrix3D(9*[0])
    varying_b_u: Vector3D
    varying_b_v: Vector3D

    uniform_l_local: Vector3D
    uniform_M_pe: Matrix4D
    uniform_M_sc: Matrix4D
    uniform_M_pe_IT: Matrix4D

    def __init__(self, mdl, light_dir, M_pe, M_pe_IT, M_viewport):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.TANGENT:
            raise ValueError("Only use only tangent space normalmaps with this shader")

        # Transform light vector
        l_local = transform_3D4D3D(light_dir, Vector4DType.DIRECTION, M_pe)
        self.uniform_l_local = l_local.normalize()

        self.uniform_M_pe = M_pe # pylint: disable=invalid-name
        self.uniform_M_pe_IT = M_pe_IT # pylint: disable=invalid-name
        self.uniform_M_viewport = M_viewport # pylint: disable=invalid-name

    def vertex(self, face_idx: int, vert_idx: int):
        # Store triangle vertex (after being transformed to perspective)
        vert = self.mdl.get_vertex(face_idx, vert_idx)
        vert = transform_3D4D3D(vert, Vector4DType.POINT, self.uniform_M_pe)
        self.varying_vert = self.varying_vert.set_col(vert_idx, vert)

        n_tri_global = self.mdl.get_normal(face_idx, vert_idx) # Read normal of the vertex
        n_tri_local = transform_3D4D3D(n_tri_global, Vector4DType.DIRECTION, \
            self.uniform_M_pe_IT).normalize()

        # Store triangle vertex normal (after being transformed to perspective)
        self.varying_normal = self.varying_normal.set_col(vert_idx, n_tri_local)

        # Get uv map point for diffuse color interpolation and store it
        self.varying_uv = \
            self.varying_uv.set_col(vert_idx, self.mdl.get_uv_map_point(face_idx, vert_idx))

        if vert_idx == 2:
            self.varying_b_u = Vector3D(self.varying_uv.u_2 - self.varying_uv.u_0, \
                                         self.varying_uv.u_1 - self.varying_uv.u_0, \
                                         0)

            self.varying_b_v = Vector3D(self.varying_uv.v_2 - self.varying_uv.v_0, \
                                         self.varying_uv.v_1 - self.varying_uv.v_0, \
                                         0)

            vd_0 = self.varying_vert.get_col(2) - self.varying_vert.get_col(0)
            vd_1 = self.varying_vert.get_col(1) - self.varying_vert.get_col(0)
            self.varying_A = (self.varying_A.set_row(0, vd_0)).set_row(1, vd_1) # pylint: disable=invalid-name

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vert, self.uniform_M_viewport)

    def fragment(self, bary: Barycentric):
        p_uv = PointUV(self.varying_uv * bary)

        n_bary = (self.varying_normal * bary).normalize()

        A_inv = self.varying_A.set_row(2, n_bary).inv() # pylint: disable=invalid-name

        vect_i = (A_inv * self.varying_b_u).normalize()
        vect_j = (A_inv * self.varying_b_v).normalize()

        B = Matrix3D([vect_i, # pylint: disable=invalid-name
                      vect_j,
                      n_bary]).tr()

         # Load normal of tangent space and transform to get global normal
        n_local = (B * self.mdl.get_normal_from_map(p_uv)).normalize()

        # Get diffuse lighting intensity
        cos_phi = max(0, n_local.tr() * self.uniform_l_local)

        color = self.mdl.get_diffuse_color(p_uv)
        color = cos_phi * color // 1

        # Do not discard pixel and return color
        return (False, color)

class DepthShader(gl.Shader):
    """Shader used to save shadow buffer."""
    mdl: ModelStorage

    # Vertices are stored col-wise
    varying_vert = Matrix3D(9*[0])

    uniform_M_sb: Matrix4D

    def __init__(self, mdl, M_sb, depth_res):
        self.mdl = mdl
        self.uniform_M_sb = M_sb # pylint: disable=invalid-name
        self.uniform_depth_res = depth_res

    def vertex(self, face_idx: int, vert_idx: int):
        vert = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        vert = transform_vertex_to_screen(vert, self.uniform_M_sb)
        self.varying_vert = self.varying_vert.set_col(vert_idx, vert)
        return vert

    def fragment(self, bary: Barycentric):
        v_bary = Vector3D(self.varying_vert * bary)
        color = (Vector3D(255, 255, 255) * v_bary.z / self.uniform_depth_res) // 1
        return (False, color) # Do not discard pixel and return color

class SpecularShadowShader(gl.Shader):
    """Shader combining global normal map shading, specular lighting and shadows."""
    mdl: ModelStorage

    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = MatrixUV(6*[0])
    varying_vert = Matrix3D(9*[0])

    uniform_light_dir: Vector3D
    uniform_M_pe: Matrix4D
    uniform_M_sc: Matrix4D
    uniform_M_pe_IT: Matrix4D
    uniform_M_sb_inv: Matrix4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT, M_sb, shadow_buffer):
        self.mdl = mdl
        if self.mdl.normal_map_type != NormalMapType.GLOBAL:
            raise ValueError("Only use global space normalmaps with this shader")

        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe # pylint: disable=invalid-name
        self.uniform_M_sc = M_sc # pylint: disable=invalid-name
        self.uniform_M_pe_IT = M_pe_IT # pylint: disable=invalid-name
        self.uniform_M_sb = M_sb # pylint: disable=invalid-name
        self.shadow_buffer = shadow_buffer

    def vertex(self, face_idx: int, vert_idx: int):
        # Read the vertex
        vert = self.mdl.get_vertex(face_idx, vert_idx)
        self.varying_vert = self.varying_vert.set_col(vert_idx, vert)
        
        # Get uv map point for diffuse color interpolation and store it
        self.varying_uv = \
            self.varying_uv.set_col(vert_idx, self.mdl.get_uv_map_point(face_idx, vert_idx))

        # Transform it to screen coordinates
        return transform_vertex_to_screen(vert, self.uniform_M_sc)

    def fragment(self, bary: Barycentric):
        # For interpolation with bary coordinates we need a 2 rows x 3 columns matrix
        p_uv = PointUV(self.varying_uv * bary)

        # Backproject interpolated point to get shadow buffer coordinates
        v_bary = Vector3D(self.varying_vert * bary)
        p_shadow = transform_vertex_to_screen(v_bary, self.uniform_M_sb)
        z_lit = self.shadow_buffer[p_shadow.x][p_shadow.y]

        shadowed_intensity = .3 if p_shadow.z + .02 * 255 < z_lit else 1.0

        n_global = self.mdl.get_normal_from_map(p_uv)
        n_local = transform_3D4D3D(n_global, Vector4DType.DIRECTION, \
            self.uniform_M_pe_IT).normalize()

        l_local = transform_3D4D3D(self.uniform_light_dir, Vector4DType.DIRECTION, \
            self.uniform_M_pe).normalize()
        cos_phi = n_local.tr() * l_local

        # Get diffuse lighting intensity
        diffuse_intensity = max(0, cos_phi)

        # Reflected light direction (already transformed as n and l got transformed)
        reflect = (2 * (cos_phi) * n_local - l_local).normalize()
        cos_r_z = max(0, reflect.z) # equals: reflect.tr() * Vector3D(0, 0, 1) == reflect.z
        specular_intensity = math.pow(cos_r_z, self.mdl.get_specular_power_from_map(p_uv))
        

        color = self.mdl.get_diffuse_color(p_uv)

        # Combine base, diffuse and specular intensity
        color = 20 * Vector3D(1,1,1) + color * shadowed_intensity * (1.2 * diffuse_intensity + .6 * specular_intensity)
        color = comp_min(Vector3D(255, 255, 255), color) // 1

        # Do not discard pixel and return color
        return (False, color)