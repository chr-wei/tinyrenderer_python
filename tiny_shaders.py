"""This modules contains all tiny shaders. Shaders all have the
   same structure maintained by an abstract base class in our_gl module."""

import math
import our_gl as gl
from geom import Matrix4D, Matrix3D, MatrixUV, \
                 Vector4DType, Vector3D, Barycentric, PointUV, \
                 transform_3D4D3D, transform_vertex_to_screen, \
                 comp_min

from model import ModelStorage, NormalMapType

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

class TinyShader(gl.Shader):
    """A tiny shader with all techniques applied."""
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

        # Get diffuse color and apply ambient occlusion intensity
        ao_intensity = self.mdl.get_ao_intensity_from_map(p_uv)
        color = self.mdl.get_diffuse_color(p_uv)
        color *= ao_intensity

        # Combine base, diffuse and specular intensity
        color = 20 * Vector3D(1,1,1) + \
            color * shadowed_intensity * (1.8 * diffuse_intensity + .6 * specular_intensity)
        color = comp_min(Vector3D(255, 255, 255), color) // 1

        # Do not discard pixel and return color
        return (False, color)
