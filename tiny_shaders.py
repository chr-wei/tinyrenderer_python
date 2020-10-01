import our_gl as gl
from geom import Matrix_4D, Matrix_3D, Vector_3D, Point_2D, transform_vertex_to_screen, cross_product, matmul, transpose, \
                 Vector_4D_Type, transform_3D4D3D, Vector_4D, comp_min
from model import Model_Storage
import math

class Flat_Shader(gl.Shader):
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

class Gouraud_Shader(gl.Shader):
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
        self.varying_intensity[vert_idx] = max(0, n.tr() * self.light_dir) # Get diffuse lighting intensity
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        
        return transform_vertex_to_screen(vertex, self.M) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2] # Interpolate intensity for the current pixel
        color = (Vector_3D(255, 255, 255) * intensity) // 1
        return (False, color) # Do not discard pixel and return color

class Gouraud_Shader_Segregated(gl.Shader):
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
        self.varying_intensity[vert_idx] = max(0, n.tr() * self.light_dir) # Get diffuse lighting intensity
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        
        return transform_vertex_to_screen(vertex, self.M) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2] # Interpolate intensity for the current pixel
        
        # Segregates intensity values to n = 'segregate_count' distinct values
        intensity = round(intensity * self.segregate_count) / self.segregate_count
        
        color = (Vector_3D(255, 255, 255) * intensity) // 1
        return (False, color) # Do not discard pixel and return color

class Diffuse_Gouraud_Shader(gl.Shader):
    mdl: Model_Storage
    varying_intensity = [None] * 3 # Written by vertex shader, read by fragment shader
    
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
        self.varying_intensity[vert_idx] = max(0, n.tr() * self.light_dir) # Get diffuse lighting intensity
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        
        self.varying_uv[vert_idx] = self.mdl.get_uv_map_point(face_idx, vert_idx) # Get uv map point for diffuse color interpolation
        return transform_vertex_to_screen(vertex, self.M) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2] # Interpolate intensity for the current pixel

        # For interpolation with barycentric coordinates we need a 2 rows x 3 columns matrix
        transposed_uv, tr_uv_shape = transpose(self.varying_uv, self.varying_uv_shape)
        p_uv, _ = matmul(transposed_uv, tr_uv_shape, barycentric, (3,1))

        p_uv = Point_2D(*p_uv)

        color = self.mdl.get_diffuse_color(p_uv.x, p_uv.y)
        color = (color * intensity) // 1
        return (False, color) # Do not discard pixel and return color

class Normalmap_Shader(gl.Shader):
    mdl: Model_Storage
    varying_intensity = [None] * 3 # Written by vertex shader, read by fragment shader
    
    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = [Point_2D(0,0)] * 3

    uniform_light_dir: Vector_3D
    uniform_M_pe: Matrix_4D
    uniform_M_sc: Matrix_4D
    uniform_M_pe_IT: Matrix_4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT):
        self.mdl = mdl
        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe
        self.uniform_M_sc = M_sc
        self.uniform_M_pe_IT = M_pe_IT

    def vertex(self, face_idx: int, vert_idx: int):
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        
        self.varying_uv[vert_idx] = self.mdl.get_uv_map_point(face_idx, vert_idx) # Get uv map point for diffuse color interpolation
        return transform_vertex_to_screen(vertex, self.uniform_M_sc) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        # For interpolation with barycentric coordinates we need a 2 rows x 3 columns matrix
        transposed_uv, tr_uv_shape = transpose(self.varying_uv, (3,2))
        p_uv, _ = matmul(transposed_uv, tr_uv_shape, barycentric, (3,1))

        p_uv = Point_2D(*p_uv)

        n = self.mdl.get_normal_from_map(p_uv.x, p_uv.y)
        n = transform_3D4D3D(n, Vector_4D_Type.DIRECTION, self.uniform_M_pe_IT).norm()
        l = transform_3D4D3D(self.uniform_light_dir, Vector_4D_Type.DIRECTION, self.uniform_M_pe).norm()
        intensity = max(0, n.tr() * l) # Get diffuse lighting intensity

        color = self.mdl.get_diffuse_color(p_uv.x, p_uv.y)
        color = (color * intensity) // 1
        return (False, color) # Do not discard pixel and return color

class Specularmap_Shader(gl.Shader):
    mdl: Model_Storage
    varying_intensity = [None] * 3 # Written by vertex shader, read by fragment shader
    
    # Points in varying_uv are stacked row-wise, 3 rows x 2 columns
    varying_uv = [Point_2D(0,0)] * 3

    uniform_light_dir: Vector_3D
    uniform_M_pe: Matrix_4D
    uniform_M_sc: Matrix_4D
    uniform_M_pe_IT: Matrix_4D

    def __init__(self, mdl, light_dir, M_pe, M_sc, M_pe_IT):
        self.mdl = mdl
        self.uniform_light_dir = light_dir
        self.uniform_M_pe = M_pe
        self.uniform_M_sc = M_sc
        self.uniform_M_pe_IT = M_pe_IT

    def vertex(self, face_idx: int, vert_idx: int):
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        
        self.varying_uv[vert_idx] = self.mdl.get_uv_map_point(face_idx, vert_idx) # Get uv map point for diffuse color interpolation
        return transform_vertex_to_screen(vertex, self.uniform_M_sc) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        # For interpolation with barycentric coordinates we need a 2 rows x 3 columns matrix
        transposed_uv, tr_uv_shape = transpose(self.varying_uv, (3,2))
        p_uv, _ = matmul(transposed_uv, tr_uv_shape, barycentric, (3,1))

        n = self.mdl.get_normal_from_map(*p_uv)
        n = transform_3D4D3D(n, Vector_4D_Type.DIRECTION, self.uniform_M_pe_IT).norm()
        l = transform_3D4D3D(self.uniform_light_dir, Vector_4D_Type.DIRECTION, self.uniform_M_pe).norm()
        n_l = n.tr() * l
        diffuse_intensity = max(0, n_l) # Get diffuse lighting intensity

        # Reflected light direction (already transformed as n and l got transformed)
        r = (2 * (n_l) * n - l).norm()
        cos_r_z = max(0, r.z) # r.tr() * Vector_3D(0, 0, 1) == r.z
        specular_intensity = math.pow(cos_r_z, self.mdl.get_specular_power_from_map(*p_uv))

        color = self.mdl.get_diffuse_color(*p_uv)

        # Combine base, diffuse and specular intensity
        color = 10 * Vector_3D(1, 1, 1) + (diffuse_intensity + 0.5 * specular_intensity) * color
        color = comp_min(Vector_3D(255, 255, 255), color) // 1
        return (False, color) # Do not discard pixel and return color