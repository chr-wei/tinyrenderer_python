import our_gl as gl
from geom import Matrix_4D, Matrix_3D, Vector_3D, transform_vertex, cross_product
from model import Model_Storage

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
            n = cross_product(self.varying_vertex[0] - self.varying_vertex[1], 
                              self.varying_vertex[2] - self.varying_vertex[0])
            self.n = n.norm()
        else:
            n = None

        return transform_vertex(vertex, self.M) # Transform it to screen coordinates

    def fragment(self, barycentric: tuple):
        if self.n is None:
            return(True, None) # discard pixel
        else:
            cos_phi = self.n.norm() * self.light_dir.norm()
            cos_phi = 0 if cos_phi < 0 else cos_phi

            color = Vector_3D(255, 255, 255) * cos_phi // 1
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
        self.varying_intensity[vert_idx] = max(0, self.mdl.get_normal(face_idx, vert_idx)*self.light_dir) # Get diffuse lighting intensity
        vertex = self.mdl.get_vertex(face_idx, vert_idx) # Read the vertex
        
        return self.M * vertex.expand_4D_point() # Transform it to screen coordinates

    def fragment(self, barycentric: tuple, color: Vector_3D):
        intensity = self.varying_intensity[0]*barycentric[0] \
                  + self.varying_intensity[1]*barycentric[1] \
                  + self.varying_intensity[2]*barycentric[2] # Interpolate intensity for the current pixel
        color = Vector_3D(255, 255, 255)*intensity // 1
        return (False, color) # Do not discard pixel and return color
