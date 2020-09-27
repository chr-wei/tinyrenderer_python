from collections import namedtuple
import numpy as np
import math
import itertools

# Tuple definitions
BoundingBox = namedtuple("BoundingBox", "x_min y_min z_min x_max y_max z_max")

class Point_2D(namedtuple("Point_2D", "x y")):
    _shape = (2,1)
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return self.x * other.x + self.y * other.y
        elif other.__class__.__name__ in ["float", "int"]:
            return Point_2D(self.x * other, self.y * other)

    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Point_2D(self.x * other, self.y * other)

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return Point_2D(self.x + other.x, self.y + other.y)

class Vector_3D(namedtuple("Vector_3D", "x y z")):
    _shape = (3,1)
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            # Calc scalar product
            return self.x * other.x + self.y * other.y + self.z * other.z
        elif other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(self.x * other, self.y * other, self.z * other)
        
    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(self.x * other, self.y * other, self.z * other)
    
    def __truediv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(self.x / other, self.y / other, self.z / other)
    
    def __floordiv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(int(self.x // other), int(self.y // other), int(self.z // other))

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return Vector_3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return Vector_3D(self.x - other.x, self.y - other.y, self.z - other.z)

    def project_2D(self):
        return Point_2D(self.x / self.z, self.y / self.z)
    
    def expand_4D_vect(self):
        return Vector_4D(self.x, self.y, self.z, 0)
    
    def expand_4D_point(self):
        return Vector_4D(self.x, self.y, self.z, 1)

    def norm(self):
        abs = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if abs > 0:
            return self / abs
    

class Vector_4D(namedtuple("Vector_4D", "x y z a")):
    _shape = (4,1)

    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            # Calc scalar product
            return self.x * other.x + self.y * other.y + self.z * other.z + self.a * other.a
        elif other.__class__.__name__ in ["float", "int"]:
            return Vector_4D(self.x * other, self.y * other, self.z * other, self.a * other)
        
    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_4D(self.x * other, self.y * other, self.z * other, self.a * other)

    def project_3D(self):
        if self.a == 0:
            return Vector_3D(self.x, self.y, self.z)
        else:
            return Vector_3D(self.x / self.a, self.y / self.a, self.z / self.a)

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return Vector_4D(self.x + other.x, self.y + other.y, self.z + other.z, self.a + other.a)

class Matrix_3D(namedtuple("Matrix_3D", "a11 a12 a13 a21 a22 a23 a31 a32 a33")):
    _shape = (3,3)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *list(itertools.chain.from_iterable(*args)))
        else:
            return super().__new__(cls, *args, **kwargs)
    
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_3D(*coeffs)

        elif other.__class__.__name__ == "Vector_3D":
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Vector_3D(*coeffs)

    def tr(self):
        (coeffs, _) = transpose(list(self._asdict().values()), self._shape)
        return Matrix_3D(*coeffs)
    
    def inv(self):
        (coeffs, _) = inverse(list(self._asdict().values()), self._shape)
        return Matrix_3D(*coeffs)

class Matrix_4D(namedtuple("Matrix_4D", "a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 a41 a42 a43 a44")):
    _shape = (4,4)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *list(itertools.chain.from_iterable(*args)))
        else:
            return super().__new__(cls, *args, **kwargs)
    
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_4D(*coeffs)

        elif other.__class__.__name__ == "Vector_4D":
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Vector_4D(*coeffs)
        
    def tr(self):
        (coeffs, _) = transpose(list(self._asdict().values()), self._shape)
        return Matrix_4D(*coeffs)
    
    def inv(self):
        (coeffs, _) = inverse(list(self._asdict().values()), self._shape)
        return Matrix_4D(*coeffs)

def matmul(mat_one: list, shape_one: tuple, mat_two: list, shape_two: tuple):
    (rows_one, cols_one) = shape_one
    (rows_two, cols_two) = shape_two

    if len(mat_one) != (rows_one * cols_one) or \
       len(mat_two) != (rows_two * cols_two) or \
       cols_one != rows_two:
        # Indices to not match to perform matrix multiplication
        raise(ShapeMissmatchException)
    else:
        # Example: (3,4) * (4,6) -> will give 3 x 6; cols_one rows_two must match
        # Init coefficients x = rows(mat_one) * cols(mat_two)

        coeffs = [None for i in range(rows_one * cols_two)]
        for row in range(rows_one):
            for col in range(cols_two):
                su = 0
                for it in range(cols_one):
                    # Actually cols_one and rows_two are and must be the same
                    c_one = mat_one[row * cols_one + it]
                    c_two = mat_two[it * cols_two + col]
                    su += c_one * c_two
                coeffs[row * cols_two + col] = su

        # Return coefficients and shape tuple
        return coeffs, (rows_one, cols_two)

def transpose(mat: list, shape: tuple):
    (rows, cols) = shape

    if len(mat) != (rows * cols):
        raise ShapeMissmatchException
    else:
        pass

    coeffs = [None for i in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            co = mat[col * rows + row]
            coeffs[row * cols + col] = co
    
    return coeffs, (cols, rows)

def inverse(mat: list, shape:tuple):
    mr = np.linalg.inv(np.reshape(mat, shape))
    return mr.flatten().tolist(), shape

def cross_product(v0: Vector_3D, v1: Vector_3D):
    c0 = v0.y*v1.z - v0.z*v1.y
    c1 = v0.z*v1.x - v0.x*v1.z
    c2 = v0.x*v1.y - v0.y*v1.x
    return Vector_3D(c0, c1, c2)

def comp_min(v0, v1):
    minx = v0.x if v0.x < v1.x else v1.x
    miny = v0.y if v0.y < v1.y else v1.y
    minz = v0.z if v0.z < v1.z else v1.z
    return Vector_3D(minx, miny, minz)
    
def comp_max(v0, v1):
    maxx = v0.x if v0.x > v1.x else v1.x
    maxy = v0.y if v0.y > v1.y else v1.y
    maxz = v0.z if v0.z > v1.z else v1.z
    return Vector_3D(maxx, maxy, maxz)

def transform_vertex(v : Vector_3D, M: Matrix_4D):
    v = M * v.expand_4D_point()
    v = v.project_3D()
    vz = v.z
    v = v // 1
    return Vector_3D(v.x, v.y, vz)
class ShapeMissmatchException(Exception):
    pass