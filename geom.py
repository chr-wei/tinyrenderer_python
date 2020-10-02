from collections import namedtuple
from collections.abc import Iterable
import numpy as np
import math
from itertools import chain
import operator
from enum import Enum

class Vector_Space(Enum):
    ROW = 0
    COLUMN = 1

class Vector_4D_Type(Enum):
    DIRECTION = 0
    POINT = 1

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

    def __floordiv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Point_2D(int(self.x // other), int(self.y // other))

class Vector_3D(namedtuple("Vector_3D", "x y z")):
    _shape: tuple
    _space: Vector_Space

    # Overwrite __new__ to add 'space' keyword parameter
    def __new__(cls, *args, space: Vector_Space = Vector_Space.COLUMN, **kwargs):
        return super().__new__(cls, *args)
    
    def __init__(self, *args, space: Vector_Space = Vector_Space.COLUMN, **kwargs):
        if space == Vector_Space.COLUMN:
            self._shape = (3,1)
            self._space = space
        elif space == Vector_Space.ROW:
            self._shape = (1,3)
            self._space = space

    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            # Calc scalar product
            return matmul(self, self._shape, other, other._shape)[0][0]
        elif other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(self.x * other, self.y * other, self.z * other, space = self._space)
        
    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(self.x * other, self.y * other, self.z * other, space = self._space)
    
    def __truediv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(self.x / other, self.y / other, self.z / other, space = self._space)
    
    def __floordiv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_3D(int(self.x // other), int(self.y // other), int(self.z // other), space = self._space)

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return Vector_3D(self.x + other.x, self.y + other.y, self.z + other.z, space = self._space)
    
    def __sub__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            return Vector_3D(self.x - other.x, self.y - other.y, self.z - other.z, space = self._space)
    
    def expand_4D(self, vtype):
        if vtype == Vector_4D_Type.DIRECTION:
            return Vector_4D(self.x, self.y, self.z, 0, space = self._space)
        elif vtype == Vector_4D_Type.POINT:
            return Vector_4D(self.x, self.y, self.z, 1, space = self._space)

    def abs(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm(self):
        ab = self.abs()
        if ab > 0:
            return self / ab
        else:
            return None
    
    def tr(self):
        # Transpose vector
        if self._space == Vector_Space.ROW:
            new_orient = Vector_Space.COLUMN
        else:
            new_orient = Vector_Space.ROW
        return Vector_3D(*self, space = new_orient)
    

class Vector_4D(namedtuple("Vector_4D", "x y z a")):
    _shape: tuple
    _space: Vector_Space

    # Overwrite __new__ to add 'space' keyword parameter
    def __new__(cls, *args, space: Vector_Space = Vector_Space.COLUMN, **kwargs):
        return super().__new__(cls, *args)
    
    def __init__(self, *args, space: Vector_Space = Vector_Space.COLUMN, **kwargs):
        if space == Vector_Space.COLUMN:
            self._shape = (4,1)
        elif space == Vector_Space.ROW:
            self._shape = (1,4)
        
        self._space = space

    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
           # Calc scalar product
            return matmul(self, self._shape, other, other._shape)[0][0]
        elif other.__class__.__name__ in ["float", "int"]:
            return Vector_4D(self.x * other, self.y * other, self.z * other, self.a * other, space = self._space)
        
    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            return Vector_4D(self.x * other, self.y * other, self.z * other, self.a * other, space = self._space)

    def project_3D(self, vtype):
        if vtype == Vector_4D_Type.DIRECTION:
            return Vector_3D(self.x, self.y, self.z, space = self._space)
        elif vtype == Vector_4D_Type.POINT:
            return Vector_3D(self.x / self.a, self.y / self.a, self.z / self.a, space = self._space)

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            if self._space == other._space:
                return Vector_4D(self.x + other.x, self.y + other.y, self.z + other.z, self.a + other.a, 
                                 space = self._space)
            else:
                raise(ShapeMissmatchException)

class Matrix_3D(namedtuple("Matrix_3D", "a11 a12 a13 a21 a22 a23 a31 a32 a33")):
    _shape = (3,3)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *unpack_nested_iterable_to_list(*args))
        else:
            return super().__new__(cls, *args)

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = add(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_3D(*coeffs)

    def __sub__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = substract(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_3D(*coeffs)

    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_3D(*coeffs)

        elif other.__class__.__name__ == "Vector_3D":
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Vector_3D(*coeffs)

    def tr(self):
        (coeffs, _) = transpose(self, self._shape)
        return Matrix_3D(*coeffs)
    
    def inv(self):
        (coeffs, _) = inverse(self, self._shape)
        return Matrix_3D(*coeffs)

class Matrix_4D(namedtuple("Matrix_4D", "a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 a41 a42 a43 a44")):
    _shape = (4,4)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *unpack_nested_iterable_to_list(*args))
        else:
            return super().__new__(cls, *args)

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = add(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_4D(*coeffs)

    def __sub__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = substract(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_4D(*coeffs)
    
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Matrix_4D(*coeffs)

        elif other.__class__.__name__ == "Vector_4D":
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, list(other._asdict().values()), other._shape)
            return Vector_4D(*coeffs)
        
    def tr(self):
        (coeffs, _) = transpose(self, self._shape)
        return Matrix_4D(*coeffs)
    
    def inv(self):
        (coeffs, _) = inverse(self, self._shape)
        return Matrix_4D(*coeffs)

def matmul(mat_0: Iterable, shape_0: tuple, mat_1: Iterable, shape_1: tuple):
    unpacked_mat_0 = unpack_nested_iterable_to_list(mat_0)
    unpacked_mat_1 = unpack_nested_iterable_to_list(mat_1)

    (rows_0, cols_0) = shape_0
    (rows_1, cols_1) = shape_1

    if len(unpacked_mat_0) != (rows_0 * cols_0) or \
       len(unpacked_mat_1) != (rows_1 * cols_1) or \
       cols_0 != rows_1:
        # Indices to not match to perform matrix multiplication
        raise(ShapeMissmatchException)
    else:
        # Example: (3,4) * (4,6) -> will give 3 x 6; cols_0 rows_1 must match
        # Init coefficients x = rows(mat_0) * cols(mat_1)

        coeffs = [None for i in range(rows_0 * cols_1)]
        for row in range(rows_0):
            for col in range(cols_1):
                su = 0
                for it in range(cols_0):
                    # Actually cols_0 and rows_1 are and must be the same
                    c_0 = unpacked_mat_0[row * cols_0 + it]
                    c_1 = unpacked_mat_1[it * cols_1 + col]
                    su += c_0 * c_1
                coeffs[row * cols_1 + col] = su

        # Return coefficients and shape tuple
        return coeffs, (rows_0, cols_1)

def transpose(mat: list, shape: tuple):
    unpacked_mat = unpack_nested_iterable_to_list(mat)

    (rows, cols) = shape

    if len(unpacked_mat) != (rows * cols):
        raise ShapeMissmatchException
    else:
        pass

    coeffs = [None for i in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            co = unpacked_mat[row * cols + col] # Read row-wise
            coeffs[col * rows + row] = co
    
    return coeffs, (cols, rows)

def inverse(mat: Iterable, shape:tuple):
    unpacked_mat = unpack_nested_iterable_to_list(mat)
    mr = np.linalg.inv(np.reshape(unpacked_mat, shape))
    return mr.flatten().tolist(), shape

def cross_product(v0: Vector_3D, v1: Vector_3D):
    c0 = v0.y*v1.z - v0.z*v1.y
    c1 = v0.z*v1.x - v0.x*v1.z
    c2 = v0.x*v1.y - v0.y*v1.x
    return Vector_3D(c0, c1, c2)

def comp_min(v0, v1):
    return Vector_3D(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z))
    
def comp_max(v0, v1):
    return Vector_3D(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z))

def transform_vertex_to_screen(v : Vector_3D, M: Matrix_4D):
    v = transform_3D4D3D(v, Vector_4D_Type.POINT, M)
    vz = v.z
    v = v // 1
    return Vector_3D(v.x, v.y, vz)

def transform_3D4D3D(v: Vector_3D, vtype: Vector_4D_Type, M: Matrix_4D):
    v = M * v.expand_4D(vtype)
    return v.project_3D(vtype)

def unpack_nested_iterable_to_list(parent_it: Iterable):
    if any(isinstance(elem, Iterable) for elem in parent_it):
        # An iterable is nested in the parent iterable
        return list(chain.from_iterable(parent_it))
    else:
        # No nested iterable - return parent iterable
        return parent_it

def add(mat_0: Iterable, shape_0: tuple, mat_1: Iterable, shape_1: tuple):
    unpacked_mat_0 = unpack_nested_iterable_to_list(mat_0)
    unpacked_mat_1 = unpack_nested_iterable_to_list(mat_1)

    (rows_0, cols_0) = shape_0
    (rows_1, cols_1) = shape_1

    if len(unpacked_mat_0) != (rows_0 * cols_0) or \
       len(unpacked_mat_1) != (rows_1 * cols_1) or \
       shape_0 != shape_1:
        # Indices to not match to perform matrix substraction
        raise(ShapeMissmatchException)
    else:
        # Return coefficients and shape tuple
        return map(operator.add, mat_0, mat_1), shape_0

def substract(mat_0: Iterable, shape_0: tuple, mat_1: Iterable, shape_1: tuple):
    unpacked_mat_0 = unpack_nested_iterable_to_list(mat_0)
    unpacked_mat_1 = unpack_nested_iterable_to_list(mat_1)
    unpacked_mat_1 = [e * -1 for e in unpacked_mat_1]
    return add(unpacked_mat_0, shape_0, unpacked_mat_1, shape_1)

class ShapeMissmatchException(Exception):
    pass