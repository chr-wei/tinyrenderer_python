"""The geom module: Includes matrix and vector classes based on NamedTuple and basic algebra."""

from typing import NamedTuple, NamedTupleMeta
import typing
from collections import namedtuple
from collections.abc import Iterable
import numpy as np
import math
from itertools import chain
import operator
from enum import Enum

class Vector_4D_Type(Enum):
    DIRECTION = 0
    POINT = 1

class NamedTupleMetaEx(typing.NamedTupleMeta):
    def __new__(cls, typename, bases, ns):
        cls_obj = super().__new__(cls, typename+'_nm_base', bases, ns)
        bases = bases + (cls_obj,)
        return type(typename, bases, {})

class MixinAlgebra(): 
    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matadd(list(self._asdict().values()), self._shape, 
                                 list(other._asdict().values()), other._shape)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs)
    
    def __sub__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matsub(list(self._asdict().values()), self._shape,
                                 list(other._asdict().values()), other._shape)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs)

    def __mul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (coeffs, _) = compmul(list(self._asdict().values()), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs)
    
    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (coeffs, _) = compmul(list(self._asdict().values()), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs)
    
    def __truediv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (coeffs, _) = compdiv(list(self._asdict().values()), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs)


class MixinMatrix(MixinAlgebra):
    def __new__(cls, *args):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *unpack_nested_iterable_to_list(*args))
        else:
            return super().__new__(cls, *args)

    def __mul__(self, other):
        if  MixinVector in other.__class__.__bases__:
            (coeffs, shape) = matmul(list(self._asdict().values()), self._shape, 
                                 list(other._asdict().values()), other._shape) 
                
            if self.is_square():         
                cl_type = globals()[other.__class__.__name__]
                return cl_type(*coeffs)
            else:
                return coeffs, shape

        elif MixinMatrix in other.__class__.__bases__:
            (coeffs, shape) = matmul(list(self._asdict().values()), self._shape, 
                                     list(other._asdict().values()), other._shape)
            
            if self.is_square() and other.is_square():
                cl_type = globals()[self.__class__.__name__]
                return cl_type(*coeffs)
            else:
                return coeffs, shape
    
    def __str__(self):
        prefix = self.__class__.__name__ + "("
        with np.printoptions(precision = 3, suppress = True):
            npa = np.array(self).reshape(self._shape)
            return prefix + np.array2string(npa, prefix=prefix) + ")"

    def is_square(self):
        return self._shape[0] == self._shape[1]

    def inv(self):
        (coeffs, _) = inverse(self, self._shape)
        cl_type = globals()[self.__class__.__name__]
        return cl_type(*coeffs)

    def tr(self):
        (coeffs, _) = transpose(self, self._shape)
        cl_type = globals()[self.__class__.__name__]
        return cl_type(*coeffs)

    def set_row(self, row_idx, other: list):
        (r,c) = self._shape
        if len(other) == c and row_idx < r:
            coeffs = list(self._asdict().values())
            start_idx = row_idx * r
            coeffs[start_idx : start_idx + len(other)] = other
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs)
    
    def set_col(self, col_idx, other: list):
        return self.tr().set_row(col_idx, other).tr()

class MixinVector(MixinAlgebra):
    # Overwrite __new__ to add 'space' keyword parameter
    def __new__(self, *args, shape: tuple = None):
        if len(args) > 0 and isinstance(args[0], Iterable):
            return super().__new__(self, *unpack_nested_iterable_to_list(*args))
        else:
            return super().__new__(self, *args)
    
    def __init__(self, *args, shape: tuple = None):
        if not shape is None:
            self._shape = shape

    def __mul__(self, other):     
        if self.__class__.__name__ == other.__class__.__name__ and \
            self._shape[0] < other._shape[0]:
            # Calc scalar product
            (coeffs, _) = matmul(list(self._asdict().values()), self._shape, 
                                 list(other._asdict().values()), other._shape)
            return coeffs[0]
            
        elif other.__class__.__name__ in ["float", "int"]:
            return super().__mul__(other)


    def __floordiv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (coeffs, _) = compfloor(list(self._asdict().values()), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*coeffs, shape = self._shape)

    def tr(self):
        # Transpose MixinVector
        (s,h) = self._shape
        
        cl_type = globals()[self.__class__.__name__]
        coeffs = self._asdict().values()
        return cl_type(*coeffs, shape = (h,s))

class Point_2D(MixinVector, metaclass=NamedTupleMetaEx):
    _shape = (2,1)
    x: float
    y: float

class Point_UV(MixinVector, metaclass=NamedTupleMetaEx):
    _shape = (2,1)
    u: float
    v: float

class Barycentric(MixinVector, metaclass=NamedTupleMetaEx):
    _shape = (3,1)
    one_u_v: float
    u: float
    v: float
class Vector_3D(MixinVector, metaclass=NamedTupleMetaEx):
    _shape = (3,1)
    x: float
    y: float
    z: float

    def expand_4D(self, vtype):
        if is_col_vect(self._shape):
            new_shape = (4,1)
        else:
            new_shape = (1,4)

        if vtype == Vector_4D_Type.DIRECTION:
            return Vector_4D(self.x, self.y, self.z, 0, shape = new_shape)
        elif vtype == Vector_4D_Type.POINT:
            return Vector_4D(self.x, self.y, self.z, 1, shape = new_shape)

    def abs(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def norm(self):
        ab = self.abs()
        if ab > 0:
            return self / ab
        else:
            return None

class Vector_4D(MixinVector, metaclass=NamedTupleMetaEx):
    _shape = (4,1)
    _space = None
    x: float
    y: float
    z: float
    a: float

    def project_3D(self, vtype):
        if is_col_vect(self._shape):
            new_shape = (3,1)
        else:
            new_shape = (1,3)

        if vtype == Vector_4D_Type.DIRECTION:
            return Vector_3D(self.x, self.y, self.z, shape = new_shape)
        elif vtype == Vector_4D_Type.POINT:
            return Vector_3D(self.x / self.a, self.y / self.a, self.z / self.a, shape = new_shape)

class Matrix_2x3(MixinMatrix, metaclass=NamedTupleMetaEx):
    _shape = (2, 3)
    a11: float
    a12: float
    a13: float
    a21: float
    a22: float
    a23: float


class Matrix_3D(MixinMatrix, metaclass=NamedTupleMetaEx):
    _shape = (3, 3)
    a11: float
    a12: float
    a13: float
    a21: float
    a22: float
    a23: float
    a31: float
    a32: float
    a33: float

class Matrix_4D(MixinMatrix, metaclass=NamedTupleMetaEx):
    _shape = (4, 4)
    a11: float
    a12: float
    a13: float
    a14: float
    a21: float
    a22: float
    a23: float
    a24: float
    a31: float
    a32: float
    a33: float
    a34: float
    a41: float
    a42: float
    a43: float
    a44: float

def matmul(mat_0: list, shape_0: tuple, mat_1: list, shape_1: tuple):
    
    (rows_0, cols_0) = shape_0
    (rows_1, cols_1) = shape_1

    if len(mat_0) != (rows_0 * cols_0) or \
       len(mat_1) != (rows_1 * cols_1) or \
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
                    c_0 = mat_0[row * cols_0 + it]
                    c_1 = mat_1[it * cols_1 + col]
                    su += c_0 * c_1
                coeffs[row * cols_1 + col] = su

        # Return coefficients and shape tuple
        return coeffs, (rows_0, cols_1)

def transpose(mat: list, shape: tuple):

    (rows, cols) = shape

    if len(mat) != (rows * cols):
        raise ShapeMissmatchException
    else:
        pass

    coeffs = [None for i in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            co = mat[row * cols + col] # Read row-wise
            coeffs[col * rows + row] = co
    
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

def compmul(mat_0: list, shape_0: tuple, c: float):

    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise(ShapeMissmatchException)
    else:
        # Return coefficients and shape tuple
        return [e * c for e in mat_0], shape_0

def compdiv(mat_0: list, shape_0: tuple, c: float):

    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise(ShapeMissmatchException)
    else:
        # Return coefficients and shape tuple
        return [e / c for e in mat_0], shape_0

def compfloor(mat_0: list, shape_0: tuple, c: float):

    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise(ShapeMissmatchException)
    else:
        # Return coefficients and shape tuple
        return [int(e // c) for e in mat_0], shape_0

def matadd(mat_0: list, shape_0: tuple, mat_1: list, shape_1: tuple):

    (rows_0, cols_0) = shape_0
    (rows_1, cols_1) = shape_1

    if len(mat_0) != (rows_0 * cols_0) or \
       len(mat_1) != (rows_1 * cols_1) or \
       shape_0 != shape_1:
        # Indices to not match to perform matrix substraction
        raise(ShapeMissmatchException)
    else:
        # Return coefficients and shape tuple
        return map(operator.add, mat_0, mat_1), shape_0

def matsub(mat_0: list, shape_0: tuple, mat_1: list, shape_1: tuple):
    mat_1 = [e * -1 for e in mat_1]
    return matadd(mat_0, shape_0, mat_1, shape_1)

def is_row_vect(shape: tuple):
    return shape[0] < shape[1]

def is_col_vect(shape: tuple):
    return shape[0] > shape[1]

class ShapeMissmatchException(Exception):
    pass