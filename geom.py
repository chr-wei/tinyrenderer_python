"""The geom module: Includes matrix and vector classes based on NamedTuple and basic algebra."""

from enum import Enum

import typing
from collections.abc import Iterable

from itertools import chain
import operator

import math
import numpy as np

class Vector4DType(Enum):
    """Enum specifying if 4D vector is meant to be direction or point."""
    DIRECTION = 0
    POINT = 1

class NamedTupleMetaEx(typing.NamedTupleMeta):
    """typing.NamedTuple metaclass to provide mixin functionalty alongside typing.NamedTuples."""
    def __new__(cls, typename, bases, ns):
        cls_obj = super().__new__(cls, typename+'_nm_base', bases, ns)
        bases = bases + (cls_obj,)
        return type(typename, bases, {})

class MixinAlgebra():
    """Mixin providing basic functionality for matrices and vectors based on typing.NamedTuple."""
    def __new__(cls, *args, shape: tuple = None): # pylint: disable=unused-argument
        if isinstance(args[0], Iterable):
            if len(cls._fields) > 1:
                return super().__new__(cls, *unpack_nested_iterable_to_list(args[0]))
            else:
                return super().__new__(cls, unpack_nested_iterable_to_list(args))
        else:
            if len(cls._fields) > 1:
                return super().__new__(cls, *args)
            else:
                return super().__new__(cls, list(args))

    # Overwrite __init__ to add 'shape' keyword parameter
    def __init__(self, shape: tuple = None):
        if not shape is None:
            self._shape = shape

            if len(self.get_field_values()) != self._shape[0] * self._shape[1]:
                raise ShapeMissmatchException

    def __add__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (elems, _) = matadd(self.get_field_values(), self._shape,
                                 other.get_field_values(), other._shape)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems)
        else:
            raise TypeError

    def __sub__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (elems, _) = matsub(self.get_field_values(), self._shape,
                                 other.get_field_values(), other._shape)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems)
        else:
            raise TypeError

    def __mul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (elems, _) = compmul(self.get_field_values(), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems)
        else:
            raise TypeError

    def __rmul__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (elems, _) = compmul(self.get_field_values(), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems)
        else:
            raise TypeError

    def __truediv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (elems, _) = compdiv(self.get_field_values(), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems)
        else:
            raise TypeError

    def get_field_values(self):
        """Returns all field values of the typing.NamedTuple._asdict method as list.
           If there is solely a single list field this list will be returned."""
        if len(self._fields) == 1 and 'elems' in self.__annotations__:
            return list(self._asdict().values())[0]
        else:
            return list(self._asdict().values())

    def __str__(self):
        prefix = self.__class__.__name__ + "("
        with np.printoptions(precision = 3, suppress = True):
            npa = np.array(self).reshape(self._shape)
            return prefix + np.array2string(npa, prefix=prefix) + ")"


class MixinMatrix(MixinAlgebra):
    """Mixin providing additional functionalty for matrices based on typing.NamedTuple."""
    def __mul__(self, other):
        if  MixinVector in other.__class__.__bases__:
            (elems, shp) = matmul(self.get_field_values(), self._shape,
                                other.get_field_values(), other._shape)
            if self.is_square():
                cl_type = globals()[other.__class__.__name__]
                return cl_type(*elems)
            else:
                return Matrix_NxN(elems, shape = shp)

        elif MixinMatrix in other.__class__.__bases__:
            (elems, shp) = matmul(self.get_field_values(), self._shape,
                                     other.get_field_values(), other._shape)

            if self.is_square() and other.is_square():
                cl_type = globals()[self.__class__.__name__]
                return cl_type(*elems)
            else:
                return Matrix_NxN(elems, shape = shp)

    def is_square(self):
        """Returns true if the matrix has square shape e.g. 2x2, 3x3, 5x5 matrices."""
        return self._shape[0] == self._shape[1]

    def inv(self):
        """Returns inverse of a matrix."""
        (elems, _) = inverse(self, self._shape)
        cl_type = globals()[self.__class__.__name__]
        return cl_type(*elems)

    def tr(self):
        """Returns transpose of a matrix."""
        (elems, shape) = transpose(self.get_field_values(), self._shape)
        cl_type = globals()[self.__class__.__name__]
        return cl_type(*elems, shape = shape)

    def get_row(self, row_idx):
        """Returns content of row as MatrixNxN object."""
        (rows, cols) = self._shape
        elems = self.get_field_values()
        start_idx = row_idx * rows
        return Matrix_NxN(elems[start_idx:start_idx+cols], shape = (1, cols))

    def get_col(self, col_idx):
        """Returns content of column as MatrixNxN oject."""
        return self.tr().get_row(col_idx)

    def set_row(self, row_idx, other: Iterable):
        """Returns same object type with replaced row content."""
        (rows, cols) = self._shape
        li = unpack_nested_iterable_to_list(other)

        if len(li) == cols and row_idx < rows:
            elems = self.get_field_values()
            start_idx = row_idx * cols
            elems[start_idx:start_idx+cols] = li
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems, shape = self._shape)
        else:
            raise ShapeMissmatchException

    def set_col(self, col_idx, other: Iterable):
        """Returns same object type with replaced col content."""
        return self.tr().set_row(col_idx, other).tr()

class MixinVector(MixinAlgebra):
    """Mixin providing additional functionalty for vectors based on typing.NamedTuple."""
    def __mul__(self, other):
        if self.__class__.__name__ == other.__class__.__name__ and \
            self._shape[0] < other._shape[0]:
            # Calc scalar product
            (elems, _) = matmul(self.get_field_values(), self._shape,
                                 other.get_field_values(), other._shape)
            return elems[0]

        elif other.__class__.__name__ in ["float", "int"]:
            return super().__mul__(other)

    def __floordiv__(self, other):
        if other.__class__.__name__ in ["float", "int"]:
            (elems, _) = compfloor(self.get_field_values(), self._shape, other)
            cl_type = globals()[self.__class__.__name__]
            return cl_type(*elems, shape = self._shape)

    def tr(self):
        """Returns a transposed vector."""
        # Transpose MixinVector
        (rows, cols) = self._shape

        cl_type = globals()[self.__class__.__name__]
        elems = self._asdict().values()
        return cl_type(*elems, shape = (cols, rows))

class Point2D(MixinVector, metaclass=NamedTupleMetaEx):
    """Two-dimensional point with x and y ordinate."""
    _shape = (2,1)
    x: float
    y: float

class PointUV(MixinVector, metaclass=NamedTupleMetaEx):
    """Two-dimensional point with u and v ordinate for interpolated texture and map coordinates."""
    _shape = (2,1)
    u: float
    v: float

class Barycentric(MixinVector, metaclass=NamedTupleMetaEx):
    """Three-dimensional vector to store barycentric coordinates of a triangle."""
    _shape = (3,1)
    one_u_v: float
    u: float
    v: float
class Vector3D(MixinVector, metaclass=NamedTupleMetaEx):
    """Three-dimensional vector with x, y, z component."""
    _shape = (3,1)
    x: float
    y: float
    z: float

    def expand_4D(self, vtype):
        """Expands 3D vector to 4D vector regarding the given type.

           Options:
                Vector4D(x, y, z, 0) for vectors with directional meaning.
                Vector4D(x, y, z, 1) for vectors identifying a vertex in 3D space.
           """

        if is_col_vect(self._shape):
            new_shape = (4,1)
        else:
            new_shape = (1,4)

        if vtype == Vector4DType.DIRECTION:
            return Vector4D(self.x, self.y, self.z, 0, shape = new_shape)
        elif vtype == Vector4DType.POINT:
            return Vector4D(self.x, self.y, self.z, 1, shape = new_shape)

    def abs(self):
        """Returns length of vector."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def normalize(self):
        """Normalizes vector to length = 1.0."""
        abl = self.abs()
        if abl > 0:
            return self / abl
        else:
            return None

class Vector4D(MixinVector, metaclass=NamedTupleMetaEx):
    """Four-dimensional vector with x, y, y and a component."""
    _shape = (4,1)
    _space = None
    x: float
    y: float
    z: float
    a: float

    def project_3D(self, vtype):
        """Reduces four-dimensional vector to three dimensions.

           If vector has directional meaning the 'a' component is just omited.
           If vector identifies vertex in 3D space the vertex is projected to screen plane z = 1
           by diving all components through last component 'a'.
        """

        if is_col_vect(self._shape):
            new_shape = (3,1)
        else:
            new_shape = (1,3)

        if vtype == Vector4DType.DIRECTION:
            return Vector3D(self.x, self.y, self.z, shape = new_shape)
        elif vtype == Vector4DType.POINT:
            return Vector3D(self.x / self.a, self.y / self.a, self.z / self.a, shape = new_shape)
        else:
            raise ValueError

class Matrix_NxN(MixinMatrix, metaclass=NamedTupleMetaEx):
    """Matrix with any size (n x n).

       Parameters:
           elems: list containing all matrix components. List may have nested lists.
           shape: tuple containing the matrix shape.
                  e.g. (2,3) for two rows and three columns
    """

    _shape = None
    elems: list

class MatrixUV(MixinMatrix, metaclass=NamedTupleMetaEx):
    """Matrix with size (2 x 3) holding three pairs of uv coordinates."""
    _shape = (2,3)
    u0: float
    u1: float
    u2: float
    v0: float
    v1: float
    v2: float

class Matrix3D(MixinMatrix, metaclass=NamedTupleMetaEx):
    """Three-dimensional square matrix."""
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

class Matrix4D(MixinMatrix, metaclass=NamedTupleMetaEx):
    """Four-dimensional square matrix."""
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
    """Function performing matrix multiplication.

       Parameters:
           mat_0: list containg components of first matrix
           shape_0: tuple containing size of first matrix
           mat_1: list containg components of second matrix
           shape_1: tuple containing size of second matrix
    """

    (rows_0, cols_0) = shape_0
    (rows_1, cols_1) = shape_1

    if len(mat_0) != (rows_0 * cols_0) or \
       len(mat_1) != (rows_1 * cols_1) or \
       cols_0 != rows_1:
        raise ShapeMissmatchException
    else:
        # Example: (3,4) * (4,6) -> will give 3 x 6; cols_0 rows_1 must match
        # Init coefficients x = rows(mat_0) * cols(mat_1)

        elems = [None for i in range(rows_0 * cols_1)]
        for row in range(rows_0):
            for col in range(cols_1):
                comp_sum = 0
                for ele in range(cols_0):
                    # Actually cols_0 and rows_1 are and must be the same
                    c_0 = mat_0[row * cols_0 + ele]
                    c_1 = mat_1[ele * cols_1 + col]
                    comp_sum += c_0 * c_1
                elems[row * cols_1 + col] = comp_sum

        # Return coefficients and shape tuple
        return elems, (rows_0, cols_1)

def transpose(mat: list, shape: tuple):
    """Function performing matrix transpose."""

    (rows, cols) = shape

    if len(mat) != (rows * cols):
        raise ShapeMissmatchException
    else:
        pass

    elems = [None for i in range(rows * cols)]
    for row in range(rows):
        for col in range(cols):
            ele = mat[row * cols + col] # Read row-wise
            elems[col * rows + row] = ele

    return elems, (cols, rows)

def inverse(mat: list, shape:tuple):
    """Calculate inverse of a matrix using numpy."""
    arr_inv = np.linalg.inv(np.reshape(mat, shape))
    return arr_inv.flatten().tolist(), shape

def cross_product(v_0: Vector3D, v_1: Vector3D):
    """Calculates cross product of two three-dimensional vectors."""
    c_0 = v_0.y * v_1.z - v_0.z * v_1.y
    c_1 = v_0.z * v_1.x - v_0.x * v_1.z
    c_2 = v_0.x * v_1.y - v_0.y * v_1.x
    return Vector3D(c_0, c_1, c_2)

def comp_min(v0, v1):
    return Vector3D(min(v0.x, v1.x), min(v0.y, v1.y), min(v0.z, v1.z))

def comp_max(v0, v1):
    return Vector3D(max(v0.x, v1.x), max(v0.y, v1.y), max(v0.z, v1.z))

def transform_vertex_to_screen(v : Vector3D, M: Matrix4D):
    v = transform_3D4D3D(v, Vector4DType.POINT, M)
    vz = v.z
    v = v // 1
    return Vector3D(v.x, v.y, vz)

def transform_3D4D3D(v: Vector3D, vtype: Vector4DType, M: Matrix4D):
    v = M * v.expand_4D(vtype)
    return v.project_3D(vtype)

def unpack_nested_iterable_to_list(it: Iterable):
    while any(isinstance(e, Iterable) for e in it):
        # An iterable is nested in the parent iterable
        it = list(chain.from_iterable(it))
    return it

def compmul(mat_0: list, shape_0: tuple, c: float):

    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise ShapeMissmatchException
    else:
        # Return coefficients and shape tuple
        return [e * c for e in mat_0], shape_0

def compdiv(mat_0: list, shape_0: tuple, c: float):
    """Performing componentwise real division."""
    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise ShapeMissmatchException
    else:
        # Return coefficients and shape tuple
        return [e / c for e in mat_0], shape_0

def compfloor(mat_0: list, shape_0: tuple, c: float):
    """Performing componentwise floor division."""
    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise ShapeMissmatchException
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
        raise ShapeMissmatchException
    else:
        # Return coefficients and shape tuple
        return map(operator.add, mat_0, mat_1), shape_0

def matsub(mat_0: list, shape_0: tuple, mat_1: list, shape_1: tuple):
    mat_1 = [e * -1 for e in mat_1]
    return matadd(mat_0, shape_0, mat_1, shape_1)

def is_row_vect(shape: tuple):
    """Returning true if vector shape is row space e.g. shape = (1,4)"""
    return shape[0] < shape[1]

def is_col_vect(shape: tuple):
    """Returning true if vector shape is col space e.g. shape = (4,1)"""
    return shape[0] > shape[1]

class ShapeMissmatchException(Exception):
    pass
