"""The geom module: Includes matrix and vector classes based on NamedTuple and basic algebra."""

from enum import Enum

import typing
from collections.abc import Iterable

from itertools import chain
from functools import reduce
import operator

from math import sqrt

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
    def __init__(self, *args, shape: tuple = None): # pylint: disable=unused-argument
        if not shape is None:
            self._shape = shape

            if len(self.get_field_values()) != self._shape[0] * self._shape[1]:
                raise ShapeMissmatchException

    def __add__(self, other):
        if type(self) == type(other):
            (elems, _) = matadd(self.get_field_values(), self._shape,
                                 other.get_field_values(), other._shape)
            return type(self)(*elems)
        else:
            raise TypeError

    def __sub__(self, other):
        if type(self) == type(other):
            (elems, _) = matsub(self.get_field_values(), self._shape,
                                 other.get_field_values(), other._shape)
            return type(self)(*elems)

        raise TypeError

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            (elems, _) = compmul(self.get_field_values(), self._shape, other)
            return type(self)(*elems)

        # All other cases should already have been handled in instance classes
        raise TypeError

    def __rmul__(self, other):
        if isinstance(other, (float, int)):
            (elems, _) = compmul(self.get_field_values(), self._shape, other)
            return  type(self)(*elems)

        raise TypeError

    def __truediv__(self, other):
        if isinstance(other, (float, int)):
            (elems, _) = compdiv(self.get_field_values(), self._shape, other)
            return type(self)(*elems)

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

    def get_row(self, row_idx):
        """Returns content of row as MatrixNxN object."""
        (rows, cols) = self._shape
        elems = self.get_field_values()
        start_idx = row_idx * rows
        shp = (1, cols)
        cl_type = get_standard_type(shp)
        return cl_type(*elems[start_idx:start_idx+cols], shape = (1, cols))

    def get_col(self, col_idx):
        """Returns content of column as MatrixNxN oject."""
        # Fixme: Improve speed here. Too many transposes
        return self.tr().get_row(col_idx)

    def set_row(self, row_idx, other):
        """Returns same object type with replaced row content."""
        (rows, cols) = self._shape
        if isinstance(other, Iterable):
            lst = unpack_nested_iterable_to_list(other)
        else:
            lst = [other]

        if len(lst) == cols and row_idx < rows:
            elems = self.get_field_values()
            start_idx = row_idx * cols
            elems[start_idx:start_idx+cols] = lst
            return type(self)(elems, shape = self._shape)

        raise ShapeMissmatchException

    def set_col(self, col_idx, other: Iterable):
        """Returns same object type with replaced col content."""
        return self.tr().set_row(col_idx, other).tr()


class MixinMatrix(MixinAlgebra):
    """Mixin providing additional functionalty for matrices based on typing.NamedTuple."""
    def __mul__(self, other):
        if  isinstance(other, (MixinMatrix, MixinVector)):
            (elems, shp) = matmul(self.get_field_values(), self._shape,
                                other.get_field_values(), other._shape)
            return get_standard_type(shp)(*elems, shape = shp)

        # Fallback to more common MixinAlgebra __mul__
        return super().__mul__(other)

    def is_square(self):
        """Returns true if the matrix has square shape e.g. 2x2, 3x3, 5x5 matrices."""
        return self._shape[0] == self._shape[1]

    def inv(self):
        """Returns inverse of a matrix."""
        (elems, _) = inverse(self, self._shape)
        return type(self)(*elems)

    def tr(self): # pylint: disable=invalid-name
        """Returns transpose of a matrix."""
        (elems, shape) = transpose(self.get_field_values(), self._shape)
        return type(self)(elems, shape = shape)

class MixinVector(MixinAlgebra):
    """Mixin providing additional functionalty for vectors based on typing.NamedTuple."""
    def __mul__(self, other):
        if isinstance(self, MixinVector) and isinstance(other, MixinVector) and \
            self._shape[0] < other._shape[0]:
            # Calc scalar product
            (elems, _) = matmul(self.get_field_values(), self._shape,
                                 other.get_field_values(), other._shape)
            return elems[0]

        # Fallback to MixinAlgebra __mul__
        return super().__mul__(other)

    def __floordiv__(self, other):
        if isinstance(other, (float, int)):
            (elems, _) = compfloor(self.get_field_values(), self._shape, other)
            return type(self)(elems, shape = self._shape)

        return ValueError

    def tr(self): # pylint: disable=invalid-name
        """Returns a transposed vector."""
        # Transpose MixinVector
        (rows, cols) = self._shape
 
        elems = self._asdict().values()
        return type(self)(elems, shape = (cols, rows))

    def abs(self):
        """Returns length of vector."""
        return vect_norm(self.get_field_values())

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

class Vector2D(MixinVector, metaclass=NamedTupleMetaEx):
    """Two-dimensional point with x and y ordinate."""
    _shape = (2,1)
    x: float
    y: float
class Vector3D(MixinVector, metaclass=NamedTupleMetaEx):
    """Three-dimensional vector with x, y, z component."""
    _shape = (3,1)
    x: float
    y: float
    z: float

    def expand_4D(self, vtype): # pylint: disable=invalid-name
        """Expands 3D vector to 4D vector regarding the given type.

           Options:
                Vector4D(x, y, z, 0) for vectors with directional meaning.
                Vector4D(x, y, z, 1) for vectors identifying a vertex in 3D space.
           """

        new_shape = (4,1) if is_col_vect(self._shape) else (1,4)

        if vtype == Vector4DType.DIRECTION:
            return Vector4D(self.x, self.y, self.z, 0, shape = new_shape)
        if vtype == Vector4DType.POINT:
            return Vector4D(self.x, self.y, self.z, 1, shape = new_shape)

        return ValueError

    def normalize(self):
        """Normalizes vector to length = 1.0."""
        abl = self.abs()
        return self / abl if abl > 0 else None

class Vector4D(MixinVector, metaclass=NamedTupleMetaEx):
    """Four-dimensional vector with x, y, y and a component."""
    _shape = (4,1)
    _space = None
    x: float
    y: float
    z: float
    a: float

    def project_3D(self, vtype): # pylint: disable=invalid-name
        """Reduces four-dimensional vector to three dimensions.

           If vector has directional meaning the 'a' component is just omited.
           If vector identifies vertex in 3D space the vertex is projected to screen plane z = 1
           by diving all components through last component 'a'.
        """

        new_shape = (3,1) if is_col_vect(self._shape) else (1,3)

        if vtype == Vector4DType.DIRECTION:
            return Vector3D(self.x, self.y, self.z, shape = new_shape)
        if vtype == Vector4DType.POINT:
            return Vector3D(self.x / self.a, self.y / self.a, self.z / self.a, shape = new_shape)

        raise ValueError

class MatrixNxN(MixinMatrix, metaclass=NamedTupleMetaEx):
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
    u_0: float
    u_1: float
    u_2: float
    v_0: float
    v_1: float
    v_2: float

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

class ScreenCoords(MixinMatrix, metaclass=NamedTupleMetaEx):
    """Three-dimensional square matrix for screen coords containing
       three x,y,z vectors in three columns.
    """
    _shape = (3, 3)
    v_0_x: float
    v_1_x: float
    v_2_x: float
    v_0_y: float
    v_1_y: float
    v_2_y: float
    v_0_z: float
    v_1_z: float
    v_2_z: float

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

def comp_min(v_0, v_1):
    """Componentwise min function. Returns min vector."""
    return Vector3D(min(v_0.x, v_1.x), min(v_0.y, v_1.y), min(v_0.z, v_1.z))

def comp_max(v_0, v_1):
    """Componentwise max function. Returns max vector."""
    return Vector3D(max(v_0.x, v_1.x), max(v_0.y, v_1.y), max(v_0.z, v_1.z))

def transform_vertex_to_screen(v : Vector3D, M: Matrix4D): # pylint: disable=invalid-name
    """Transforms 3D vertex to screen coordinates.
       Usually at least viewport matrix is passed for this step as matrix M.

       Returns 3D vector containing int screen coordinates x,y and float z component.
    """

    v = transform_3D4D3D(v, Vector4DType.POINT, M)
    v_z = v.z
    v = v // 1
    return Vector3D(v.x, v.y, v_z)

def transform_3D4D3D(vert: Vector3D, vtype: Vector4DType, M: Matrix4D): # pylint: disable=invalid-name
    """Transforms 3D vertex with matrix. Projects vector to screen plane
       if vector type is point (dividing by a component of internal 4D vector).
    """
    vert = M * vert.expand_4D(vtype)
    return vert.project_3D(vtype)

def unpack_nested_iterable_to_list(it_er: Iterable):
    """Unpacks nested iterables. e.g. [[1,2], [3,4]] becomes [1,2,3,4]."""
    while any(isinstance(e, Iterable) for e in it_er):
        # An iterable is nested in the parent iterable
        it_er = list(chain.from_iterable(it_er))
    return it_er

def compmul(mat_0: list, shape_0: tuple, factor: float):
    """Performing componentwise multiplication with factor c."""
    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise ShapeMissmatchException
    else:
        # Return coefficients and shape tuple
        return [e * factor for e in mat_0], shape_0

def compdiv(mat_0: list, shape_0: tuple, divisor: float):
    """Performing componentwise real division by divisor."""
    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise ShapeMissmatchException
    else:
        # Return coefficients and shape tuple
        return [e / divisor for e in mat_0], shape_0

def compfloor(mat_0: list, shape_0: tuple, divisor: float):
    """Performing componentwise floor division."""
    (rows_0, cols_0) = shape_0

    if len(mat_0) != (rows_0 * cols_0):
        # Indices to not match to perform matrix substraction
        raise ShapeMissmatchException
    else:
        # Return coefficients and shape tuple
        return [int(e // divisor) for e in mat_0], shape_0

def vect_norm(all_elems: list):
    """Return norm of n-dim vector."""
    squared = [elem**2 for elem in all_elems]
    return sqrt(reduce(operator.add, squared))

def matadd(mat_0: list, shape_0: tuple, mat_1: list, shape_1: tuple):
    """Performing componentwise addition."""
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
    """Performing componentwise substraction."""
    mat_1 = [e * -1 for e in mat_1]
    return matadd(mat_0, shape_0, mat_1, shape_1)

def is_row_vect(shape: tuple):
    """Returning true if vector shape is row space e.g. shape = (1,4)"""
    return shape[0] < shape[1]

def is_col_vect(shape: tuple):
    """Returning true if vector shape is col space e.g. shape = (4,1)"""
    return shape[0] > shape[1]

def get_standard_type(shape: tuple):
    """Return standard return classes for given shapes."""
    rows, cols = shape
    if cols > rows:
        # Switch to have shape sorted
        shape = (cols, rows)
    if shape == (4,4):
        return Matrix4D
    if shape == (3,3):
        return Matrix3D
    if shape == (4,1):
        return Vector4D
    if shape == (3,1):
        return Vector3D
    if shape == (2,1):
        return Point2D
     # Fallback to NxN Matrix if no special shape applies
    return MatrixNxN

class ShapeMissmatchException(Exception):
    """Exception raised when matrix, vector dimensions do not fit."""
