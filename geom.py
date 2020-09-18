from collections import namedtuple
import numpy as np
import itertools

# Tuple definitions
Point_2D = namedtuple("Point", "x y")
BoundingBox = namedtuple("BoundingBox", "x_min y_min z_min x_max y_max z_max")

class Vector_2D(namedtuple("Vector_2D", "x y")):
    __shape = (3,1)
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            # Calc scalar product
            return self.x * other.x + self.y * other.y

class Vector_3D(namedtuple("Vector_3D", "x y z")):
    __shape = (3,1)
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            # Calc scalar product
            return self.x * other.x + self.y * other.y + self.z * other.z
        
    def project_2D(self):
        return Point_2D(self.x / self.z, self.y / self.z)

class Vector_4D(namedtuple("Vector_4D", "x y z a")):
    __shape = (4,1)

    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            # Calc scalar product
            return self.x * other.x + self.y * other.y + self.z * other.z + self.a * other.a

    def project_3D(self):
        return Point_2D(self.x / self.a, self.y / self.a, self.z / self.a)

class Matrix_3D(namedtuple("Matrix_3D", "a11 a12 a13 a21 a22 a23 a31 a32 a33")):
    __shape = (3,3)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *list(itertools.chain.from_iterable(*args)))
        else:
            return super().__new__(cls, *args, **kwargs)
    
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matmul(list(self._asdict().values()), self.__shape, list(other._asdict().values()), other.__shape)
            return Matrix_3D(*coeffs)

        elif other.__class__.__name__ == "Vector_3D":
            (coeffs, _) = matmul(list(self._asdict().values()), self.__shape, list(other._asdict().values()), other.__shape)
            return Vector_3D(*coeffs)

    def tr(self):
        (coeffs, _) = transpose(list(self._asdict().values()), self.__shape)
        return Matrix_3D(*coeffs)
    
    def inv(self):
        (coeffs, _) = inverse(list(self._asdict().values()), self.__shape)
        return Matrix_3D(*coeffs)

class Matrix_4D(namedtuple("Matrix_4D", "a11 a12 a13 a14 a21 a22 a23 a24 a31 a32 a33 a34 a41 a42 a43 a44")):
    __shape = (4,4)

    def __new__(cls, *args, **kwargs):
        if len(args) > 0 and isinstance(args[0], list):
            return super().__new__(cls, *list(itertools.chain.from_iterable(*args)))
        else:
            return super().__new__(cls, *args, **kwargs)
    
    def __mul__(self, other):
        if other.__class__.__name__ == self.__class__.__name__:
            (coeffs, _) = matmul(list(self._asdict().values()), self.__shape, list(other._asdict().values()), other.__shape)
            return Matrix_3D(*coeffs)

        elif other.__class__.__name__ == "Vector_4D":
            (coeffs, _) = matmul(list(self._asdict().values()), self.__shape, list(other._asdict().values()), other.__shape)
            return Vector_4D(*coeffs)

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

class ShapeMissmatchException(Exception):
    pass