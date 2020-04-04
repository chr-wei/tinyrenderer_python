import numpy as np
import re
from collections import namedtuple

Vertex = namedtuple("Vertex", "x y z")

def get_model_faces():
    faces = []
    return faces

def read_face_vertices(face_number):
    v0 = Vertex(0, 0, 0)
    v1 = Vertex(0, 0, 0)
    v2 = Vertex(0, 0, 0)

    return (v0, v1, v2)