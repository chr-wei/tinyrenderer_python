import numpy as np
import re
from collections import namedtuple

Facedata_Ids = namedtuple("Facedata_Ids", "Vertex_Ids Tangent_Ids Normal_Ids")
Vertex_Ids = namedtuple("Vertex_Ids", "id_one id_two id_three")
Tangent_Ids = namedtuple("Tangent_Ids", "id_one id_two id_three")
Normal_Ids = namedtuple("Normal_Ids", "id_one id_two id_three")

Vertex = namedtuple("Vertex", "x y z")


def get_model_face_ids(obj_filename):

    face_line_pattern = r"^f"
    face_id_data_dict = {}

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(face_line_pattern, line)
            if match:
                face_data = read_face_ids(line)
                face_count = len(face_id_data_dict)
                face_id_data_dict[face_count + 1] = face_data

    return face_id_data_dict



def read_face_ids(face_data_line):

    face_elem_pattern = r"(\d+)\/(\d+)\/(\d+)"
    match = re.findall(face_elem_pattern, face_data_line)

    vert_list = []
    tang_list = []
    norm_list = []

    for idx in range(0, len(match)):
        vert_list.append(int(match[idx][0]))
        tang_list.append(int(match[idx][1]))
        norm_list.append(int(match[idx][2]))

    vert_ids = Vertex_Ids(*vert_list)
    tang_ids = Tangent_Ids(*tang_list)
    norm_ids = Normal_Ids(*norm_list)

    return Facedata_Ids(vert_ids, tang_ids, norm_ids)


def get_vertices(obj_filename):
    vertex_dict = {}

    vertex_pattern = r"^v"

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(vertex_pattern, line)
            if match:
                vertex_elem_pattern = r"[+-]?[0-9]*[.]?[0-9]+[e\+\-\d]*"

                match = re.findall(vertex_elem_pattern, line)
                elem_list = []
                if match:
                    for elem in match:
                        elem_list.append(float(elem))

                    vert = Vertex(*elem_list)
                    vertex_count = len(vertex_dict)
                    vertex_dict[vertex_count + 1] = vert

    return vertex_dict