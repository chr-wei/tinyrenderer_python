import re
import sys

from our_gl import Point, Vertex, BoundingBox
from collections import namedtuple
from tiny_image import TinyImage

VertexIds = namedtuple("VertexIds", "id_one id_two id_three")
TexturePointIds = namedtuple("TexturePointIds", "id_one id_two id_three")
NormalIds = namedtuple("NormalIds", "id_one id_two id_three")
FacedataIds = namedtuple("FacedataIds", "VertexIds TexturePointIds NormalIds")

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

    face_elem_pattern = r"(\d+)\/(\d*)\/(\d+)"
    match = re.findall(face_elem_pattern, face_data_line)

    vert_list = []
    tang_list = []
    norm_list = []

    for idx in range(0, len(match)):
        vert_list.append(int(match[idx][0]))
        tang = match[idx][1]
        if tang.isdigit():
            tang = int(tang)
        tang_list.append(tang)
        norm_list.append(int(match[idx][2]))

    vert_ids = VertexIds(*vert_list[:3])
    text_pt_ids = TexturePointIds(*tang_list[:3])
    norm_ids = NormalIds(*norm_list[:3])

    return FacedataIds(vert_ids, text_pt_ids, norm_ids)

def get_model_texture_points(obj_filename):

    coord_line_pattern = r"^vt"
    texture_coord_dict = {}

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(coord_line_pattern, line)
            if match:
                coord = read_texture_points(line)
                coord_count = len(texture_coord_dict)
                texture_coord_dict[coord_count + 1] = coord

    return texture_coord_dict

def read_texture_points(texture_data_line):

    vertex_elem_pattern = r"[+-]?[0-9]*[.]?[0-9]+[e\+\-\d]*"
    match = re.findall(vertex_elem_pattern, texture_data_line)

    return Point(float(match[0]), float(match[1])) # match[2] is not read

def get_vertices(obj_filename):
    vertex_dict = {}

    vertex_pattern = r"^v\s"
    x_min = sys.float_info.max
    y_min = sys.float_info.max
    z_min = sys.float_info.max

    x_max = sys.float_info.min
    y_max = sys.float_info.min
    z_max = sys.float_info.min

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

                    x_min = min(vert.x, x_min)
                    y_min = min(vert.y, y_min)
                    z_min = min(vert.z, z_min)

                    x_max = max(vert.x, x_max)
                    y_max = max(vert.y, y_max)
                    z_max = max(vert.z, z_max)

                    vertex_count = len(vertex_dict)
                    vertex_dict[vertex_count + 1] = vert
    
    bounding_box = BoundingBox(x_min, y_min, z_min, x_max, y_max, z_max)

    return vertex_dict, bounding_box

def get_texture_color(texture : TinyImage, rel_x : float, rel_y : float):
    return (0, 0, 0)