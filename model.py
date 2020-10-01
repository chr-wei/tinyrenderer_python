import re
import sys

import our_gl as gl
from collections import namedtuple
from tiny_image import TinyImage

from numpy import array
from enum import Enum
from geom import Vector_3D, Point_2D, comp_min, comp_max

VertexIds = namedtuple("VertexIds", "id_one id_two id_three")
DiffusePointIds = namedtuple("DiffusePointIds", "id_one id_two id_three")
NormalIds = namedtuple("NormalIds", "id_one id_two id_three")
FacedataIds = namedtuple("FacedataIds", "VertexIds DiffusePointIds NormalIds")

class NormalMapType(Enum):
    GLOBAL = 1
    TANGENT = 2

def get_model_face_ids(obj_filename):

    face_line_pattern = r"^f"
    face_id_data_list = []

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(face_line_pattern, line)
            if match:
                face_data = read_face_ids(line)
                face_id_data_list.append(face_data)

    return face_id_data_list

def read_face_ids(face_data_line):

    face_elem_pattern = r"(\d+)\/(\d*)\/(\d+)"
    match = re.findall(face_elem_pattern, face_data_line)

    vert_list = []
    diffuse_point_list = []
    norm_list = []

    for idx in range(0, len(match)):
        # Decrease all indices as .obj files are indexed starting at one
        vert_list.append(int(match[idx][0]) - 1)
        
        diffuse_point_id = match[idx][1]
        if diffuse_point_id.isdigit():
            diffuse_point_id = int(diffuse_point_id)
            diffuse_point_list.append(diffuse_point_id - 1)
        else:
            diffuse_point_list.append(None)

        norm_list.append(int(match[idx][2]) - 1)

    vert_ids = VertexIds(*vert_list[:3])
    diffuse_pt_ids = DiffusePointIds(*diffuse_point_list[:3])
    norm_ids = NormalIds(*norm_list[:3])

    return FacedataIds(vert_ids, diffuse_pt_ids, norm_ids)

def get_model_diffuse_points(obj_filename):

    coord_line_pattern = r"^vt"
    diffuse_point_list = []

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(coord_line_pattern, line)
            if match:
                pt = read_diffuse_points(line)
                diffuse_point_list.append(pt)

    return diffuse_point_list

def read_diffuse_points(diffuse_data_line):

    vertex_elem_pattern = r"[+-]?[0-9]*[.]?[0-9]+[e\+\-\d]*"
    match = re.findall(vertex_elem_pattern, diffuse_data_line)

    return Point_2D(float(match[0]), float(match[1])) # match[2] is not read

def get_vertices(obj_filename):
    vertex_list = []

    vertex_pattern = r"^v\s"
    bb_min = Vector_3D(float('inf'), float('inf'), float('inf'))
    bb_max = Vector_3D(float('-inf'), float('-inf'), float('-inf'))

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

                    vert = Vector_3D(*elem_list)
                    vertex_list.append(vert)

                    bb_min = comp_min(vert, bb_min)
                    bb_max = comp_max(vert, bb_max)

    return vertex_list, (bb_min, bb_max)

def get_normals(obj_filename):
    normal_list = []

    normal_pattern = r"^vn\s"

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(normal_pattern, line)
            if match:
                normal_elem_pattern = r"[+-]?[0-9]*[.]?[0-9]+[e\+\-\d]*"

                match = re.findall(normal_elem_pattern, line)
                elem_list = []
                if match:
                    for elem in match:
                        elem_list.append(float(elem))

                    vert = Vector_3D(*elem_list)
                    normal_list.append(vert)

    return normal_list

class Model_Storage():
    face_id_data = []
    vertices = []
    normals = []
    bbox =  ()
    diffuse_points = []

    diffuse_map = None
    diffuse_map_w = 0
    diffuse_map_h = 0

    normal_map_type = NormalMapType.GLOBAL
    normal_map = None
    normal_map_w = 0
    normal_map_h = 0

    def __init__(self, object_name: str = None, obj_filename: str = None, 
                 diffuse_map_filename: str = None, 
                 normal_map_filename: str = None, normal_map_type = NormalMapType.GLOBAL):

        self.face_id_data = get_model_face_ids(obj_filename)
        (self.vertices, self.bbox) = get_vertices(obj_filename)
        
        # Load texture ('diffuse_map')
        if not diffuse_map_filename is None:
            self.diffuse_points = get_model_diffuse_points(obj_filename)
            self.diffuse_map = TinyImage()
            self.diffuse_map.load_image(diffuse_map_filename)
            self.diffuse_map_w = self.diffuse_map.get_width()
            self.diffuse_map_h = self.diffuse_map.get_height()

        # Load normal map
        if not normal_map_filename is None:
            self.normal_map_type = normal_map_type
            self.normals = get_normals(obj_filename)
            self.normal_map = TinyImage()
            self.normal_map.load_image(normal_map_filename)
            self.normal_map_w = self.normal_map.get_width()
            self.normal_map_h = self.normal_map.get_height()

    def get_normal(self, face_idx, face_vertex_idx):
        normal_idx = self.face_id_data[face_idx].NormalIds[face_vertex_idx]
        return self.normals[normal_idx]

    def get_vertex(self, face_idx, face_vertex_idx):
        vertex_idx = self.face_id_data[face_idx].VertexIds[face_vertex_idx]
        return self.vertices[vertex_idx]

    def get_uv_map_point(self, face_idx, face_vertex_idx):
        diffuse_idx = self.face_id_data[face_idx].DiffusePointIds[face_vertex_idx]
        return self.diffuse_points[diffuse_idx]

    def get_diffuse_color(self, rel_x, rel_y):
        return Vector_3D(*self.diffuse_map.get(int(rel_x * self.diffuse_map_w), int(rel_y * self.diffuse_map_h)))
    
    def get_normal_from_map(self, rel_x, rel_y):
        if self.normal_map_type == NormalMapType.GLOBAL:
            rgb = Vector_3D(*self.normal_map.get(int(rel_x * self.normal_map_w), 
                                                  int(rel_y * self.normal_map_h)))
            return (rgb / 255 * 2 - Vector_3D(1, 1, 1)).norm()

        elif self.normal_map_type == NormalMapType.TANGENT:
            return None
        else:
            return None
    
    def get_vertex_count(self):
        return len(self.vertices)

    def get_face_count(self):
        return len(self.face_id_data)