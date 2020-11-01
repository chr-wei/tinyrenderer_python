"""Module providing functionality for storing and reading .obj and texture map data."""
import re
from enum import Enum
from collections import namedtuple
from tiny_image import TinyImage

from geom import Vector3D, Point2D, PointUV, comp_min, comp_max

VertexIds = namedtuple("VertexIds", "id_one id_two id_three")
DiffusePointIds = namedtuple("DiffusePointIds", "id_one id_two id_three")
NormalIds = namedtuple("NormalIds", "id_one id_two id_three")
FacedataIds = namedtuple("FacedataIds", "VertexIds DiffusePointIds NormalIds")

class NormalMapType(Enum):
    """Enum specifying normal map type: Global normals or tangent space normals."""
    GLOBAL = 1
    TANGENT = 2

def get_model_face_ids(obj_filename):
    """Returns ids associated to a model's face: Vertex, normal and uv point ids."""
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
    """Returns ids associated to a face of a .obj dataline."""
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
    """Returns all uv points of .obj file."""
    coord_line_pattern = r"^vt"
    diffuse_point_list = []

    with open(obj_filename) as obj_file:
        for line in obj_file:
            match  = re.search(coord_line_pattern, line)
            if match:
                pt_two_d = read_diffuse_points(line)
                diffuse_point_list.append(pt_two_d)

    return diffuse_point_list

def read_diffuse_points(diffuse_data_line):
    """Returns all points of a uv point dataline."""
    vertex_elem_pattern = r"[+-]?[0-9]*[.]?[0-9]+[e\+\-\d]*"
    match = re.findall(vertex_elem_pattern, diffuse_data_line)

    return Point2D(float(match[0]), float(match[1])) # match[2] is not read

def get_vertices(obj_filename):
    """Returns all vertices of .obj file."""
    vertex_list = []

    vertex_pattern = r"^v\s"
    bb_min = Vector3D(float('inf'), float('inf'), float('inf'))
    bb_max = Vector3D(float('-inf'), float('-inf'), float('-inf'))

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

                    vert = Vector3D(*elem_list)
                    vertex_list.append(vert)

                    bb_min = comp_min(vert, bb_min)
                    bb_max = comp_max(vert, bb_max)

    return vertex_list, (bb_min, bb_max)

def get_normals(obj_filename):
    """Returns all vertex normals of .obj file."""
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

                    normal = Vector3D(*elem_list).normalize()
                    normal_list.append(normal)

    return normal_list

class ModelStorage():
    """Class storing model data."""
    object_name = ""
    face_id_data = []
    vertices = []
    normals = []
    bbox =  ()
    diffuse_points = []

    diffuse_map = None
    normal_map_type = NormalMapType.GLOBAL
    normal_map = None
    specular_map = None
    ao_map = None

    def __init__(self, object_name: str = None, obj_filename: str = None,
                 diffuse_map_filename: str = None,
                 normal_map_filename: str = None, normal_map_type = NormalMapType.GLOBAL,
                 specular_map_filename:str = None, ao_map_filename:str = None):

        self.object_name = object_name
        self.face_id_data = get_model_face_ids(obj_filename)
        (self.vertices, self.bbox) = get_vertices(obj_filename)

        # Load texture ('diffuse_map')
        if not diffuse_map_filename is None:
            self.diffuse_points = get_model_diffuse_points(obj_filename)
            self.diffuse_map = TinyImage()
            self.diffuse_map.load_image(diffuse_map_filename)

        # Load normal map
        if not normal_map_filename is None:
            self.normal_map_type = normal_map_type
            self.normals = get_normals(obj_filename)
            self.normal_map = TinyImage()
            self.normal_map.load_image(normal_map_filename)

        # Specular normal map
        if not specular_map_filename is None:
            self.specular_map = TinyImage()
            self.specular_map.load_image(specular_map_filename)

        # Ambient occlusion map
        if not ao_map_filename is None:
            self.ao_map = TinyImage()
            self.ao_map.load_image(ao_map_filename)

    def get_normal(self, face_idx, face_vertex_idx):
        """Returns face vertex normal."""
        normal_idx = self.face_id_data[face_idx].NormalIds[face_vertex_idx]
        return self.normals[normal_idx].normalize()

    def get_vertex(self, face_idx, face_vertex_idx):
        """Returns face vertex."""
        vertex_idx = self.face_id_data[face_idx].VertexIds[face_vertex_idx]
        return self.vertices[vertex_idx]

    def get_uv_map_point(self, face_idx, face_vertex_idx):
        """Returns uv map point."""
        diffuse_idx = self.face_id_data[face_idx].DiffusePointIds[face_vertex_idx]
        return self.diffuse_points[diffuse_idx]

    def get_diffuse_color(self, pnt: PointUV):
        """Returns diffuse color from texture map."""
        # Make sure to only use RGB components in Vector3D
        return Vector3D(self.diffuse_map.get(int(pnt.u * self.diffuse_map.get_width()),
                                             int(pnt.v * self.diffuse_map.get_height()))[:3])

    def get_normal_from_map(self, pnt: PointUV):
        """Returns normal from model normalmap."""
        # Make sure to only use RGB components in Vector3D
        rgb = Vector3D(*self.normal_map.get(int(pnt.u * self.normal_map.get_width()),
                                            int(pnt.v * self.normal_map.get_height()))[:3])
        return (rgb / 255 * 2 - Vector3D(1, 1, 1)).normalize()

    def get_specular_power_from_map(self, pnt: PointUV):
        """Returns specular power coefficient from specular map."""
        # Make sure to only use GRAY component
        comp = self.specular_map.get(int(pnt.u * self.specular_map.get_width()),
                                     int(pnt.v * self.specular_map.get_height()))
        if comp is tuple:
            comp = comp[0]
        return comp

    def get_ao_intensity_from_map(self, pnt: PointUV):
        """Returns ao_intensity from ao map."""
        # Make sure to only use GRAY component
        comp = self.ao_map.get(int(pnt.u * self.ao_map.get_width()),
                                     int(pnt.v * self.ao_map.get_height()))
        if comp is tuple:
            comp = comp[0]
        return comp / 255.0

    def get_vertex_count(self):
        """Returns count of model vertices."""
        return len(self.vertices)

    def get_face_count(self):
        """Returns count of model faces."""
        return len(self.face_id_data)
