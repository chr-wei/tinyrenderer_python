"""A tiny shader fork written in Python 3"""

from progressbar import progressbar

from tiny_image import TinyImage
import our_gl as gl
from geom import ScreenCoords, Vector3D
from model import ModelStorage, NormalMapType
from tiny_shaders import FlatShader, GouraudShader, GouraudShaderSegregated, \
                         DiffuseGouraudShader, GlobalNormalmapShader, SpecularmapShader, \
                         TangentNormalmapShader

if __name__ == "__main__":
    # Model property selection
    MODEL_PROP_SET = 1
    if MODEL_PROP_SET == 0:
        OBJ_FILENAME = "obj/autumn/autumn.obj"
        DIFFUSE_FILENAME = "obj/autumn/TEX_autumn_body_color.png"
        NORMAL_MAP_FILENAME = "obj/autumn/TEX_autumn_body_normals_wrld_space.tga"
        NORMAL_MAP_TYPE = NormalMapType.GLOBAL
        SPECULAR_MAP_FILENAME = "obj/autumn/TEX_autumn_body_spec.tga"
        OUTPUT_FILENAME = "renders/out.png"
    elif MODEL_PROP_SET == 1:
        OBJ_FILENAME = "obj/head/head.obj"
        DIFFUSE_FILENAME = "obj/head/head_diffuse.tga"
        NORMAL_MAP_FILENAME = "obj/head/head_nm_tangent.tga"
        NORMAL_MAP_TYPE = NormalMapType.TANGENT
        SPECULAR_MAP_FILENAME = "obj/head/head_spec.tga"
        OUTPUT_FILENAME = "renders/out.png"
    else:
        OBJ_FILENAME = "obj/head/head.obj"
        DIFFUSE_FILENAME = "obj/head/head_diffuse.tga"
        NORMAL_MAP_FILENAME = "obj/head/head_nm.tga"
        NORMAL_MAP_TYPE = NormalMapType.GLOBAL
        SPECULAR_MAP_FILENAME = "obj/head/head_spec.tga"
        OUTPUT_FILENAME = "renders/out.png"

    # Image property selection
    IMG_PROP_SET = 1
    if IMG_PROP_SET == 0:
        (w, h) = (2000, 2000)
    else:
        (w, h) = (800, 800)

    image = TinyImage(w, h)

    # View property selection
    VIEW_PROP_SET = 1
    if VIEW_PROP_SET == 0:
        EYE = Vector3D(0, 0, 1) # Lookat camera 'EYE' position
        CENTER = Vector3D(0, 0, 0) # Lookat 'CENTER'. 'EYE' looks at CENTER
        UP = Vector3D(0, 1, 0) # Camera 'UP' direction
        SCALE = .8 # Viewport scaling
    elif VIEW_PROP_SET == 1:
        EYE = Vector3D(1, 0, 1)
        CENTER = Vector3D(0, 0, 0)
        UP = Vector3D(0, 1, 0)
        SCALE = .8
    else:
        EYE = Vector3D(1, 0, 0) # Lookat camera 'EYE' position
        CENTER = Vector3D(0, 0, 0) # Lookat 'CENTER'. 'EYE' looks at CENTER
        UP = Vector3D(0, 1, 0) # Camera 'UP' direction
        SCALE = .8 # Viewport scaling

    # Light property
    LIGHT_DIR = Vector3D(1, 0, 1).normalize()

    print("Reading modeldata ...")
    mdl = ModelStorage(object_name = "autumn", obj_filename=OBJ_FILENAME,
                       diffuse_map_filename=DIFFUSE_FILENAME,
                       normal_map_filename=NORMAL_MAP_FILENAME, normal_map_type=NORMAL_MAP_TYPE,
                       specular_map_filename=SPECULAR_MAP_FILENAME)

    # Define tranformation matrices

    # Generate model transformation matrix which transforms
    # vertices according to the model bounding box
    #
    # min[-1, -1, -1] to max[1, 1, 1] object space
    M_model = gl.model_transform(mdl.bbox[0], mdl.bbox[1])

    # Generate cam transformation
    M_lookat = gl.lookat(EYE, CENTER, UP)

    # Generate perspective transformation
    M_perspective = gl.perspective(4.0)

    # Generate transformation to final viewport
    M_viewport = gl.viewport(+SCALE*w/8, +SCALE*h/8, SCALE*w, SCALE*h, 255)

    # Combine matrices
    M_modelview = M_lookat * M_model
    M_pe = M_perspective * M_modelview
    M_sc = M_viewport * M_pe

    M_pe_IT = M_pe.tr().inv()

    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)]

    SHADER_PROP_SET = 5
    if SHADER_PROP_SET == 0:
        shader = GouraudShader(mdl, LIGHT_DIR, M_sc)
    elif SHADER_PROP_SET == 1:
        shader = GouraudShaderSegregated(mdl, LIGHT_DIR, M_sc, 4)
    elif SHADER_PROP_SET == 2:
        shader = DiffuseGouraudShader(mdl, LIGHT_DIR, M_sc)
    elif SHADER_PROP_SET == 3:
        shader = GlobalNormalmapShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT)
    elif SHADER_PROP_SET == 4:
        shader = SpecularmapShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT)
    elif SHADER_PROP_SET == 5:
        shader = TangentNormalmapShader(mdl, LIGHT_DIR, M_pe, M_pe_IT, M_viewport)
    else:
        shader = FlatShader(mdl, LIGHT_DIR, M_sc)

    # Iterate model faces
    print("Drawing triangles ...")

    screen_coords = ScreenCoords(9*[0])

    for face_idx in progressbar(range(mdl.get_face_count())):
        for face_vert_idx in range(3):
            # Get transformed vertex and prepare internal shader data
            vert = shader.vertex(face_idx, face_vert_idx)
            screen_coords = screen_coords.set_col(face_vert_idx, vert)

        # Rasterize triangle
        image = gl.draw_triangle(screen_coords, shader, zbuffer, image)

    image.save_to_disk(OUTPUT_FILENAME)
