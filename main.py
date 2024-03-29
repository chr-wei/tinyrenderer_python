"""A tiny shader fork written in Python 3"""
import progressbar

from tiny_image import TinyImage
import our_gl as gl
from geom import ScreenCoords, Vector3D
from model import ModelStorage, NormalMapType

from tiny_shaders import TinyShader, DepthShader

if __name__ == "__main__":
    # Model property definition
    OBJ_FILENAME = "obj/autumn/autumn.obj"
    DIFFUSE_FILENAME = "obj/autumn/TEX_autumn_body_color.tga"
    NORMAL_MAP_FILENAME = "obj/autumn/TEX_autumn_body_normals_wrld.tga"
    NORMAL_MAP_TYPE = NormalMapType.GLOBAL
    SPECULAR_MAP_FILENAME = "obj/autumn/TEX_autumn_body_spec.tga"
    AO_MAP_FILENAME = "obj/autumn/TEX_autumn_body_ao.tga"
    OUTPUT_FILENAME = "renders/out.png"

    # Image property definition
    (w, h) = (800, 800)
    image = TinyImage(w, h)

    # View property definition
    VIEW_PROP_SET = 0
    EYE = Vector3D(0, 0, 4) # Lookat camera 'EYE' position
    CENTER = Vector3D(0, 0, 0) # Lookat 'CENTER'. 'EYE' looks at CENTER
    UP = Vector3D(0, 1, 0) # Camera 'UP' direction
    SCALE = .8 # Viewport scaling
    
    # Light property
    LIGHT_DIR = Vector3D(1, 1, 1).normalize()

    print("Reading modeldata ...")
    mdl = ModelStorage(object_name = "autumn", obj_filename=OBJ_FILENAME,
                       diffuse_map_filename=DIFFUSE_FILENAME,
                       normal_map_filename=NORMAL_MAP_FILENAME, normal_map_type=NORMAL_MAP_TYPE,
                       specular_map_filename=SPECULAR_MAP_FILENAME, ao_map_filename=AO_MAP_FILENAME)

    # Define tranformation matrices

    # Generate model transformation matrix which transforms
    # vertices according to the model bounding box
    #
    # min[-1, -1, -1] to max[1, 1, 1] object space
    M_model = gl.model_transform(mdl.bbox[0], mdl.bbox[1])

    # Generate cam transformation
    M_lookat = gl.lookat(EYE, CENTER, UP)

    # Generate perspective transformation
    z_cam_dist = (EYE - CENTER).abs()
    M_perspective = gl.perspective(z_cam_dist)

    # Generate transformation to final viewport
    DEPTH_RES = 255 # [0 ... 255]
    M_viewport = gl.viewport(+SCALE*w/8, +SCALE*h/8, SCALE*w, SCALE*h, DEPTH_RES)

    # Combine matrices
    M_modelview = M_lookat * M_model
    M_pe = M_perspective * M_modelview
    M_sc = M_viewport * M_pe

    M_pe_IT = M_pe.tr().inv()

    # Init vars for normal shader run
    zbuffer = [[-float('Inf') for bx in range(w)] for y in range(h)]
    screen_coords = ScreenCoords(9*[0])

    # Init vars for shaders which use a shadow buffer
    shadow_buffer = None
    shadow_image = None
    M_sb = None

    PREPARE_AO_SHADER = False
    PREPARE_SHADOW_SHADER = False

    # Shader definition
    PREPARE_SHADOW_SHADER = True
    shader = TinyShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT, None, None)

    if PREPARE_SHADOW_SHADER:
        # Fill shadow buffer and set data for final shader

        # Calculate shadow buffer matrices
        M_lookat_cam_light = gl.lookat(LIGHT_DIR, CENTER, UP)
        M_sb = M_viewport * M_lookat_cam_light * M_model

        shadow_buffer = [[-float('Inf') for bx in range(w)] for y in range(h)]
        shadow_image = TinyImage(w, h)

        # Depth shader and shadow buffer
        depth_shader = DepthShader(mdl, M_sb, DEPTH_RES)

        # Apply data to normal shader
        shader.uniform_M_sb = M_sb
        shader.shadow_buffer = shadow_buffer

        print("Saving shadow buffer ...")
        for face_idx in progressbar.progressbar(range(mdl.get_face_count())):
            for face_vert_idx in range(3):
                # Get transformed vertex and prepare internal shader data
                vert = depth_shader.vertex(face_idx, face_vert_idx)
                screen_coords = screen_coords.set_col(face_vert_idx, vert)

            # Rasterize image (z heigth in dir of light). Shadow buffer is filles as well
            shadow_image = gl.draw_triangle(screen_coords, depth_shader, shadow_buffer, 
                                            shadow_image)

        shadow_image.save_to_disk("renders/shadow_buffer.png")

    # Final shader run
    print("Drawing triangles ...")
    for face_idx in progressbar.progressbar(range(mdl.get_face_count())):
        for face_vert_idx in range(3):
            # Get transformed vertex and prepare internal shader data
            vert = shader.vertex(face_idx, face_vert_idx)
            screen_coords = screen_coords.set_col(face_vert_idx, vert)

        # Rasterize triangle
        image = gl.draw_triangle(screen_coords, shader, zbuffer, image)

    image.save_to_disk(OUTPUT_FILENAME)
