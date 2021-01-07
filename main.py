"""A tiny shader fork written in Python 3"""
import progressbar

from tiny_image import TinyImage
import our_gl as gl
from geom import ScreenCoords, Vector3D
from model import ModelStorage, NormalMapType

from tiny_shaders import FlatShader, GouraudShader, GouraudShaderSegregated, \
                         DiffuseGouraudShader, GlobalNormalmapShader, SpecularmapShader, \
                         TangentNormalmapShader, DepthShader, SpecularShadowShader, \
                         ZShader, AmbientOcclusionShader, AmbientOcclusionMapShader, \
                         TinyShader

if __name__ == "__main__":
    # Model property selection
    MODEL_PROP_SET = 1
    if MODEL_PROP_SET == 0:
        OBJ_FILENAME = "obj/autumn/autumn.obj"
        DIFFUSE_FILENAME = "obj/autumn/TEX_autumn_body_color.tga"
        NORMAL_MAP_FILENAME = "obj/autumn/TEX_autumn_body_normals_tngt.tga"
        NORMAL_MAP_TYPE = NormalMapType.TANGENT
        SPECULAR_MAP_FILENAME = "obj/autumn/TEX_autumn_body_spec.tga"
        AO_MAP_FILENAME = "obj/autumn/TEX_autumn_body_ao.tga"
        OUTPUT_FILENAME = "renders/out.png"
    elif MODEL_PROP_SET == 1:
        OBJ_FILENAME = "obj/autumn/autumn.obj"
        DIFFUSE_FILENAME = "obj/autumn/TEX_autumn_body_color.tga"
        NORMAL_MAP_FILENAME = "obj/autumn/TEX_autumn_body_normals_wrld.tga"
        NORMAL_MAP_TYPE = NormalMapType.GLOBAL
        SPECULAR_MAP_FILENAME = "obj/autumn/TEX_autumn_body_spec.tga"
        AO_MAP_FILENAME = "obj/autumn/TEX_autumn_body_ao.tga"
        OUTPUT_FILENAME = "renders/out.png"
    elif MODEL_PROP_SET == 2:
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
        (w, h) = (200, 200)
    elif IMG_PROP_SET == 1:
        (w, h) = (800, 800)
    elif IMG_PROP_SET == 2:
        (w, h) = (2000, 2000)
    else:
        raise ValueError

    image = TinyImage(w, h)

    # View property selection
    VIEW_PROP_SET = 0
    if VIEW_PROP_SET == 0:
        EYE = Vector3D(0, 0, 4) # Lookat camera 'EYE' position
        CENTER = Vector3D(0, 0, 0) # Lookat 'CENTER'. 'EYE' looks at CENTER
        UP = Vector3D(0, 1, 0) # Camera 'UP' direction
        SCALE = .8 # Viewport scaling
    elif VIEW_PROP_SET == 1:
        EYE = Vector3D(2.828, 0, 2.828)
        CENTER = Vector3D(0, 0, 0)
        UP = Vector3D(0, 1, 0)
        SCALE = .8
    elif VIEW_PROP_SET == 2:
        EYE = Vector3D(0, 2, 0)
        CENTER = Vector3D(0, 0, 0)
        UP = Vector3D(1, 0, 0)
        SCALE = .8
    else:
        EYE = Vector3D(4, 0, 0) # Lookat camera 'EYE' position
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

    SHADER_PROP_SET = 10
    if SHADER_PROP_SET == 0:
        shader = FlatShader(mdl, LIGHT_DIR, M_sc)
    elif SHADER_PROP_SET == 1:
        shader = GouraudShader(mdl, LIGHT_DIR, M_sc)
    elif SHADER_PROP_SET == 2:
        shader = GouraudShaderSegregated(mdl, LIGHT_DIR, M_sc, 4)
    elif SHADER_PROP_SET == 3:
        shader = DiffuseGouraudShader(mdl, LIGHT_DIR, M_sc)
    elif SHADER_PROP_SET == 4:
        shader = GlobalNormalmapShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT)
    elif SHADER_PROP_SET == 5:
        shader = SpecularmapShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT)
    elif SHADER_PROP_SET == 6:
        shader = TangentNormalmapShader(mdl, LIGHT_DIR, M_pe, M_pe_IT, M_viewport)
    elif SHADER_PROP_SET == 7:
        PREPARE_SHADOW_SHADER = True
        shader = SpecularShadowShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT, None, None)
    elif SHADER_PROP_SET == 8:
        PREPARE_AO_SHADER = True
        shader = AmbientOcclusionShader(mdl, M_sc, None, w, h)
    elif SHADER_PROP_SET == 9:
        shader = AmbientOcclusionMapShader(mdl, M_sc)
    elif SHADER_PROP_SET == 10:
        PREPARE_SHADOW_SHADER = True
        shader = TinyShader(mdl, LIGHT_DIR, M_pe, M_sc, M_pe_IT, None, None)
    else:
        raise ValueError

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

    if PREPARE_AO_SHADER:
        # Fill zbuffer and set data for final shader
        print("Precalculating zbuffer for ambient occlusion shader ...")
        precalc_z_buffer = [[-float('Inf') for bx in range(w)] for y in range(h)]
        z_image = TinyImage(w, h)
        z_shader = ZShader(mdl, M_sc)

        for face_idx in progressbar.progressbar(range(mdl.get_face_count())):
            for face_vert_idx in range(3):
                # Get transformed vertex and prepare internal shader data
                vert = z_shader.vertex(face_idx, face_vert_idx)
                screen_coords = screen_coords.set_col(face_vert_idx, vert)

            # Rasterize triangle
            image = gl.draw_triangle(screen_coords, z_shader, precalc_z_buffer, z_image)

        # Prepare and set data for the next shader
        shader.uniform_precalc_zbuffer = precalc_z_buffer

        # Lower precalc z buffer a little bit to make next shader raster the top pixels but
        # not all pixels since AO shader calls are expensive
        for x_sc in range(w):
            for y_sc in range(h):
                zbuffer[x_sc][y_sc] = precalc_z_buffer[x_sc][y_sc] - 1e-10

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