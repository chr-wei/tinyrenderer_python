### Python port of ssloy/tinyrenderer in Python3

To install just download the code. 
Make sure to have python3 running and the following libs are installed:

`pip install pillow`

Blender generate .obj-file steps:

1. object mode -> Object -> Convert to -> Mesh from Curve/Meta/Surf/Text
1. remove all keyframes
1. delete rigs, cam, particles ...
1. move objects

1. add Decimate modifier to one object
1. select all in viewpane
1. ctrl + l -> Modifier to add last selected modifier in 1. sidepane to all objects
1. set origin to 3d cursor

1. select all
1. export obj (triangulate, no obj groups, selected only, write normals)