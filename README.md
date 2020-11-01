### Python port of ssloy/tinyrenderer in Python3

To install just download the code. 
Make sure to have python3 running and the following libs are installed:

`pip install pillow`

## What is special about this fork?

For the general idea please visit ssloy's tinyrenderer repository.

This fork is intended to be a starting point of teaching programming in python3 to everyone interested in programming.
With this in mind I tried to follow ssloys tutorial closely and chose some guidelines:

1. Simplicity over perfection
1. Readability over performance

To accomplish these goals I included a special linear algebra module `geom.py` instead of using `numpy`. 
Numpy is undoubtly more universal and elaborated but harder to read for beginners.

The `geom.py` module contains some basic vector and matrix classes and supports indexing via '.x', '.y' and '.z' notation based on namedtuples.

After finishing the fork I am going to be working on course material and structured lessons to roll out this exciting course of 3D animation and programming sometime in the future.

## Here are some examples generated by the renderer

### Rendered image with shadows and specular lighting (current optimum)
![Rendered image with shadows and specular lighting](https://github.com/rap1ide/tinyrenderer_python/blob/master/docs/images/shadow_shade.png?raw=true)

### Rendered wire mesh
<img src="https://github.com/rap1ide/tinyrenderer_python/blob/master/docs/images/e04_autumn_mesh.png" alt="Rendered wire mesh" width="400">

### Rendered randomly colored filled mesh triangles
![Randomly colored filled mesh triangles](https://github.com/rap1ide/tinyrenderer_python/blob/master/docs/images/e06_autumn_filled.png?raw=true)
