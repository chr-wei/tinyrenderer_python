from tiny_image import TinyImage
from our_gl import triangle
from model import get_model_face_ids, get_vertices

if __name__ == "__main__":
    image = TinyImage(600, 600)
    
    #Excercises
    #image.set(50,30, "red") ##1
    #image = our_gl.line(0, 0, 100, 20, image, "white")##2
    #image = our_gl.line(0, 0, 20, 100, image, "white")##3
    #image = triangle((3,5), (20,100), (110,50), image, "white")##4
    
    face_id_data = get_model_face_ids("obj/head.obj")
    vertices = get_vertices("obj/head.obj")
    
    for id, face in face_id_data.items():
        vert_ids = face.Vertex_Ids
        v0 = vertices[vert_ids.id_one]
        v1 = vertices[vert_ids.id_two]
        v2 = vertices[vert_ids.id_three]

        x0 = int((v0.x+1.)*image.width/2)
        y0 = int((v0.y+1.)*image.height/2)

        x1 = int((v1.x+1.)*image.width/2)
        y1 = int((v1.y+1.)*image.height/2) 

        x2 = int((v2.x+1.)*image.width/2) 
        y2 = int((v2.y+1.)*image.height/2)

        image = triangle((x0, y0), (x1, y1), (x2, y2), image, "white")
     
        
    image.save_to_disk("out.png")