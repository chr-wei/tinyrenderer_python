from tiny_image import TinyImage
from our_gl import triangle


if __name__ == "__main__":
    image = TinyImage(200,200)
    
    #Excercises
    #image.set(50,30, "red") ##1
    #image = our_gl.line(0, 0, 100, 20, image, "white")##2
    #image = our_gl.line(0, 0, 20, 100, image, "white")##3
    image = triangle((3,5), (20,100), (110,50), image, "white")##4
    
    image.save_to_disk("out.png")