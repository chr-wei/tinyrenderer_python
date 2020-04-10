from PIL import Image
from PIL import ImageDraw

class TinyImage:
    """Get a new image canvas to draw on."""

    def __init__(self, width, height):
        self.__im = Image.new(size=(width, height), mode="RGB")
        self.width = self.__im.width
        self.height = self.__im.height

    def set(self, x,y, color):
        """Draw a point onto the image."""
        ImageDraw.Draw(self.__im).point((x,y), fill=color)

    def save_to_disk(self, fname):
        """Save your image to a given filename."""

        self.__im = self.__im.transpose(Image.FLIP_TOP_BOTTOM)
        self.__im.save(fname)
    
    def get_width(self):
        return self.__im.width

    def get_height(self):
        return self.__im.height