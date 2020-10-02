from PIL import Image
from PIL import ImageDraw

class TinyImage:
    """Get a new image canvas to draw on."""

    def __init__(self, width = None, height = None):
        if width != None and height != None:
            self._im = Image.new(size=(width, height), mode="RGB")
            self._draw = ImageDraw.Draw(self._im)
            self.width = self._im.width
            self.height = self._im.height

    def load_image(self, ipath):
        self._im = Image.open(ipath).transpose(Image.FLIP_TOP_BOTTOM)
        self._width = self._im.width
        self._height = self._im.height

    def set(self, x, y, color):
        """Draw a point onto the image."""
        self._draw.point((x,y), fill=color)

    def get(self, x, y):
        """Read color of image."""
        # Read pixel color.
        return self._im.getpixel((x,y))

    def save_to_disk(self, fname):
        """Save your image to a given filename."""
        imc = self._im.copy()
        imc.transpose(Image.FLIP_TOP_BOTTOM).save(fname)
    
    def get_width(self):
        return self._im.width

    def get_height(self):
        return self._im.height