from PIL import Image
from PIL import ImageDraw

class TinyImage:
    """Get a new image canvas to draw on."""

    def __init__(self, width, height):
        self._im = Image.new(size=(width, height), mode="RGB")
        self._draw = ImageDraw.Draw(self._im)
        self.width = self._im.width
        self.height = self._im.height

    def set(self, x,y, color):
        """Draw a point onto the image."""
        self._draw.point((x,y), fill=color)

    def save_to_disk(self, fname):
        """Save your image to a given filename."""

        self._im = self._im.transpose(Image.FLIP_TOP_BOTTOM)
        self._im.save(fname)
    
    def get_width(self):
        return self._im.width

    def get_height(self):
        return self._im.height