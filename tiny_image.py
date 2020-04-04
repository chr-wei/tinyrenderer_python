from PIL import Image
from PIL import ImageDraw

class TinyImage:
    """Get a new image canvas to draw on."""

    def __init__(self, width, height):
        self.im = Image.new(size=(width, height), mode="RGB")
        self.width = self.im.width
        self.height = self.im.height
        self.draw = ImageDraw.Draw(self.im)

    def set(self, x,y, color):
        """Draw a point onto the image."""
        self.draw.point((x,y), fill=color)

    def save_to_disk(self, fname):
        """Save your image to a given filename."""

        self.im = self.im.transpose(Image.FLIP_TOP_BOTTOM)
        self.im.save(fname)

    def get_size(self):
        return ()