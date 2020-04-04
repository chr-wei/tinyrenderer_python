from PIL import Image
from PIL import ImageDraw

class TinyImage:
    """Get a new image canvas to draw on."""

    def __init__(self, width, height):
        self.im = Image.new(size=(width, height), mode="RGB")
        # Flip image to have (0,0) bottom left
        self.draw = ImageDraw.Draw(self.im)

    def set(self, x,y, color):
        """Draw a point onto the image."""
        self.draw.point((x,y), fill=color)

    def save_to_disk(self, fname):
        """Save your image to a given filename."""

        self.im = self.im.transpose(Image.FLIP_TOP_BOTTOM)
        self.im.save(fname)