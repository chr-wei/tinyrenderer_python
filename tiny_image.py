"""tiny_image module for TinyImage class used in tiny_renderer."""
from PIL import Image
from PIL import ImageDraw

class TinyImage:
    """
    This is the TinyImage class.

    Examples:
        new_image = TinyImage(100, 100)
        new_image = TinyImage().load_image("path/to/image/tiny.png")
        new_image = new_image.set(x = 10, y = 100, color = 'white')
    """

    _im: Image
    _draw: ImageDraw

    def __init__(self, width = None, height = None):
        if not width is None and not height is None:
            self._im = Image.new(size=(width, height), mode="RGB", color="lightgray")
            self._draw = ImageDraw.Draw(self._im)

    def load_image(self, ipath):
        """Loads image from disk."""
        self._im = Image.open(ipath)

    def set(self, x, y, color): # pylint: disable=invalid-name
        """Draw a point onto the image."""
        self._draw.point((x, self.get_height() - y - 1), fill = color)

    def get(self, x, y): # pylint: disable=invalid-name
        """Read color of image."""
        # Read pixel color.
        return self._im.getpixel((x, self.get_height() - y - 1))

    def save_to_disk(self, fname):
        """Save your image to a given filename."""
        self._im.save(fname)

    def get_width(self):
        """Get width of image."""
        return self._im.width

    def get_height(self):
        """Get height of image."""
        return self._im.height
