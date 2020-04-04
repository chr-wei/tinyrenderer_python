
def line(x0, y0, x1, y1, image, color):
    """Draw a line onto an image."""

    if abs(x1-x0) < abs(y1-y0):
        # Swap to prevent whitespace when y distance is higher than x distance (steep line)
        steep_line = True
        (y1, y0, x1, x0) = (x1, x0, y1, y0)
    else:
        steep_line = False

    if x0 > x1: 
        (y1, y0, x1, x0) = (y0, y1, x0, x1)

    for x in range(x0, x1):
        #ToDo: Optimize speed using non float operations
        y = y0 + (y1-y0) / (x1-x0) * (x-x0)

        if steep_line:
            image.set(y, x, color)
        else:
            image.set(x, y, color)

    return image



def triangle(p0, p1, p2, image, color):
    image = line(p0[0], p0[1], p1[0], p1[1], image, color)
    image = line(p1[0], p1[1], p2[0], p2[1], image, color)
    image = line(p2[0], p2[1], p0[0], p0[1], image, color)
    return image