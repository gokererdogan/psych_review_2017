def rgb2gray(rgb):
    """
    Convert RGB image to grayscale. Uses the same
    parameters with MATLAB's rgb2gray
    """
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray
