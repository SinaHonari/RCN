import numpy as np

def pol_to_cart(deg, radius=1.0):
    # This method transforms degree to cartesian values
    # deg is a value in the range [-180, 180]

    # conveting deg to rad
    rad = np.array(deg) * (np.pi / 180.)
    x = radius * np.cos(rad)
    y = radius * np.sin(rad)
    return [x, y]

def cart_to_pol(x, y):
    # This method takes x, y values and convert
    # it to the degree in the range [-180, 180]
    deg = np.arctan2(y, x) * 180 / np.pi
    return deg
