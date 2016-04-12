import cv
import cv2
import math
import numpy as np
from PIL import Image

def rotate_bbox(bbox, angle):
    """
    Rotate given bounding box by angle degrees by modifying bounding box
    coordinates accordingly.

    Parameters
    ----------
    bbox: tuple
          tuple of integers defining coordinates of the form
          (x1, y1, x2, y2, x3, y3, x4, y4).
    angle: int
           integer defining the amount of rotation in degrees.
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    xc = (x1 + x3) / 2
    yc = (y1 + y3) / 2
    x1_rot, y1_rot = rotate_coords(x1, y1, xc, yc, angle)
    x2_rot, y2_rot = rotate_coords(x2, y2, xc, yc, angle)
    x3_rot, y3_rot = rotate_coords(x3, y3, xc, yc, angle)
    x4_rot, y4_rot = rotate_coords(x4, y4, xc, yc, angle)

    return (x1_rot, y1_rot, x2_rot, y2_rot, x3_rot, y3_rot, x4_rot, y4_rot)


def rotate_coords(x, y, xc, yc, angle):
    """
    Compute the new value of the given coordinates (x, y)
    with the corresponding center of the rotation and the angle.

    Parameters
    ----------
    x: int
       x coordinate on which to apply the rotation.
    y: int
       y coordinate on which to apply the rotation.
    xc: int
        x coordinate of the rotation center.
    yc: int
        y coordinate of the rotation center.
    angle: int
           angle, in degrees, of the rotation.
    """
    cosin = math.cos(math.radians(angle))
    sin = math.sin(math.radians(angle))
    x_rot = cosin * (x - xc) - sin * (y - yc) + xc
    y_rot = sin * (x - xc) + cosin * (y - yc) + yc
    return x_rot, y_rot


def translate_bbox(bbox, translation):
    """
    Translate given bbox by the amount in translation.

    Parameters
    ----------
    bbox: tuple
          tuple of integers defining coordinates of the form
          (x1, y1, x2, y2, x3, y3, x4, y4).
    translation: tuple
                 tuple of integers defining the translation to apply on
                 the x axis (translation[0]) and the y axis (translation[1]).
    """
    new_bbox = (bbox[0] + translation[0],
                bbox[1] + translation[1],
                bbox[2] + translation[0],
                bbox[3] + translation[1],
                bbox[4] + translation[0],
                bbox[5] + translation[1],
                bbox[6] + translation[0],
                bbox[7] + translation[1])
    return new_bbox


def scale_bbox(bbox, scalefactor):
    """
    Increase or decrease given bounding box by the scalefactor.

    Parameters
    ----------
    bbox: tuple
          tuple of integers defining coordinates of the form
          (x1, y1, x2, y2, x3, y3, x4, y4).
    scalefactor: float
                 factor of the scaling.
                 To reduce the bbox by half, scalefactor should be 0.5.
    """
    xc = (bbox[0] + bbox[4]) / 2
    yc = (bbox[1] + bbox[5]) / 2
    new_w = (bbox[4] - bbox[0]) * scalefactor
    new_h = (bbox[5] - bbox[1]) * scalefactor
    x1 = xc - new_w / 2
    y1 = yc - new_h / 2
    x2 = xc + new_w / 2
    y2 = yc - new_h / 2
    x3 = xc + new_w / 2
    y3 = yc + new_h / 2
    x4 = xc - new_w / 2
    y4 = yc + new_h / 2
    return (x1, y1, x2, y2, x3, y3, x4, y4)


def transform_bbox(bbox, angle, translation, scalefactor):
    """
    Return a new bounding box transformed by all 3 transformations.
    Rotation should be applied at the end.

    Parameters
    ----------
    bbox: tuple
          tuple of integers defining coordinates of the form
          (x1, y1, x2, y2, x3, y3, x4, y4).
    angle: int
           angle, in degrees, of the rotation.
    translation: tuple
                 tuple of integers defining the translation to apply on
                 the x axis (translation[0]) and the y axis (translation[1]).
    scalefactor: float
                 factor of the scaling.
    """
    trans_bbox = translate_bbox(bbox, translation)
    scaled_bbox = scale_bbox(trans_bbox, scalefactor)
    rot_bbox = rotate_bbox(scaled_bbox, angle)
    return rot_bbox
