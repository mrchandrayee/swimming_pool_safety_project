import cv2
import numpy as np

def polygon_pixels(normalized, width, height):
    pts = np.array([[round(x*width), round(y*height)] for x,y in normalized],
                   dtype=np.int32)
    if len(pts) < 3:
        raise ValueError("pool_polygon must contain at least 3 points")
    return pts

def classify_point(point, polygon, near_distance):
    signed = cv2.pointPolygonTest(polygon, (float(point[0]),float(point[1])), True)
    if signed >= 0:
        return "IN WATER"
    if abs(signed) <= near_distance:
        return "NEAR"
    return "SAFE"

def foot_point(box):
    x1,y1,x2,y2 = box
    return ((x1+x2)//2, y2)
