import numpy as np

def calculate_angle(a, b, c):
    # calculate the angle formed by three points
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle


def calculate_distance(landmark_list):
    # calculate the distance between two landmarks
    if len(landmark_list) < 2:
        return
    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    L = np.hypot(x2 - x1, y2 - y1)  # calculate Euclidean distance
    return np.interp(L, [0, 1], [0, 1000])  # normalize the distance
