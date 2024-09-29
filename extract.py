import cv2
import numpy as np

def extract(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    plate_width = 450
    plate_height = int(plate_width / 4.5)

    dst_points = np.array([
        [0, 0],
        [plate_width, 0],
        [plate_width, plate_height],
        [0, plate_height]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(points, dst_points)

    extracted_plate = cv2.warpPerspective(image, matrix, (plate_width, plate_height))

    return extracted_plate
