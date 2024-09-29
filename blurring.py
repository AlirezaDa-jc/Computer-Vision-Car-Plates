import cv2
import numpy as np
from extract import extract

def blur(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    h_image, w_image, _ = image.shape

    points = np.array([
        [int(points[0][0] * w_image), int(points[0][1] * h_image)],
        [int(points[1][0] * w_image), int(points[1][1] * h_image)],
        [int(points[2][0] * w_image), int(points[2][1] * h_image)],
        [int(points[3][0] * w_image), int(points[3][1] * h_image)]
    ], dtype=np.float32)
    plate_image = extract(image, points)
    plate_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    plate_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

    resized_plate = cv2.resize(plate_image, (plate_width, plate_height))
    blurred_plate = cv2.GaussianBlur(resized_plate, (15, 15), 0)

    dst_points = np.array([
        [0, 0],
        [plate_width - 1, 0],
        [plate_width - 1, plate_height - 1],
        [0, plate_height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(dst_points, points)
    warped_blur = cv2.warpPerspective(blurred_plate, matrix, (w_image, h_image))
    mask = np.zeros((h_image, w_image), dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(points), 255)
    masked_blur = cv2.bitwise_and(warped_blur, warped_blur, mask=mask)
    
    result = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask))
    result = cv2.add(result, masked_blur)

    return result
