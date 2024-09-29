import cv2
import numpy as np

def mask(image: np.ndarray, points: np.ndarray, cover: np.ndarray) -> np.ndarray:
    h_image, w_image, _ = image.shape

    points = np.array([
        [int(points[0][0] * w_image), int(points[0][1] * h_image)],
        [int(points[1][0] * w_image), int(points[1][1] * h_image)],
        [int(points[2][0] * w_image), int(points[2][1] * h_image)],
        [int(points[3][0] * w_image), int(points[3][1] * h_image)]
    ], dtype=np.float32)

    h_cover, w_cover, _ = cover.shape

    dst_points = np.array([
        [points[0][0], points[0][1]],
        [points[1][0], points[1][1]],
        [points[2][0], points[2][1]],
        [points[3][0], points[3][1]]
    ], dtype=np.float32)

    src_points = np.array([
        [0, 0],
        [w_cover, 0],
        [w_cover, h_cover],
        [0, h_cover]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_cover = cv2.warpPerspective(cover, matrix, (w_image, h_image))

    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(dst_points), (255, 255, 255))
    mask_inv = cv2.bitwise_not(mask)

    img_bg = cv2.bitwise_and(image, mask_inv)
    cover_fg = cv2.bitwise_and(warped_cover, mask)
    result = cv2.add(img_bg, cover_fg)

    return result
