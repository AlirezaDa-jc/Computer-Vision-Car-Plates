import os
import cv2
import numpy as np

def augment(image: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, cols, _ = image.shape

    # Random rotation
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))

    # Update points
    ones = np.ones(shape=(len(points), 1))
    points_ones = np.hstack([points, ones])
    points = M.dot(points_ones.T).T

    # Random translation
    tx = np.random.uniform(-10, 10)
    ty = np.random.uniform(-10, 10)
    T = np.float32([[1, 0, tx], [0, 1, ty]])
    image = cv2.warpAffine(image, T, (cols, rows))

    # Update points
    points = points + [tx, ty]

    # Random scaling
    scale = np.random.uniform(0.8, 1.2)
    image = cv2.resize(image, None, fx=scale, fy=scale)
    points = points * scale

    return image, points

def resize(image: np.ndarray, points: np.ndarray, size=(128, 128)) -> tuple[np.ndarray, np.ndarray]:
    original_size = image.shape[:2]
    image = cv2.resize(image, size)
    points[:, 0] = points[:, 0] * size[0] / original_size[1]
    points[:, 1] = points[:, 1] * size[1] / original_size[0]
    return image, points

def load(dataset_dir: str) -> tuple[np.ndarray, np.ndarray]:
    images = []
    points = []

    images_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')

    for filename in os.listdir(images_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_dir, filename)
            image = cv2.imread(image_path)

            label_path = os.path.join(labels_dir, os.path.splitext(filename)[0] + '.txt')
            with open(label_path, 'r') as f:
                label_data = f.read().strip().split()
                coords = list(map(float, label_data[1:]))  # Skip class label
                coords = np.array(coords).reshape(-1, 2) * np.array([image.shape[1], image.shape[0]])  # Convert to absolute coordinates

            image, coords = augment(image, coords)
            image, coords = resize(image, coords)
            
            image = image / 255.0  # Normalize image to [0, 1]

            images.append(image)
            points.append(coords.flatten())

    X = np.array(images)
    y = np.array(points)
    
    return X, y
