import cv2
import numpy as np
import os
from masking import mask
from blurring import blur
from extract import extract

import os
import cv2
import numpy as np
from extract import extract
from blurring import blur

def read_points_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    points = []
    for line in lines:
        coords = list(map(float, line.strip().split()))
        points.append([coords[1], coords[2]])
        points.append([coords[3], coords[4]])
        points.append([coords[5], coords[6]])
        points.append([coords[7], coords[8]])
    return np.array(points, dtype=np.float32)


def process_images(image_dir, label_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + '.txt')

            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                points = read_points_from_file(label_path)

                if len(points) == 4:
                    blurred_image = blur(image, points)
                    output_path = os.path.join(output_dir, image_filename)
                    cv2.imwrite(output_path, blurred_image)

if __name__ == "__main__":
    image_dir = './images'
    label_dir = './labels'
    output_dir = './output_blur'

    process_images(image_dir, label_dir, output_dir)
