import cv2
import numpy as np
import os
from masking import mask

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
    return points

def process_images(image_dir, label_dir, cover_image_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cover_image = cv2.imread(cover_image_path)
    for image_filename in os.listdir(image_dir):
        if image_filename.endswith(('.jpg', '.png')):
            image_path = os.path.join(image_dir, image_filename)
            label_path = os.path.join(label_dir, os.path.splitext(image_filename)[0] + '.txt')

            if os.path.exists(label_path):
                image = cv2.imread(image_path)
                points = read_points_from_file(label_path)

                if len(points) == 4:
                    masked_image = mask(image, points, cover_image)
                    output_path = os.path.join(output_dir, image_filename)
                    cv2.imwrite(output_path, masked_image)
                    print(f'Successfully processed and saved: {output_path}')
                else:
                    print(f'Error: Incorrect number of points in {label_path}')
            else:
                print(f'Error: Label file not found for {image_path}')

if __name__ == "__main__":
    image_dir = './images'
    label_dir = './labels'
    cover_image_path = 'kntu.jpg'
    output_dir = './output'

    process_images(image_dir, label_dir, cover_image_path, output_dir)