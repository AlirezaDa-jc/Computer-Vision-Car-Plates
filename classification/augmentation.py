import imgaug.augmenters as iaa
import cv2
import numpy as np

def augment_image(image):
    aug = iaa.Sequential([
        iaa.Sometimes(0.5, iaa.Affine(
            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},  # scale images to 80-120% of their size, individually per axis
            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},  # translate by -20 to +20 percent (per axis)
            rotate=(-10, 10),  # rotate by -10 to +10 degrees
            shear=(-8, 8),  # shear by -8 to +8 degrees
        )),
        iaa.Sometimes(0.5, iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))),  # add gaussian noise
        iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),  # blur images with a sigma of 0 to 1.0
        iaa.Sometimes(0.5, iaa.Multiply((0.8, 1.2))),  # change brightness, doesn't affect BBs
        iaa.Sometimes(0.5, iaa.ContrastNormalization((0.8, 1.2))),  # improve or worsen the contrast
        iaa.Sometimes(0.5, iaa.PerspectiveTransform(scale=(0.01, 0.15))),  # perspective transform
        iaa.Sometimes(0.5, iaa.Crop(percent=(0, 0.1))),  # random crops
        iaa.Sometimes(0.5, iaa.LinearContrast((0.75, 1.5))),  # linear contrast change
        iaa.Sometimes(0.5, iaa.Add((-10, 10))),  # change brightness of images (by -10 to 10 of original value)
        iaa.Sometimes(0.5, iaa.JpegCompression(compression=(70, 99)))  # add JPEG compression artifacts
    ])

    image_aug = aug(image=image)
    return image_aug

def generate_dataset(num_samples, save_path):
    for i in range(num_samples):
        image = generate_license_plate_image()  # Generate synthetic license plate image
        augmented_image = augment_image(image)
        cv2.imwrite(f"{save_path}/augmented_plate_{i}.jpg", augmented_image)

def generate_license_plate_image():
    # Dummy function to simulate license plate generation
    image = np.ones((64, 128, 3), dtype=np.uint8) * 255
    cv2.putText(image, 'ABC1234', (5, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
    return image

if __name__ == "__main__":
    save_path = "./augmented_plates"
    num_samples = 1000
    generate_dataset(num_samples, save_path)
