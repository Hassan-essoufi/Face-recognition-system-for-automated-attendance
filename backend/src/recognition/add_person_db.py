import os
import cv2
import time
import numpy as np
from random import uniform, randint


def crop_face_from_image(image_path):
    """
    Detect and crop the main face from an image file.
    The cropped face overwrites the original image path.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Invalid image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )

    if len(faces) == 0:
        raise ValueError("No face detected")

    x, y, w, h = max(faces, key=lambda face: face[2] * face[3])

    x1 = max(0, x - 5)
    y1 = max(0, y - 5)
    x2 = min(img.shape[1], x + w + 5)
    y2 = min(img.shape[0], y + h + 5)

    face_crop = img[y1:y2, x1:x2]
    if face_crop.size == 0:
        raise ValueError("Échec du recadrage du visage")

    cv2.imwrite(image_path, face_crop)
    return image_path

def augment_image(image_path, person_name, nb_imgs=150, should_cancel=None):
    """
    Generate multiple augmented images from a single person image.
    """
    output_dir = os.path.join('backend/dataset', person_name)
    os.makedirs(output_dir, exist_ok=True)
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    for i in range(nb_imgs):
        if should_cancel and should_cancel():
            raise RuntimeError("Opération annulée")

        augmented = img.copy()
        
        # Random rotation (-15 to +15 degrees)
        angle = uniform(-15, 15)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1)
        augmented = cv2.warpAffine(augmented, M, (w, h))
        
        # Random horizontal flip
        if randint(0, 1):
            augmented = cv2.flip(augmented, 1)
        
        # Random brightness adjustment (0.8 to 1.2)
        factor = uniform(0.8, 1.2)
        augmented = cv2.convertScaleAbs(augmented, alpha=factor, beta=0)
        
        # Random shift
        tx = randint(-5, 5)
        ty = randint(-5, 5)
        M_shift = np.float32([[1, 0, tx], [0, 1, ty]])
        augmented = cv2.warpAffine(augmented, M_shift, (w, h))
        
        # Save image
        out_path = os.path.join(output_dir, f"aug{i+1}.jpg")
        cv2.imwrite(out_path, augmented)
