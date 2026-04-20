import cv2
import os
import numpy as np
import random

# Paths
input_folder = "C:/Users/Computer/Downloads/receipt_project/thaidata"
output_folder = "C:/Users/Computer/Downloads/receipt_project/augmented_thaireceipts"

os.makedirs(output_folder, exist_ok=True)

image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png"))]

TOTAL_TARGET = 600
num_images = len(image_files)

# divide across rotations (4 types)
per_rotation = TOTAL_TARGET // (num_images * 4)

count = 0

#  Rotation function (no crop)
def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, M, (new_w, new_h))


#  Affine transform (shift + shear + zoom)
def affine_transform(image):
    h, w = image.shape[:2]

    pts1 = np.float32([[0,0],[w,0],[0,h]])
    
    shift = random.randint(-20, 20)
    shear = random.uniform(-0.2, 0.2)
    scale = random.uniform(0.8, 1.2)

    pts2 = np.float32([
        [0+shift, 0],
        [w+shift, 0],
        [int(w*shear), h]
    ])

    M = cv2.getAffineTransform(pts1, pts2)
    M[0,0] *= scale
    M[1,1] *= scale

    return cv2.warpAffine(image, M, (w, h))


for img_name in image_files:
    img_path = os.path.join(input_folder, img_name)
    image = cv2.imread(img_path)

    if image is None:
        print("Failed:", img_name)
        continue

    base_name = img_name.split('.')[0]

    # 🎯 Required rotations
    rotations = [
        ("rot45_cw", rotate_image(image, -45)),
        ("rot45_ccw", rotate_image(image, 45)),
        ("rot90_cw", cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        ("rot_random", rotate_image(image, random.randint(-15, 15)))  # small rotation
    ]

    for label, img in rotations:

        for i in range(per_rotation):

            aug = img.copy()

            # 🔆 Brightness
            alpha = random.uniform(0.7, 1.3)
            beta = random.randint(-30, 30)
            aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

            # 🌫️ Blur
            if random.random() < 0.5:
                aug = cv2.GaussianBlur(aug, (5, 5), 0)

            # 🔄 Flip (horizontal only)
            if random.random() < 0.4:
                aug = cv2.flip(aug, 1)

            # 📐 Shift + Shear + Zoom
            if random.random() < 0.7:
                aug = affine_transform(aug)

            # 💾 Save
            filename = f"{base_name}_{label}_{i}.jpg"
            cv2.imwrite(os.path.join(output_folder, filename), aug)

            count += 1


print("✅ Done! Total images:", count)