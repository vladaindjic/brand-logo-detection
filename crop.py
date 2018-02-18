"""
    Operacije na slikama:
    1. crop_logos
    2. aug_pos
    3. aug_scale
    4. aug_rot

Parametri slike
144503924.jpg Adidas 1 38 12 234 142

"""

import cv2
from itertools import product
import os
import matplotlib.pyplot as plt

X1 = 38
Y1 = 12
X2 = 234
Y2 = 142
COORDINATES = (X1, Y1, X2, Y2)
IMAGE_PATH = "144503924.jpg"
OPENCV_PATH = "opencv/"
PILLOW_PATH = "pillow/"
DATA_AUG_POS_SHIFT_MIN = -2
DATA_AUG_POS_SHIFT_MAX = 2
DATA_AUG_SCALES = [0.9, 1.1]
DATA_AUG_ROT_MIN = -15
DATA_AUG_ROT_MAX = 15
FLICKR_27_PATH = "../flickr-27/flickr_logos_27_dataset/"
FLICKR_27_PATH_IMAGES = os.path.join(FLICKR_27_PATH, "flickr_logos_27_dataset_images/")
CROPPED_PATH = "../cropped_images/"
DATASET_PATH = "../dataset"
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32


RESIZE_IMAGE_WIDTH = 128
RESIZE_IMAGE_HEIGHT = 128

SUN2012_IMAGES_PATH = "../SUN2012/Images/"
logo_aug_images_dict = {}


class MyRect(object):
    def __init__(self, (x1, y1, x2, y2)):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.cx = (x1 + x2) // 2
        self.cy = (y1 + y2) // 2
        self.width = abs(x2 - x1)
        self.height = abs(y2 - y1)


def read_image(image_path):
    try:
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    except:
        print(image_path)


def crop_image(image, coordinates=COORDINATES):
    x1, y1, x2, y2 = coordinates
    return image[y1:y2, x1:x2]


# # Menanje centara ili tako nesto
# def aug_pos(image, coordinates=COORDINATES):
#     aug_pos_images = []
#     rect = MyRect(coordinates)
#     for i, (sx, sy) in enumerate(
#             product(
#             range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX),
#             range(DATA_AUG_POS_SHIFT_MIN, DATA_AUG_POS_SHIFT_MAX))
#     ):
#         cx = rect.cx + sx
#         cy = rect.cy + sy
#         cropped_image = crop_image(image, (cx - rect.width // 2, cy - rect.height // 2,
#                       cx + rect.width // 2, cy + rect.height // 2))
#         aug_pos_images.append(cropped_image)
#         # cv2.imwrite(os.path.join(OPENCV_PATH, 'aug_pos%d.jpg' % i), cropped_image)
#     return aug_pos_images


# # Skaliranje
# def aug_scale(image, coordinates=COORDINATES):
#     rect = MyRect(coordinates)
#     aug_scaled_images = []
#     for i, s in enumerate(DATA_AUG_SCALES):
#         w = int(rect.width * s)
#         h = int(rect.height * s)
#         cropped_image = crop_image(image, (rect.cx - w // 2, rect.cy - h // 2,
#                                         rect.cx + w // 2, rect.cy + h // 2))
#         aug_scaled_images.append(cropped_image)
#         # cv2.imwrite(os.path.join(OPENCV_PATH, "scale%d.jpg" % i), cropped_image)
#     return aug_scaled_images


# def rotate_image(image, angle):
#     rows, cols = image.shape[:2]
#     rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
#     return cv2.warpAffine(image, rotation_matrix, (cols, rows))


# # Rotiranje
# def aug_rot(image, coordinates=COORDINATES):
#     aug_rot_images = []
#
#     rect = MyRect(coordinates)
#     for i, r in enumerate(range(DATA_AUG_ROT_MIN, DATA_AUG_ROT_MAX)):
#         rotated_image = rotate_image(image, r)
#         # cropped_image = crop_image(rotated_image,
#         #                            (rect.cx - rect.width // 2, rect.cy - rect.height // 2,
#         #                             rect.cx + rect.width // 2, rect.cy + rect.height // 2))
#         aug_rot_images.append(rotated_image)
#         # cv2.imwrite(os.path.join(OPENCV_PATH, "rotated%d.jpg" % i), cropped_image)
#     return aug_rot_images


# Cuvanje slika
def save_image(image, file_name):
    cv2.imwrite(file_name, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def is_small(coordinates):
    rect = MyRect(coordinates)
    if rect.height <= 0 or rect.width <= 0:
        return True
    return False


# def aug_image(image_path=IMAGE_PATH, coordinates=COORDINATES):
#     # proveravamo da li uopste treba da se posmatra slika
#     if is_small(coordinates):
#         return []
#     aug_images = []
#     image = read_image(image_path)
#     cropped_image = crop_image(image, coordinates)
#     aug_images.append(cropped_image)
#     # aug_images.extend(aug_pos(image, coordinates))
#     # aug_images.extend(aug_scale(image, coordinates))
#     # aug_images.extend(aug_rot(cropped_image, coordinates))
#     # print(len(aug_images))
#     return aug_images


def get_logo_information(text_line):
    image_path, class_name, _, x1, y1, x2, y2 = text_line.strip().split(" ")
    return image_path, class_name, (int(x1), int(y1), int(x2), int(y2))


def resize_image(image):
    return cv2.resize(image, (RESIZE_IMAGE_WIDTH, RESIZE_IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)


# def save_logo_images(images, class_name):
#     if not os.path.exists(CROPPED_PATH):
#         os.makedirs(CROPPED_PATH)
#
#     if not os.path.exists(os.path.join(CROPPED_PATH, class_name)):
#         os.makedirs(os.path.join(CROPPED_PATH, class_name))
#
#     for image in images:
#         if image.size <= 0:
#             continue
#         resized_image = resize_image(image)
#         save_image(resized_image, os.path.join(CROPPED_PATH, class_name, "%d.jpg" % logo_aug_images_dict[class_name]))
#         logo_aug_images_dict[class_name] += 1


def save_and_crop(image_path, coordinates, class_name):
    # da li je odgovarajuca velicina
    if is_small(coordinates):
        return []
    # kropujemo sliku
    image = read_image(image_path)
    cropped_image = crop_image(image, coordinates)

    if not os.path.exists(CROPPED_PATH):
        os.makedirs(CROPPED_PATH)

    if not os.path.exists(os.path.join(CROPPED_PATH, class_name)):
        print("Creating directory for: %s" % class_name)
        os.makedirs(os.path.join(CROPPED_PATH, class_name))

    resized_image = resize_image(cropped_image)
    save_image(resized_image, os.path.join(CROPPED_PATH, class_name, "%d.jpg" % logo_aug_images_dict[class_name]))
    logo_aug_images_dict[class_name] += 1


def process_logo_images():
    print("Processing logo images.")
    with open(os.path.join(FLICKR_27_PATH + 'flickr_logos_27_dataset_training_set_annotation.txt')) as f_train:
        for line in f_train.readlines():
            image_path, class_name, coordinates = get_logo_information(line)
            # print("%s,%s,%s" % (image_path, class_name, coordinates))
            if class_name not in logo_aug_images_dict:
                logo_aug_images_dict[class_name] = 1
            save_and_crop(os.path.join(FLICKR_27_PATH_IMAGES, image_path), coordinates, class_name)
            # save_logo_images(aug_image(os.path.join(FLICKR_27_PATH_IMAGES, image_path), coordinates), class_name)


def crop_and_resize_non_logo_image(image_path, image_number):
    image = read_image(image_path)
    if image is None:
        print(image_path)
        return
    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2
    if width >= IMAGE_WIDTH and height >= IMAGE_HEIGHT:
        cropped_image = crop_image(image,
                                   (cx - IMAGE_WIDTH // 2, cy - IMAGE_HEIGHT // 2,
                                    cx + IMAGE_WIDTH // 2, cy + IMAGE_HEIGHT // 2))
        resized_image = resize_image(cropped_image)
        save_image(resized_image, os.path.join(DATASET_PATH, "None", "%d.JPEG" % (image_number)))
    # eventualno ukloniti ovaj deo
    else:
        print("Ova slika ne valja: %s", image_path)
        # resized_image = resize_image(image)
        # save_image(resized_image, os.path.join(DATASET_PATH, "None", "%d.jpg" % (image_number * 2 + 2)))


def find_and_process_non_logo_image(path, image_number):
    if os.path.isdir(path):
        for dir_item in os.listdir(path):
            image_number = find_and_process_non_logo_image(os.path.join(path, dir_item), image_number)
    elif path.endswith(".jpg"):
        if image_number % 2 == 0:
            crop_and_resize_non_logo_image(path, image_number)
        image_number += 1
    return image_number


def process_non_logo_images():
    print("Processing non logo images.")
    if not os.path.exists(os.path.join(DATASET_PATH, "None")):
        os.makedirs(os.path.join(DATASET_PATH, "None"))
    if not os.path.exists(SUN2012_IMAGES_PATH):
        return
    image_number = find_and_process_non_logo_image(SUN2012_IMAGES_PATH, 0)
    print(image_number)


if __name__ == '__main__':
    process_logo_images()
    # process_non_logo_images()
    # read_image("../SUN2012/Images/g/garage/indoor/sun_akbocuwclkxqlofx.jpg")

