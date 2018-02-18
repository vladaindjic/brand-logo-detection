import Augmentor
import os


IMAGES_PATH = "../clean-dataset"
DATASET_PATH = "../dataset"


def aug_one_class(class_name):
    print("\n\n\n====================\nProcessing: %s" % class_name)
    class_dir_path = os.path.join(DATASET_PATH, class_name)
    if not os.path.exists(class_dir_path):
        os.mkdir(class_dir_path)

    print os.path.join(IMAGES_PATH, class_name)

    p = Augmentor.Pipeline(os.path.join(IMAGES_PATH, class_name), output_directory="../"+class_dir_path)
    # p.zoom(probability=0.8, min_factor=1, max_factor=1.1)
    # p.flip_left_right(probability=0.7)
    # p.flip_top_bottom(probability=0.7)
    p.rotate(probability=0.9, max_right_rotation=5, max_left_rotation=5)
    # p.crop_random(0.8, 0.7)
    # p.skew_tilt(0.8)
    # p.skew_left_right(0.8)
    # p.skew_corner(0.8)
    # p.rotate90(probability=0.5)
    # p.rotate180(probability=0.5)
    # p.rotate270(probability=0.5)
    p.sample(500)


def main():
    if not os.path.exists(DATASET_PATH):
        os.makedirs(DATASET_PATH)

    for class_name in os.listdir(IMAGES_PATH):
        if class_name == "None":
            continue
        aug_one_class(class_name)


if __name__ == '__main__':
    main()
