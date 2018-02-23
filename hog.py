import cv2
import numpy as np
from sklearn.externals import joblib
import os
from skimage.feature import hog
import matplotlib.pyplot as plt

IMAGE_PATH = "1.jpg"
DATASET_PATH = "../dataset"

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


def read_image(image_path):
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    width, height, channels = img.shape
    if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
        img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
        # plt.imshow(img)
        # plt.show()
        # print(img.shape)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


class HogApplier(object):
    hog_applier = None

    @staticmethod
    def get_instance():
        if HogApplier.hog_applier is None:
            HogApplier.hog_applier = HogApplier(128, hist_bins=128, small_size=40, orientations=9, pix_per_cell=8,
                                                cell_per_block=1)
        return HogApplier.hog_applier

    def __init__(self, size, hist_bins, small_size, orientations=9, pix_per_cell=8, cell_per_block=2,
                 hist_range=None, scaler=None, classifier=None, window_sizes=None, window_rois=None,
                 warped_size=None, transform_matrix=None, pix_per_meter=None):
        self.size = size
        self.small_size = small_size
        self.hist_bins = hist_bins
        self.hist_range = (0, 256)
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.orientations = orientations
        self.scaler = scaler
        self.cls = classifier
        self.num_cells = self.size // self.pix_per_cell
        self.num_blocks = self.num_cells - (self.cell_per_block - 1)
        self.num_features = self.calc_num_features()
        if hist_range is not None:
            self.hist_range = hist_range

        self.window_sizes = window_sizes
        self.window_rois = window_rois
        self.cars = []
        self.first = True
        self.warped_size = warped_size
        self.transformation_matrix = transform_matrix
        self.pix_per_meter = pix_per_meter

    def calc_num_features(self):
        return self.small_size ** 2 * 3 + self.hist_bins * 3 + 3 * self.num_blocks ** 2 * self.cell_per_block ** 2 * self.orientations

    def get_features(self, image_path):
        image = read_image(image_path)
        # Grayscale features
        features = hog(image, self.orientations, block_norm='L2-Hys',
                       pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=False,
                       feature_vector=True, visualise=False)

        # plt.imshow(features[1])
        # plt.show()
        # return np.hstack((img_feature.ravel(), hist_l[0], hist_u[0], hist_v[0], features_l, features_u, features_v))
        if len(features) != 2304:
            raise Exception("Neodgovarajuci format descriptora")
        return features

    def get_features_from_image(self, image):
        features = hog(image, self.orientations, block_norm='L2-Hys',
                       pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block), transform_sqrt=False,
                       feature_vector=True, visualise=False)
        # plt.imshow(features[1])
        # plt.show()
        # return features[0]
        return features


def show_image(image):
    plt.imshow(image, 'gray')
    plt.show()


def get_feature_path_and_create_if_needed(image_path):
    feature_path = image_path.replace("dataset", "features").replace(".jpg", ".feat").replace(".JPEG", ".feat")
    feature_directory = feature_path[:feature_path.rfind("/")]
    if not os.path.exists(feature_directory):
        print("Feature directory created: %s" % feature_directory)
        os.makedirs(feature_directory)
    return feature_path


def save_feature(hog_descriptor, feature_path):
    joblib.dump(hog_descriptor, feature_path)


def calculate_and_save_hog_descriptor(image_path, hog_applier):
    # if "BMW" in image_path:
    hog_descriptor = hog_applier.get_features(image_path)
    feature_path = get_feature_path_and_create_if_needed(image_path)
    save_feature(hog_descriptor, feature_path)


def find_image_and_apply_hog(path):
    hog_applier = HogApplier.get_instance()
    if os.path.isdir(path):
        for dir_item in os.listdir(path):
            find_image_and_apply_hog(os.path.join(path, dir_item))
    elif path.endswith(".jpg") or path.endswith(".JPEG"):
        # ovde dodaj resize
        calculate_and_save_hog_descriptor(path, hog_applier)


if __name__ == '__main__':
    # hog_applier = HogApplier.get_instance()
    # features = hog_applier.get_features("adidas333.JPEG")
    # print(len(features))
    # features = hog_applier.get_features("none1.JPEG")
    # print(len(features))
    # features = hog_applier.get_features("mini1.JPEG")
    # print(len(features))
    # features = hog_applier.get_features("../dataset_svm/None/1.jpg")
    # print(len(features))
    find_image_and_apply_hog(DATASET_PATH)
