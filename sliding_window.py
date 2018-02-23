# Import the required modules
import re
from os.path import join

import cv2
import matplotlib.pyplot as plt
import numpy as np
# from skimage.feature import hog
from keras.models import load_model
from skimage.transform import pyramid_gaussian

from hog import HogApplier
from nms import nms

TEST_IMAGES_PATH = "test_images/"
class_names = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google',
               'Heineken', 'HP', 'Intel', 'McDonalds', 'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma',
               'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'None']

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
SHOW_IMAGES = True
TEST_SLIDING_WINDOW_PATH = "../test-detection"
DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH = "../demonstration"

HITS_NUM = 0
MISSES_NUM = 0
NUMBER_OF_TRIES = 0


def read_image(image_path):
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)


def greyscale_image(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0)
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])





def show_image(img):
    if SHOW_IMAGES:
        plt.imshow(img)
        plt.show()


def find_logo(image_path):
    # Read the image
    img = read_image(image_path)
    show_image(img)
    # img = cv2.resize()
    img_gs = greyscale_image(img)

    height, width, channels = img.shape
    print("******************************************************")
    print(width, height)

    min_wdw_sz = (width//4, height//4)
    step_size = (width//16, height//16)

    downscale = 1.25
    visualize_det = False

    # Load the classifier
    clf = load_model("nn.hdf5")

    # List to store the detections
    detections = []
    # The current scale of the image
    scale = 0
    hog_applier = HogApplier(128, hist_bins=128, small_size=40, orientations=9, pix_per_cell=8,
                             cell_per_block=1)
    i = 0
    # Downscale the image and iterate
    for im_scaled in pyramid_gaussian(img_gs, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
            break
        for (x, y, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            # resize slike
            im_window_gs = cv2.resize(np.asarray(im_window * 255, np.uint8), (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_CUBIC)
            i += 1
            # izracunavanje deskriptora
            fd = hog_applier.get_features_from_image(im_window_gs)
            # predikcija
            prediction = clf.predict(np.array([fd]))
            pred_class = clf.predict_classes(np.array([fd]))
            pred_index = pred_class[0]
            pred_class_name = class_names[pred_index]
            confidence = prediction[0][pred_index]
            print(pred_class_name)
            # da li je nesto pronadjeno
            if pred_class_name != "None":
                print("Detection:: Location -> ({}, {})".format(x, y))
                print("Scale ->  {} | Confidence Score {} \n".format(scale, confidence))
                detections.append((x, y, confidence,
                                   int(min_wdw_sz[0] * (downscale ** scale)),
                                   int(min_wdw_sz[1] * (downscale ** scale)),
                                   pred_class_name))
                cd.append(detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = im_scaled.copy()
                    for x1, y1, _, _, _, _ in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                                                        im_window.shape[0]), (0, 0, 0), thickness=2)
                    plt.imshow(clone, 'gray')
                    plt.show()
    #
        # Move the the next scale
        scale += 1
    #
    # Display the results before performing NMS
    clone = img.copy()
    for (x_tl, y_tl, _, w, h, _) in detections:
        # Draw the detections
        cv2.rectangle(img, (x_tl, y_tl), (x_tl + w, y_tl + h), (255, 0, 0), thickness=2)
    show_image(img)
    print(len(detections))

    # Perform Non Maxima Suppression
    detections = nms(detections, 0.1)
    print(len(detections))

    found = set()
    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h, pred_class_name) in detections:
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl + w, y_tl + h), (255, 0, 0), thickness=2)
        # cv2.putText(clone, pred_class_name, (x_tl, y_tl), cv2.FONT_ITALIC, 4, (0, 0, 0))
        print("Ovo je pronadjeno: " + pred_class_name)
        found.add(pred_class_name)
    show_image(clone)
    return found


def test_all_images():
    global SHOW_IMAGES
    global HITS_NUM
    global MISSES_NUM
    global NUMBER_OF_TRIES
    HITS_NUM = 0
    MISSES_NUM = 0
    NUMBER_OF_TRIES = 0
    SHOW_IMAGES = False
    with open(join(TEST_SLIDING_WINDOW_PATH, "labels.txt")) as annot:
        for line in annot.readlines():
            if line.strip() == "":
                continue
            elif "=" in line:
                break
            img_rel_path, class_name = re.split("\\s+", line.strip())
            print("Looking for logo at image: %s" % img_rel_path)
            found = find_logo(join(TEST_SLIDING_WINDOW_PATH, img_rel_path.strip()))
            # found = set()
            # da li smo pogodili
            hit_num = 0
            # da li smo pogodili
            if class_name in found:
                hit_num += 1
            # koliko smo promasili
            miss_num = len(found) - hit_num
            if miss_num > 0:
                print("****************************************Greska prilikom trazenja loga na slici: " + img_rel_path)
            print("=" * 33)
            print("Number of hits for %s is %d" % (img_rel_path, hit_num))
            print("Number of misses for %s is %d" % (img_rel_path, miss_num))
            print("=" * 33)
            HITS_NUM += hit_num
            MISSES_NUM += miss_num
            NUMBER_OF_TRIES += 1
    print("\n\n\n")
    print("Number of tries: %d" % NUMBER_OF_TRIES)
    print("Number of hits: %d" % HITS_NUM)
    print("Number of misses: %d" % MISSES_NUM)


def test_all_images_demonstration():
    global SHOW_IMAGES
    global HITS_NUM
    global MISSES_NUM
    global NUMBER_OF_TRIES
    HITS_NUM = 0
    MISSES_NUM = 0
    NUMBER_OF_TRIES = 0
    SHOW_IMAGES = False
    with open(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "labels.txt")) as annot:
        for line in annot.readlines():
            if line.strip() == "":
                continue
            elif "=" in line:
                break
            img_rel_path, class_name = re.split("\\s+", line.strip())
            print("Looking for logo at image: %s" % img_rel_path)
            found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, img_rel_path.strip()))
            # found = set()
            # da li smo pogodili
            hit_num = 0
            # da li smo pogodili
            if class_name in found:
                hit_num += 1
            # koliko smo promasili
            miss_num = len(found) - hit_num
            if miss_num > 0:
                print("****************************************Greska prilikom trazenja loga na slici: " + img_rel_path)
            print("=" * 33)
            print("Number of hits for %s is %d" % (img_rel_path, hit_num))
            print("Number of misses for %s is %d" % (img_rel_path, miss_num))
            print("=" * 33)
            HITS_NUM += hit_num
            MISSES_NUM += miss_num
            NUMBER_OF_TRIES += 1
    print("\n\n\n")
    print("Number of tries: %d" % NUMBER_OF_TRIES)
    print("Number of hits: %d" % HITS_NUM)
    print("Number of misses: %d" % MISSES_NUM)


def main():
    # test_all_images()
    # slika koju uspe da prepozna
    # find_logo(join(TEST_SLIDING_WINDOW_PATH, "fedex1.jpg"))
    test_all_images_demonstration()


if __name__ == "__main__":
    main()


    # fedex1.jpg daje iole normalan rezultat
