import cnn
from keras.models import load_model
from cnn import read_image
from cnn import resize_and_save
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import selectivesearch
from cnn import class_names
import cv2
from nms import nms
import re


TEST_SELECTIVE_SEARCH_IMAGE_PATH = "../test-detection"
DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH = "../demonstration"

clf = load_model("cnn.hdf5")
SHOW_IMAGES = True


HITS_NUM = 0
MISSES_NUM = 0
NUMBER_OF_TRIES = 0


def show_images(img):
    if SHOW_IMAGES:
        plt.imshow(img)
        plt.show()


def get_object_proposals(img, scale=500, sigma=0.9, min_size=10):
    # Selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=scale, sigma=sigma, min_size=min_size)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 500 pixels
        x, y, w, h = r['rect']
        if r['size'] < 2000 or w > 0.95 * img.shape[1] or h > 0.95 * img.shape[0]:
            continue
        # excluding the zero-width or zero-height box
        if r['rect'][2] == 0 or r['rect'][3] == 0:
            continue
        # distorted rects
        if w / h > 5 or h / w > 5:
            continue
        candidates.add(r['rect'])
        # print(r)

    return candidates


def get_images_and_regions(image_path):
    full_img = read_image(image_path)
    regions = get_object_proposals(full_img)
    # print(len(regions))
    region_list = []
    images = []
    for r in regions:
        region_list.append(r)
        x, y, w, h = r
        img = full_img[y:y+h, x:x+w]
        images.append(img)
        region_list.append(r)
    return np.array(images), np.array(region_list)


# def find_logo(image_path):
#     image = read_image(image_path)
#     clone = image.copy()
#     images, regions = get_images_and_regions(image_path)
#     # for img in images:
#     for i in xrange(len(images)):
#         print(i)
#         img = images[i]
#         r = regions[i]
#         res_img = resize_and_save(img, "")
#         res_img = np.array(res_img, np.float32)
#         res_img /= 255
#         pred_class_ind = clf.predict_classes(np.array([res_img]))[0]
#         predictions = clf.predict(np.array([res_img]))[0][pred_class_ind]
#         pred_class_name = class_names[pred_class_ind]
#         print(pred_class_name)
#         # pred_class_name = class_names[]
#         if pred_class_name != "None":
#             x, y, w, h = r
#             print(r)
#             plt.imshow(clone[y:y+h, x:x+w])
#             plt.show()
#             cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 0, 0), thickness=2)
#             plt.imshow(img)
#             plt.show()
#             print("Predicted class: " + pred_class_name)
#     plt.imshow(clone)
#     plt.show()

def find_logo(image_path):
    image = read_image(image_path)
    clone = image.copy()
    regions = get_object_proposals(image)

    detections = []

    for r in regions:
        x, y, w, h = r
        img = image[y:y+h, x:x+w]
        res_img = resize_and_save(img, "")
        res_img = np.array(res_img, np.float32)
        res_img /= 255
        pred_class_ind = clf.predict_classes(np.array([res_img]))[0]
        predictions = clf.predict(np.array([res_img]))[0][pred_class_ind]
        pred_class_name = class_names[pred_class_ind]
        if pred_class_name != "None":
            print("Prediceted classes: %s" % pred_class_name)
            # print(r)
            print("Sure for this: %f" % predictions)
            cv2.rectangle(clone, (x, y), (x+w, y+h), (255, 255, 0), thickness=5)
            detections.append((x, y, predictions, w, h, pred_class_name))
    print("before nms")
    # plt.imshow(clone)
    # plt.show()
    #
    show_images(clone)

    detections = nms(detections, 0.3)
    found = set()
    clone = image.copy()
    for d in detections:
        x, y, _, w, h, class_name = d
        cv2.rectangle(clone, (x, y), (x + w, y + h), (255, 255, 0), thickness=5)
        found.add(class_name)
    print("after nms")
    # plt.imshow(clone)
    # plt.show()
    show_images(clone)

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
    with open(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "labels.txt")) as annot:
        for line in annot.readlines():
            if line.strip() == "":
                continue
            elif "=" in line:
                break
            img_rel_path, class_name = re.split("\\s+", line.strip())
            print("Looking for logo at image: %s" % img_rel_path)
            found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, img_rel_path.strip()))
            # found = set()
            # da li smo pogodili
            hit_num = 0
            # da li smo pogodili
            if class_name in found:
                hit_num += 1
            # koliko smo promasili
            miss_num = len(found) - hit_num
            if miss_num > 0:
                print("****************************************Greska prilikom trazenja loga na slic: " + img_rel_path)
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
                print("****************************************Greska prilikom trazenja loga na slic: " + img_rel_path)
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
    global DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH
    # find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "adidas1.jpg"))
    # find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "yahoo2.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "nike123.jpg"))
    # find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "adidas2.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "bmw3.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "minibmw.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "puma3.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "minibmw1.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "apple-pridodat2.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "ferrari-pridodat3.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "texaco-pridodat2.jpg"))
    # found = find_logo(join(TEST_SELECTIVE_SEARCH_IMAGE_PATH, "redbull-pridodat4.jpg"))


    # DONE: Adidas
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "adidas1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "adidas2.jpg"))
    # print(found)


    # Done: Apple
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "apple1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "apple2.jpg"))
    # print(found)


    # DONE: BMW
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "bmw1.jpg"))
    # print(found)
    #
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "bmw2.jpg"))
    # print(found)


    # DONE: Citroen
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "citroen2.jpg"))
    # print(found)



    # DONE: Cocacola
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "cocacola1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "cocacola2.jpg"))
    # print(found)


    # DONE: DHL
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "dhl1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "dhl2.jpg"))
    # print(found)


    # DONE: fedex
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "fedex4.jpg"))
    # print(found)




    # DONE: Ferrari
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "ferrari1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "ferrari2.jpg"))
    # print(found)


    # DONE: Ford
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "ford1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "ford2.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "ford3.jpg"))
    # print(found)



    # DONE: Google
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "google1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "google2.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "google3.jpg"))
    # print(found)

    # DONE: Heineken
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "heineken3.jpg"))
    # print(found)


    # DONE: HP
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "hp2.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "hp3.jpg"))
    # print(found)


    # DONE: Intel
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "intel1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "intel2.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "intel3.jpg"))
    # print(found)


    # DONE: McDonalds
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "mcdonalds2.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "mcdonalds3.jpg"))
    # print(found)


    #DONE: MINI
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "mini1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "mini2.jpg"))
    # print(found)


    # DONE: NBC
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "nbc1.jpg"))
    # print(found)


    # DONE: Nike
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "nike1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "nike3.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "nike4.jpg"))
    # print(found)

    # DONE: Pepsi
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "pepsi1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "pepsi2.jpg"))
    # print(found)


    # DONE: Porsche
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "porsche1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "porsche2.jpg"))
    # print(found)


    # DONE: Puma
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "puma1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "puma2.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "puma3.jpg"))
    # print(found)



    # DONE: RedBull 1
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "redbull1.jpg"))
    # print(found)



    # DONE: Sprite
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "sprite2.jpg"))
    # print(found)


    # DONE: Starbucks
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "starbucks1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "starbucks2.jpg"))
    # print(found)



    # DONE: Texaco
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "texaco1.jpg"))
    # print(found)
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "texaco3.jpg"))
    # print(found)


    # DONE: Unicef
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "unicef3.jpg"))
    # print(found)



    # DONE: Vodafone
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "vodafone1.jpg"))
    # print(found)



    # DONE: Yahoo
    # found = find_logo(join(DEMONSTRATION_SELECTIVE_SEARCH_IMAGE_PATH, "yahoo1.jpg"))
    # print(found)


    # test_all_images()
    test_all_images_demonstration()


if __name__ == '__main__':
    main()
