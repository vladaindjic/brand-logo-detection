import os

import cv2
import numpy as np
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


DATASET_PATH = "../dataset"
class_names = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google',
               'Heineken', 'HP', 'Intel', 'McDonalds',   'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma',
               'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'None']


def read_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def resize_and_save(img, path):
    img = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_CUBIC)
    if path != "":
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return img


def read_images_from_directory(directory_path, class_name):
    images = []
    labels = []
    for i, rel_im_path in enumerate(os.listdir(directory_path)):
        # if i % 5 != 0:
        #     continue
        full_image_path =os.path.join(directory_path, rel_im_path)
        # preskacemo visak
        if not ("JPEG" in full_image_path or "jpg" in full_image_path):
            continue
        img = read_image(full_image_path)
        width, height, channels = img.shape
        if width != IMAGE_WIDTH or height != IMAGE_HEIGHT:
            print("Neodgovarajuca velicina: " + full_image_path)
            img = resize_and_save(img, full_image_path)
        images.append(img)
        if img.shape != (128, 128, 3):
            raise Exception("Ova ne valjda: " + full_image_path)
        labels.append(class_names.index(class_name))
    return images, labels


def read_all_images():
    global DATASET_PATH
    all_images = []
    all_labels = []
    for class_dir in os.listdir(DATASET_PATH):
        # if class_dir in class_names: # and class_dir != "BMW":
        if class_dir in class_names:  #and class_dir == "None":
            print("Reading from %s" % class_dir)
            images, labels = read_images_from_directory(os.path.join(DATASET_PATH, class_dir), class_dir)
            all_images.extend(images)
            all_labels.extend(labels)
    return np.array(all_images), np.array(all_labels)


def create_cnn(x_train, y_train):
    print(x_train.shape)
    num_classes = len(class_names)
    model = Sequential()
    d = Conv2D
    model.add(d(32, (5, 5), input_shape=x_train[0].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    # num_pixels = RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT
    model.add(Dense(256, activation='relu'))
    # model.add(Dense(num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # kompajliramo model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_cnn(cnn, x_train, y_train):
    print(x_train.shape)

    x_train = np.array(x_train, np.float32)
    x_train /= 255
    # print(x_train[0])
    y_train = np.array(y_train, np.int32)

    x_train, x_test_and_valid, y_train, y_test_and_valid = train_test_split(x_train, y_train, test_size=0.3,
                                                                            random_state=2)

    x_test, x_valid, y_test, y_valid = train_test_split(x_test_and_valid, y_test_and_valid, test_size=0.33,
                                                        random_state=2)

    # cuvamo podatke za testiranje
    joblib.dump(x_test, "test.features")
    joblib.dump(y_test, "test.labels")

    cnn.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5, batch_size=100, verbose=2, shuffle=True)
    # nn.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2, shuffle=True)
    cnn.save("cnn.hdf5")


def main():
    if os.path.exists("all.images") and os.path.exists("all.labels"):
    # if False:
        images = joblib.load("all.images")
        labels = joblib.load("all.labels")
        print(images.shape)
        print(labels.shape)
    else:
        images, labels = read_all_images()
        labels = np_utils.to_categorical(labels, len(class_names))
        joblib.dump(images, "all.images")
        joblib.dump(labels, "all.labels")
        print(images.shape)
        print(labels.shape)
        # print(len(labels[0]))
    cnn = create_cnn(images, labels)
    train_cnn(cnn, images, labels)


if __name__ == '__main__':
    main()
