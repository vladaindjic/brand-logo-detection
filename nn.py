import os

import numpy as np
from keras.initializers import glorot_normal
from keras.layers.core import Dense
from keras.models import Sequential
from sklearn.externals import joblib
# from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split

FEATURE_PATH = '../features/'
class_names = ['Adidas', 'Apple', 'BMW', 'Citroen', 'Cocacola', 'DHL', 'Fedex', 'Ferrari', 'Ford', 'Google',
               'Heineken', 'HP', 'Intel', 'McDonalds',   'Mini', 'Nbc', 'Nike', 'Pepsi', 'Porsche', 'Puma',
               'RedBull', 'Sprite', 'Starbucks', 'Texaco', 'Unicef', 'Vodafone', 'Yahoo', 'None']

class_feat_num_dict = {}
INPUT_NUMBER = 2304


def get_features_number():
    num = 0
    for class_name, feat_num in class_feat_num_dict.iteritems():
        num += feat_num
    return num


def read_features_from_directory(directory_path):
    features = []
    for i, rel_im_path in enumerate(os.listdir(directory_path)):
        # if i % 10 == 0:
        feature = read_feature(os.path.join(directory_path, rel_im_path))
        features.append(feature)
    return features


def read_data(features_path=FEATURE_PATH):
    all_features = []
    for class_name in class_names:
        # if class_name == "None":
        #     continue
        print("Reading features from: %s" % class_name)
        class_features = read_features_from_directory(os.path.join(features_path, class_name))
        class_feat_num_dict[class_name] = len(class_features)
        all_features.extend(class_features)
    return np.array(all_features)


def label_data():
    all_labels = []
    for i, class_name in enumerate(class_names):
        print("Labeling data: %s" % class_name)
        class_feat_num = class_feat_num_dict[class_name]
        for ind in xrange(class_feat_num):
            label = np.zeros(len(class_names))
            label[i] = 1
            all_labels.append(label)
    return np.array(all_labels)


def read_feature(feature_path):
    return joblib.load(feature_path)


def create_nn():
    global INPUT_NUMBER
    nn = Sequential()
    # sledece pokretanje za 128
    print("%d, 256, %d" % (INPUT_NUMBER, len(class_names)))
    nn.add(Dense(1024, input_dim=INPUT_NUMBER, activation='relu'))  #, kernel_initializer=glorot_normal(3)))
    nn.add(Dense(256, activation='relu'))  #, kernel_initializer=glorot_normal(3)))
    # nn.add(Dense(500, activation='relu'))  #, kernel_initializer=glorot_normal(3)))
    # nn.add(Dense(100, activation='relu'))  #, kernel_initializer=glorot_normal(3)))
    # nn.add(Dense(256, input_dim=INPUT_NUMBER, activation='relu'))  #, kernel_initializer=glorot_normal(3)))
    # nn.add(Dense(128, activation='relu', kernel_initializer=glorot_normal(3)))
    # nn.add(Dense(64, activation='softmax'))
    # nn.add(Dense(64, activation='sigmoid'))
    # nn.add(Dense(32, activation='sigmoid'))
    nn.add(Dense(len(class_names), activation='softmax'))  #, kernel_initializer=glorot_normal(3)))
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return nn


def train_nn(nn, x_train, y_train):
    # print(len(x_train[0]))
    # x_train = np.array(x_train, np.float32)
    # print("Before scaling.")
    # print(x_train[0])
    # # proces skaliranja
    # scaler = StandardScaler().fit(x_train)
    # x_train = scaler.transform(x_train)
    # print("After scaling")
    # print(x_train[0])
    print(x_train.shape[0])
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.int64)

    x_train, x_test_and_valid, y_train, y_test_and_valid = train_test_split(x_train, y_train, test_size=0.3, random_state=2)

    x_test, x_valid, y_test, y_valid = train_test_split(x_test_and_valid, y_test_and_valid, test_size=0.33, random_state=2)

    # cuvamo podatke za testiranje
    joblib.dump(x_test, "test.features")
    joblib.dump(y_test, "test.labels")

    nn.fit(x_train, y_train, validation_data=(x_valid, y_valid), epochs=5, batch_size=100, verbose=2, shuffle=True)
    # nn.fit(x_train, y_train, epochs=10, batch_size=1, verbose=2, shuffle=True)
    nn.save("nn.hdf5")
    return nn


def predict(nn):
    rel_feat_path = ["133.feat", "233.feat", "333.feat", "433.feat", "533.feat",
                     "633.feat", "733.feat", "833.feat", "933.feat", "1033.feat"]
    examples = []
    expected_classes = []

    for class_name in class_names:
        for i in xrange(len(rel_feat_path)):
            f = read_feature(os.path.join(FEATURE_PATH, class_name, rel_feat_path[i]))
            examples.append(f)
            expected_classes.append(class_names.index(class_name))

    predicted_classes = nn.predict_classes(np.array(examples))
    # print(predicted_classes)
    expected_classes = np.array(expected_classes)
    # print(expected_classes)
    matches = predicted_classes == expected_classes
    correct_matches = matches.sum()
    print("Correct matches: %d" % correct_matches)
    tries = len(examples)
    print("Tries: %d" % tries)
    print("Percentage: %f" % (float(correct_matches) / tries))


if __name__ == '__main__':
    print("Najnovije")
    # if False:
    if os.path.exists("all.features") and os.path.exists("all.labels"):
        features = joblib.load("all.features")
        labels = joblib.load("all.labels")
        print(features.shape)
        print(labels.shape)
    else:
        features = read_data()
        joblib.dump(features, "all.features")
        labels = label_data()
        joblib.dump(labels, "all.labels")
    neural_net = create_nn()
    neural_net = train_nn(neural_net, features, labels)
    # predict(neural_net)
