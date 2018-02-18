from keras.models import load_model
from sklearn.externals import joblib
import numpy as np


def main():
    print("Ovo bi trebalo da bude novi prediktor")
    test_features = joblib.load("test.features")
    test_labels = joblib.load("test.labels")
    print(len(test_features))
    print(len(test_labels))
    print(test_features.shape)
    print(test_labels.shape)
    test_classes = np.array([np.argmax(label) for label in test_labels])
    nn = load_model("nn.hdf5")
    print("Prediction starts!")
    pred_classes = nn.predict_classes(test_features)
    print("Prediction finishes!")
    test_cases_num = len(test_features)
    matches_num = sum(test_classes == pred_classes)
    success = float(matches_num) / test_cases_num
    success_percentage = success * 100
    print("Number of test cases: %d" % test_cases_num)
    print("Number of matches: %d" % matches_num)
    print("Success: %f%%" % success_percentage)


if __name__ == '__main__':
    main()
