from sklearn import datasets
from sklearn import model_selection
from sklearn import naive_bayes
import sys


class HardCodedModel:
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def predict(self, data_test):
        return [0] * len(data_test)


class HardCodedClassifier:
    def fit(self, data, targets):
        return HardCodedModel(data, targets)


def gaussian_classifier(random_seed, split):
    iris = datasets.load_iris()

    iris_data_train, iris_data_test = model_selection.train_test_split(iris.data, test_size=split,
                                                                       random_state=random_seed)
    iris_target_train, iris_target_test = model_selection.train_test_split(iris.target, test_size=split,
                                                                           random_state=random_seed)

    classifier = naive_bayes.GaussianNB()
    iris_model = classifier.fit(iris_data_train, iris_target_train)

    targets_predicted = iris_model.predict(iris_data_test)

    right = sum([1 for x in range(len(targets_predicted)) if targets_predicted[x] == iris_target_test[x]])

    print("Percentage correct for built-in Gaussian classifier: " + str(right / (len(targets_predicted)) * 100))


def hard_coded_classifier(random_seed, split):
    iris = datasets.load_iris()

    iris_data_train, iris_data_test = model_selection.train_test_split(iris.data, test_size=split,
                                                                       random_state=random_seed)
    iris_target_train, iris_target_test = model_selection.train_test_split(iris.target, test_size=split,
                                                                           random_state=random_seed)

    classifier = HardCodedClassifier()
    iris_model = classifier.fit(iris_data_train, iris_target_train)

    targets_predicted = iris_model.predict(iris_data_test)

    right = sum([1 for x in range(len(targets_predicted)) if targets_predicted[x] == iris_target_test[x]])

    print("Percentage correct for hard coded classifier: " + str(int(right / (len(targets_predicted)) * 1000) / 10.0))


if __name__ == "__main__":
    classifier = "h"
    test_split = 0.3
    seed = 42

    if len(sys.argv) == 4:
        if 0 < float(sys.argv[2]) < 1 < int(sys.argv[1]) < 10000 and (sys.argv[3] == 'g' or sys.argv[3] == 'h'):
            seed = int(sys.argv[1])
            test_split = float(sys.argv[2])
            classifier = str(sys.argv[3])
        else:
            print("Command-line arguments were not entered correctly. Please try again")
            if type(sys.argv[1]) is int:
                print(type(sys.argv[1]))
            print(sys.argv[1])
            print(sys.argv[2])
            print(sys.argv[3])
            sys.exit()
    else:
        print("The random seed, the size of the test split, and the type of classifier can be specified.")
        print("The seed must be an int between 1 and 10000. (Default 42)")
        print("The test split must be between 0 and 1. (Default 0.3)")
        print("The classifier can be specified any of these options:")
        print("'h' (hardcoded, default)")
        print("'g' (built-in Gaussian)")

    if classifier == "h":
        hard_coded_classifier(seed, test_split)
    elif classifier == "g":
        gaussian_classifier(seed, test_split)
